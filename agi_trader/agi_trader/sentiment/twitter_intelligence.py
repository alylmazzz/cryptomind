"""
Twitter/X İstihbarat Motoru (NLP / Sentiment Mühendisi rolü).

Spec: kritik hesapları takip et; her tweet için etki puanı, geçmiş doğruluk,
manipülasyon/bot olasılığı, sentiment (fear/greed/bullish/bearish/nötr),
volatilite etkisi, retweet hızı ve trend olasılığı hesapla; sonra coin başına
ağırlıklı toplam sentiment üret -> ayı/boğa sinyali.

Bu modül üç katmanlı çalışır:
  1) tweepy + TWITTER_BEARER_TOKEN varsa  -> CANLI tweet çekimi
  2) transformers (cryptobert/finbert) varsa -> model tabanlı sentiment
  3) hiçbiri yoksa -> kural tabanlı (lexicon) sentiment + nötr fallback
Böylece sistem her koşulda çalışır, anahtar varsa otomatik güçlenir.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..core.models import LayerVote
from .accounts import all_accounts, account_stats
from .causal import analyze_causal

# Opsiyonel bağımlılıklar
try:
    import tweepy  # type: ignore
    _HAS_TWEEPY = True
except Exception:
    _HAS_TWEEPY = False

from ..core.light import LIGHT_MODE

if LIGHT_MODE:
    # hafif mod: transformers torch'u da yükler (~400 MB) — kural tabanlı
    # sözlük fallback'i kullanılır.
    _HAS_TRANSFORMERS = False
else:
    try:
        from transformers import pipeline  # type: ignore
        _HAS_TRANSFORMERS = True
    except Exception:
        _HAS_TRANSFORMERS = False


# Kural tabanlı sözlük (model yoksa)
BULLISH_WORDS = {
    "moon", "bullish", "buy", "long", "breakout", "pump", "rally", "surge",
    "accumulate", "support", "bottom", "undervalued", "ath", "green", "up",
    "yükseliş", "boğa", "al", "kırılım", "destek", "dip",
}
BEARISH_WORDS = {
    "dump", "bearish", "sell", "short", "crash", "drop", "breakdown", "fear",
    "liquidation", "resistance", "top", "overvalued", "red", "down", "rug",
    "düşüş", "ayı", "sat", "çöküş", "direnç", "tepe",
}
HYPE_WORDS = {"100x", "guaranteed", "lambo", "easy money", "millionaire", "🚀🚀🚀", "financial advice"}

KNOWN_COINS = {
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC",
    "LINK", "UNI", "ATOM", "LTC", "NEAR", "OP", "ARB", "SUI", "SEI", "TIA", "INJ",
}
COIN_NAMES = {"BITCOIN": "BTC", "ETHEREUM": "ETH", "SOLANA": "SOL", "RIPPLE": "XRP", "CARDANO": "ADA"}


@dataclass
class TweetSignal:
    handle: str
    weight: float
    sentiment_score: float      # -1..1
    impact: float               # 0..100
    manipulation: float         # 0..1
    coins: List[str] = field(default_factory=list)
    text: str = ""
    causal_dir: int = 0         # +1 boğa / -1 ayı / 0
    causal_mag: float = 0.0     # 0..1
    causal_reasons: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)


class TwitterIntelligence:
    def __init__(self, config):
        self.config = config
        self.accounts = all_accounts()
        self.bearer = config.secret("TWITTER_BEARER_TOKEN")
        self._tw_client = None
        self._sent_model = None
        self._model_loaded = False     # tembel yükleme bayrağı
        self._init_clients()

    def _init_clients(self):
        # NOT: Ağır transformers/cryptobert modeli BAŞLANGIÇTA yüklenmez.
        # Sunucu/CLI anında açılsın diye model yalnızca canlı tweet işlenirken
        # (ilk _score_text çağrısında) tembel yüklenir. Bearer yoksa hiç yüklenmez.
        if _HAS_TWEEPY and self.bearer:
            try:
                self._tw_client = tweepy.Client(bearer_token=self.bearer, wait_on_rate_limit=True)
            except Exception:
                self._tw_client = None

    def _ensure_model(self):
        """Sentiment modelini ilk gerçek ihtiyaçta tembel yükle."""
        if self._model_loaded:
            return
        self._model_loaded = True
        if not _HAS_TRANSFORMERS:
            return
        try:
            self._sent_model = pipeline("sentiment-analysis", model="ElKulako/cryptobert")
        except Exception:
            try:
                self._sent_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            except Exception:
                self._sent_model = None

    @property
    def live(self) -> bool:
        return self._tw_client is not None

    # ----------------------------------------------------------- sentiment
    def _score_text(self, text: str) -> float:
        """Tek tweet -> -1..1 sentiment."""
        self._ensure_model()
        if self._sent_model is not None:
            try:
                res = self._sent_model(text[:512])[0]
                label = res["label"].lower()
                score = float(res["score"])
                if "bull" in label or "positive" in label:
                    return score
                if "bear" in label or "negative" in label:
                    return -score
                return 0.0
            except Exception:
                pass
        # kural tabanlı
        t = text.lower()
        b = sum(1 for w in BULLISH_WORDS if w in t)
        s = sum(1 for w in BEARISH_WORDS if w in t)
        total = b + s
        return 0.0 if total == 0 else (b - s) / total

    def _extract_coins(self, text: str) -> List[str]:
        up = text.upper()
        found = set(re.findall(r"\$([A-Z]{2,6})", up))
        for c in KNOWN_COINS:
            if re.search(rf"\b{c}\b", up):
                found.add(c)
        for name, sym in COIN_NAMES.items():
            if name in up:
                found.add(sym)
        return list(found)

    def _manipulation(self, text: str, weight: float) -> float:
        t = text.lower()
        signals = 0.0
        if sum(1 for w in HYPE_WORDS if w in t) >= 1:
            signals += 0.4
        if re.search(r"bit\.ly|tinyurl|ref=|aff=|t\.me/", t):
            signals += 0.4
        if weight < 6 and ("🚀" in text or "100x" in t):
            signals += 0.2
        return min(1.0, signals)

    # ----------------------------------------------------------- live fetch
    def fetch_recent(self, handle: str, max_results: int = 5) -> List[str]:
        if not self.live:
            return []
        try:
            user = self._tw_client.get_user(username=handle)
            if not user or not user.data:
                return []
            tweets = self._tw_client.get_users_tweets(
                user.data.id, max_results=max_results,
                exclude=["retweets", "replies"])
            return [t.text for t in (tweets.data or [])]
        except Exception:
            return []

    # ----------------------------------------------------------- sosyal ısı (çoklu-parite)
    def scan_social_heat(self, max_accounts: int = 25) -> Dict[str, Dict]:
        """Kritik hesaplardan TÜM coinler için sosyal ısı haritası.
        coin -> {score(-1..1), mentions, weight, reasons}. Canlı değilse {} döner."""
        if not self.live:
            return {}
        heat: Dict[str, Dict] = {}
        top = sorted(self.accounts.items(), key=lambda kv: -kv[1]["weight"])[:max_accounts]
        for handle, info in top:
            for text in self.fetch_recent(handle, 3):
                coins = self._extract_coins(text)
                if not coins:
                    continue
                s = self._score_text(text)
                manip = self._manipulation(text, info["weight"])
                if manip >= 0.5:
                    continue
                causal = analyze_causal(text, info["weight"])
                eff = s + causal.net_direction * causal.magnitude
                for c in coins:
                    h = heat.setdefault(c, {"raw": 0.0, "mentions": 0, "weight": 0.0,
                                            "reasons": [], "events": []})
                    h["mentions"] += 1
                    h["weight"] += info["weight"]
                    h["raw"] += eff * info["weight"]
                    if causal.reasons:
                        h["reasons"].append(f"@{handle}: {causal.reasons[0]}")
                        h["events"].extend(causal.events)
        for c, h in heat.items():
            h["score"] = round(h["raw"] / (h["weight"] + 1e-9), 3)
            h["reasons"] = h["reasons"][:3]
            del h["raw"]
        return heat

    # ----------------------------------------------------------- aggregate
    def analyze(self, symbol: str) -> List[TweetSignal]:
        coin = symbol.split("/")[0].upper()
        out: List[TweetSignal] = []
        if not self.live:
            return out
        # En etkili hesaplardan örnekle (rate limit için sınırlı)
        top = sorted(self.accounts.items(), key=lambda kv: -kv[1]["weight"])[:15]
        for handle, info in top:
            assets = info.get("assets", [])
            if coin not in assets and "all" not in assets:
                continue
            for text in self.fetch_recent(handle, 3):
                coins = self._extract_coins(text)
                if coin not in coins and "all" not in assets:
                    continue
                s = self._score_text(text)
                manip = self._manipulation(text, info["weight"])
                impact = info["weight"] * 10 * (1 - manip)
                causal = analyze_causal(text, info["weight"])
                # nedensel olay varsa sentiment'i o yöne güçlendir
                if causal.net_direction != 0:
                    s = float(max(-1, min(1, s + causal.net_direction * causal.magnitude)))
                    impact = min(100, impact + causal.magnitude * 40)
                out.append(TweetSignal(
                    handle, info["weight"], s, impact, manip, coins, text[:140],
                    causal_dir=causal.net_direction, causal_mag=causal.magnitude,
                    causal_reasons=causal.reasons, events=causal.events,
                ))
        return out


def sentiment_vote(ti: "TwitterIntelligence", symbol: str) -> LayerVote:
    """Bir parite için ağırlıklı sentiment oyu (boğa/ayı)."""
    signals = ti.analyze(symbol)

    stats = account_stats()
    if not signals:
        # CANLI değil veya yeni tweet yok -> nötr, düşük güven (graceful)
        mode = "CANLI" if ti.live else "kapalı (TWITTER_BEARER_TOKEN yok)"
        return LayerVote(
            name="sentiment", score=0.0, confidence=0.15,
            reasons=[f"Twitter sentiment {mode}; nötr kabul edildi",
                     f"İzlenen kritik hesap: {stats['total']} "
                     f"({stats['crypto_exchange']} kripto/exchange + {stats['political_macro']} siyasi/makro)"],
            detail={"live": ti.live, "tweet_count": 0, **stats},
        )

    # Manipülasyon filtresi
    clean = [s for s in signals if s.manipulation < 0.5]
    use = clean or signals
    tw = sum(s.weight * (s.impact / 100) for s in use) + 1e-12
    score = sum(s.sentiment_score * s.weight * (s.impact / 100) for s in use) / tw

    reasons = [f"{len(use)} kritik hesap tweet'i analiz edildi "
               f"({len(signals) - len(clean)} manipülasyon filtrelendi) "
               f"| havuz: {stats['total']} hesap"]
    # Önce NEDENSEL olaylı tweet'ler (neden artış/azalış)
    causal_signals = [s for s in use if s.causal_dir != 0]
    for s in sorted(causal_signals, key=lambda x: -x.causal_mag)[:3]:
        for cr in s.causal_reasons[:1]:
            reasons.append(f"@{s.handle} (w{s.weight:.1f}): {cr}")
    # Sonra en etkili düz sentiment tweet'leri
    for s in sorted(use, key=lambda x: -abs(x.sentiment_score * x.weight))[:3]:
        arrow = "↑" if s.sentiment_score > 0 else "↓" if s.sentiment_score < 0 else "→"
        reasons.append(f"{arrow} @{s.handle} (w{s.weight:.1f}): {s.sentiment_score:+.2f}")

    confidence = min(0.9, 0.3 + 0.05 * len(use))
    return LayerVote(
        name="sentiment",
        score=float(np.clip(score, -1, 1)),
        confidence=float(confidence),
        reasons=reasons[:10],
        detail={"live": ti.live, "tweet_count": len(signals),
                "causal_events": [e for s in use for e in s.events], **stats},
    )
