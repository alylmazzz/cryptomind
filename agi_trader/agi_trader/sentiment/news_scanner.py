"""
HABER & SOSYAL TARAYICI — anahtarsız kaynaklardan parite bazlı katalizör/risk taraması.

Betik olarak:   python -m agi_trader.sentiment.news_scanner --once --symbols BTC/USDT,ETH/USDT
Modül olarak:   NewsScanner(symbols_fn).start()  → runner rolü/tetikleyicisi buradan okur

KAYNAKLAR (hepsi ücretsiz, anahtarsız; anahtar varsa CryptoPanic de eklenir):
  • Haber RSS: CoinDesk, Cointelegraph, Decrypt, The Block, CryptoSlate, Bitcoin Magazine,
    CryptoPotato, NewsBTC, U.Today
  • Google News RSS — varlık adıyla arama ("solana crypto")
  • Reddit (r/CryptoCurrency, r/CryptoMarkets, r/SatoshiStreetBets) — public JSON
  • StockTwits — sembol akışı; kullanıcıların Bullish/Bearish etiketleri (gerçek sosyal oy)
  • Binance duyuruları — yeni listeleme / delist (kural: listeleme boğa, delist ayı-risk)

ÇIKTI (parite başına): skor −1..+1 (sözlük + kaynak ağırlığı + zaman azalımı, yarı-ömür 6 sa),
buzz (dikkat), boğa/ayı sayısı, katalizör etiketleri, RİSK bayrakları (hack/exploit/delist/
dava/yasak/kesinti/kilit açılışı), sosyal boğa/ayı oranı, en yeni başlıklar.
Piyasa geneli: risk-off puanı (borsa hack'i, düzenleyici darbe, sistemik haber).

HAREKETLİLİK DOĞRULAMASI: bir haber ancak fiyat/hacim onu doğrularsa "confirmed" olur —
son 3 saat hacmi 24 sa ortalamasının ≥1,5 katı VE 4 sa fiyat hareketi ≥ 0,5 ATR (haber yönünde).
Haber tek başına işlem açtırmaz; komitede ROL olarak oy verir, katalizör TETİKLEYİCİSİ ancak
doğrulanmış hareketle çalışır.

DÜRÜSTLÜK: sözlük tabanlı duygu KABADIR ve bu projede ölçülmedi. Bu yüzden rolün taban
ağırlığı düşüktür ve ders motoru isabetine göre günceller. Veri gelmezse "veri yok" der,
nötr uydurmaz.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

UA = {"User-Agent": "Mozilla/5.0 (CryptoMind news scanner; +https://mindcorplab.com)"}
TIMEOUT = 8
HALF_LIFE_H = 6.0
MAX_AGE_H = 48.0

RSS_SOURCES = [
    ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/", 1.0),
    ("cointelegraph", "https://cointelegraph.com/rss", 1.0),
    ("decrypt", "https://decrypt.co/feed", 0.9),
    ("theblock", "https://www.theblock.co/rss.xml", 1.0),
    ("cryptoslate", "https://cryptoslate.com/feed/", 0.8),
    ("bitcoinmagazine", "https://bitcoinmagazine.com/feed", 0.7),
    ("cryptopotato", "https://cryptopotato.com/feed/", 0.6),
    ("newsbtc", "https://www.newsbtc.com/feed/", 0.6),
    ("utoday", "https://u.today/rss", 0.5),
]
GOOGLE_NEWS = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
REDDIT = [
    "https://www.reddit.com/r/CryptoCurrency/new/.rss?limit=50",
    "https://www.reddit.com/r/CryptoMarkets/new/.rss?limit=40",
    "https://www.reddit.com/r/SatoshiStreetBets/new/.rss?limit=40",
]
STOCKTWITS = "https://api.stocktwits.com/api/2/streams/symbol/{base}.X.json"
BINANCE_ANN = ("https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
               "?type=1&pageNo=1&pageSize=20&catalogId={cat}")
BINANCE_CATS = {"48": "listing", "161": "delist"}

# Varlık takma adları — başlıkta eşleşme (küçük harf). Ticker ayrıca $TICKER / kelime sınırı ile.
ASSET_ALIASES: Dict[str, List[str]] = {
    "BTC": ["bitcoin"], "ETH": ["ethereum", "ether"], "SOL": ["solana"], "DOGE": ["dogecoin"],
    "AVAX": ["avalanche"], "LINK": ["chainlink"], "SUI": ["sui network", "sui blockchain"],
    "NEAR": ["near protocol"], "PEPE": ["pepe coin", "pepecoin", "pepe token"], "AAVE": ["aave"],
    "UNI": ["uniswap"], "LTC": ["litecoin"], "DOT": ["polkadot"], "FIL": ["filecoin"],
    "BCH": ["bitcoin cash"], "ADA": ["cardano"], "XRP": ["ripple", "xrp"], "XLM": ["stellar"],
    "TRUMP": ["trump coin", "trump token", "official trump", "$trump"], "PENDLE": ["pendle"],
    "SHIB": ["shiba inu"], "OP": ["optimism"], "RENDER": ["render network", "render token"],
    "ARKM": ["arkham"], "ENA": ["ethena"], "WLD": ["worldcoin"], "TIA": ["celestia"],
    "ARB": ["arbitrum"], "GALA": ["gala games", "gala token"], "ONDO": ["ondo finance", "ondo"],
    "ICP": ["internet computer", "dfinity"], "BONK": ["bonk"], "ATOM": ["cosmos"],
    "JUP": ["jupiter exchange", "jupiter dex", "jupiter aggregator"], "WIF": ["dogwifhat"],
    "EIGEN": ["eigenlayer", "eigen layer"], "SEI": ["sei network"], "LDO": ["lido"],
    "VIRTUAL": ["virtuals protocol", "virtuals"], "STRK": ["starknet"], "BNB": ["binance coin", "bnb chain"],
    "TRX": ["tron"], "TON": ["toncoin", "the open network"],
}
GENERIC_TICKER_MIN_LEN = 3

# Sözlük — ağırlıklı. Katalizör = boğa + olay; risk = ayı + olay.
BULL = {"surge": 0.6, "rally": 0.6, "soar": 0.7, "bullish": 0.6, "breakout": 0.5, "adoption": 0.6,
        "partnership": 0.7, "partners with": 0.7, "approval": 0.8, "approved": 0.8, "etf": 0.6,
        "inflow": 0.5, "upgrade": 0.6, "record high": 0.7, "all-time high": 0.7, "ath": 0.5,
        "listing": 0.8, "lists": 0.6, "listed on": 0.7, "mainnet": 0.7, "launch": 0.5, "burn": 0.5,
        "buyback": 0.7, "integration": 0.5, "integrates": 0.5, "acquire": 0.4, "treasury buys": 0.7,
        "accumulat": 0.5, "whale buys": 0.5, "airdrop": 0.4, "staking": 0.3, "institutional": 0.4,
        "spot etf": 0.8, "approves": 0.8, "rebound": 0.4, "recover": 0.3, "jumps": 0.5, "gains": 0.4}
BEAR = {"crash": -0.8, "plunge": -0.7, "dump": -0.6, "bearish": -0.6, "selloff": -0.6, "sell-off": -0.6,
        "hack": -0.9, "hacked": -0.9, "exploit": -0.9, "drained": -0.9, "ban": -0.7, "bans": -0.7,
        "lawsuit": -0.7, "sues": -0.7, "sec charges": -0.8, "outflow": -0.5, "downgrade": -0.5,
        "liquidation": -0.5, "liquidated": -0.5, "collapse": -0.8, "fraud": -0.8, "scam": -0.7,
        "rug": -0.8, "delist": -0.9, "delisting": -0.9, "halts": -0.7, "halted": -0.7, "outage": -0.6,
        "paused": -0.5, "bankrupt": -0.9, "insolven": -0.9, "unlock": -0.4, "token unlock": -0.5,
        "investigation": -0.5, "warning": -0.3, "falls": -0.4, "drops": -0.4, "slumps": -0.5,
        "fine": -0.4, "penalt": -0.4, "shutdown": -0.7, "vulnerabilit": -0.6}
CATALYST_TAGS = {"listing": ["listing", "lists", "listed on"], "etf": ["etf"], "partnership": ["partner"],
                 "mainnet/upgrade": ["mainnet", "upgrade", "hard fork"], "burn/buyback": ["burn", "buyback"],
                 "treasury": ["treasury", "institutional"], "airdrop": ["airdrop"]}
RISK_TAGS = {"hack/exploit": ["hack", "exploit", "drained", "vulnerabilit"], "delist": ["delist"],
             "legal": ["lawsuit", "sues", "sec charges", "investigation", "charges"],
             "ban": ["ban", "bans"], "halt/outage": ["halt", "outage", "paused", "shutdown"],
             "unlock": ["unlock"], "insolvency": ["bankrupt", "insolven", "collapse"]}
# Piyasa geneli risk — yalnız SİSTEMİK ve GERÇEKLEŞMİŞ olaylar (spekülatif başlıklar hariç).
# "tether" gibi tek kelime SAYILMAZ; ölçüldü: sıradan Tether davası haberleri paneli
# NAKİT MODU'na sokuyordu. Soru cümlesi / could / may / prediction içeren başlık dışlanır.
MARKET_RISK_STRONG = ["exchange hack", "exchange hacked", "exchange exploit", "binance hack", "coinbase hack",
                      "bybit hack", "okx hack", "halts withdrawals", "withdrawals halted", "pauses withdrawals",
                      "stablecoin depeg", "depegs", "loses peg", "files for bankruptcy", "insolvency",
                      "bans crypto", "crypto ban", "sec sues", "cftc sues", "doj charges", "market crash",
                      "crashes", "liquidations top", "billion liquidated", "contagion"]
MARKET_SPECULATIVE = ["?", " could ", " would ", " may ", " might ", "prediction", "what if", "analyst says"]

# Kaynak katmanı: 1 resmî (borsa/proje/düzenleyici) · 2 güvenilir medya · 3 sosyal/forum.
# Tier-3 tek başına katalizör TETİKLEMEZ (rol oyu verir, tetikleyici için Tier-1/2 şart).
SOURCE_TIER = {"binance_listing": 1, "binance_delist": 1, "coindesk": 2, "cointelegraph": 2, "decrypt": 2,
               "theblock": 2, "cryptoslate": 2, "bitcoinmagazine": 2, "cryptopotato": 2, "newsbtc": 2,
               "utoday": 2, "googlenews": 2, "reddit": 3, "stocktwits": 3}
# Olay taksonomisi (öncelik sırasıyla ilk eşleşen)
EVENT_TAXONOMY = [
    ("DELISTING", ["delist"]), ("HACK", ["hacked", "hack", "drained", "stolen"]), ("EXPLOIT", ["exploit", "vulnerabilit"]),
    ("CHAIN_OUTAGE", ["outage", "halted", "halts", "downtime", "network down"]), ("BANKRUPTCY", ["bankrupt", "insolven", "chapter 11"]),
    ("LISTING", ["listing", "lists", "listed on", "will list"]), ("ETF", ["etf"]), ("TOKEN_UNLOCK", ["unlock"]),
    ("TOKEN_BURN", ["burn", "buyback"]), ("PROTOCOL_UPGRADE", ["mainnet", "upgrade", "hard fork", "testnet", "v2 "]),
    ("PARTNERSHIP", ["partner", "integrat", "collaborat"]), ("TREASURY", ["treasury", "institutional", "acquire", "accumulat"]),
    ("WHALE", ["whale"]), ("REGULATORY", ["sec ", "cftc", "regulator", "regulation", "ban", "bans", "approval", "approves", "license"]),
    ("LEGAL", ["lawsuit", "sues", "charges", "court", "investigation", "fine", "penalt"]),
    ("FUNDING", ["raises", "funding round", "series a", "series b", "valuation"]), ("LIQUIDITY", ["liquidat", "depeg", "peg"]),
    ("MACRO", ["fed ", "fomc", "cpi", "inflation", "rate cut", "rate hike", "treasury yield"]), ("RUMOR", ["rumor", "reportedly", "sources say"]),
]
EVENT_PRIOR = {"LISTING": 0.6, "ETF": 0.5, "TOKEN_BURN": 0.4, "PROTOCOL_UPGRADE": 0.3, "PARTNERSHIP": 0.3, "TREASURY": 0.4,
               "WHALE": 0.0, "FUNDING": 0.2, "DELISTING": -0.9, "HACK": -0.9, "EXPLOIT": -0.8, "CHAIN_OUTAGE": -0.6,
               "BANKRUPTCY": -0.9, "TOKEN_UNLOCK": -0.3, "REGULATORY": 0.0, "LEGAL": -0.4, "LIQUIDITY": -0.5, "MACRO": 0.0, "RUMOR": 0.0}


def classify_event(text: str) -> Dict:
    t = (text or "").lower()
    for ev, ws in EVENT_TAXONOMY:
        if any(_has(t, w) for w in ws):
            return {"event_type": ev, "directional_prior": EVENT_PRIOR.get(ev, 0.0)}
    return {"event_type": "OTHER", "directional_prior": 0.0}


def _dedup_key(title: str) -> str:
    """Kaynaklar arası tekrar: noktalama/kaynak son eki atılır, ilk 8 anlamlı kelime."""
    t = re.sub(r"[^a-z0-9 ]", " ", (title or "").lower())
    t = re.sub(r"\s+-\s+[a-z .]+$", "", t)
    words = [w for w in t.split() if len(w) > 2][:8]
    return " ".join(words)


# ═══════════════════════════════════════════════════════════════════════════
# yardımcılar
# ═══════════════════════════════════════════════════════════════════════════
def _get(url: str, timeout: int = TIMEOUT, headers: Optional[Dict] = None):
    if requests is None:
        return None
    try:
        r = requests.get(url, timeout=timeout, headers=headers or UA)
        if r.status_code != 200:
            return None
        return r
    except Exception:
        return None


def _parse_rss(text: str, source: str, weight: float, limit: int = 40) -> List[Dict]:
    out = []
    items = re.findall(r"<item>(.*?)</item>", text, re.S)[:limit]
    for it in items:
        t = re.search(r"<title>(.*?)</title>", it, re.S)
        d = re.search(r"<pubDate>(.*?)</pubDate>", it, re.S)
        l = re.search(r"<link>(.*?)</link>", it, re.S)
        if not t:
            continue
        title = html.unescape(re.sub(r"<!\[CDATA\[|\]\]>", "", t.group(1))).strip()
        ts = _parse_date(d.group(1)) if d else time.time()
        out.append({"source": source, "weight": weight, "title": title, "ts": ts,
                    "url": (l.group(1).strip() if l else "")})
    return out


def _parse_date(s: str) -> float:
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(s.strip()).timestamp()
    except Exception:
        return time.time()


# Gövde (stem) terimleri: sonu açık ("accumulat" → accumulating/accumulation). Diğer her
# terim KELİME SINIRIYLA aranır — ölçüldü: alt-dize eşleşmesi "Binance" içindeki "ban"ı
# yasak riski, "path" içindeki "ath"i rekor sayıyordu.
STEMS = {"accumulat", "insolven", "vulnerabilit", "penalt", "liquidat", "partner", "halt", "charge"}
_TERM_RE: Dict[str, "re.Pattern"] = {}


def _has(t: str, w: str) -> bool:
    pat = _TERM_RE.get(w)
    if pat is None:
        esc = re.escape(w)
        pat = re.compile(r"(?<![a-z0-9])" + esc + (r"" if w in STEMS else r"(?![a-z0-9])"))
        _TERM_RE[w] = pat
    return bool(pat.search(t))


def score_text(text: str) -> Dict:
    """Sözlük skoru −1..+1 + katalizör/risk etiketleri. Ölçülmemiş, kaba."""
    t = (text or "").lower()
    pos = [w for w in BULL if _has(t, w)]
    neg = [w for w in BEAR if _has(t, w)]
    raw = sum(BULL[w] for w in pos) + sum(BEAR[w] for w in neg)
    n = len(pos) + len(neg)
    score = max(-1.0, min(1.0, raw / max(1, n))) if n else 0.0
    cats = [k for k, ws in CATALYST_TAGS.items() if any(_has(t, w) for w in ws)]
    risks = [k for k, ws in RISK_TAGS.items() if any(_has(t, w) for w in ws)]
    return {"score": round(score, 3), "pos": pos, "neg": neg, "catalysts": cats, "risks": risks}


def match_assets(text: str, bases: List[str]) -> List[str]:
    t = (text or "").lower()
    hit = []
    for b in bases:
        aliases = ASSET_ALIASES.get(b, [])
        ok = any(a in t for a in aliases)
        if not ok and len(b) >= GENERIC_TICKER_MIN_LEN:
            ok = bool(re.search(r"(?<![a-z0-9])\$?" + re.escape(b.lower()) + r"(?![a-z0-9])", t))
        if ok:
            hit.append(b)
    return hit


def _decay(ts: float, now: float) -> float:
    age_h = max(0.0, (now - ts) / 3600.0)
    if age_h > MAX_AGE_H:
        return 0.0
    return 0.5 ** (age_h / HALF_LIFE_H)


# ═══════════════════════════════════════════════════════════════════════════
# kaynaklar
# ═══════════════════════════════════════════════════════════════════════════
def fetch_rss_all() -> List[Dict]:
    items: List[Dict] = []
    for name, url, w in RSS_SOURCES:
        r = _get(url)
        if r is not None:
            items += _parse_rss(r.text, name, w)
    return items


def fetch_google_news(base: str) -> List[Dict]:
    q = (ASSET_ALIASES.get(base, [base.lower()])[0]).replace(" ", "+") + "+crypto"
    r = _get(GOOGLE_NEWS.format(q=q))
    if r is None:
        return []
    out = _parse_rss(r.text, "googlenews", 0.6, limit=15)
    for it in out:
        it["forced_asset"] = base
    return out


def fetch_reddit() -> List[Dict]:
    """Reddit Atom akışı (anahtarsız). Public JSON ucu betiklere 403 döndüğü için RSS."""
    items: List[Dict] = []
    for url in REDDIT:
        r = _get(url, headers={**UA, "Accept": "application/atom+xml,application/xml"})
        if r is None:
            continue
        for e in re.findall(r"<entry>(.*?)</entry>", r.text, re.S)[:50]:
            t = re.search(r"<title>(.*?)</title>", e, re.S)
            d = re.search(r"<updated>(.*?)</updated>", e, re.S) or re.search(r"<published>(.*?)</published>", e, re.S)
            l = re.search(r'<link[^>]*href="([^"]+)"', e)
            if not t:
                continue
            title = html.unescape(re.sub(r"<!\[CDATA\[|\]\]>", "", t.group(1))).strip()
            try:
                ts = _parse_iso(d.group(1)[:19] + "Z") if d else time.time()
            except Exception:
                ts = time.time()
            items.append({"source": "reddit", "weight": 0.4, "title": title, "ts": ts,
                          "url": (l.group(1) if l else "")})
    return items


def fetch_stocktwits(base: str) -> Optional[Dict]:
    r = _get(STOCKTWITS.format(base=base))
    if r is None:
        return None
    try:
        msgs = r.json().get("messages", [])
    except Exception:
        return None
    now = time.time()
    bull = bear = n = 0
    for m in msgs:
        try:
            ts = _parse_iso(m.get("created_at", ""))
        except Exception:
            ts = now
        if now - ts > 24 * 3600:
            continue
        n += 1
        s = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
        if s == "Bullish":
            bull += 1
        elif s == "Bearish":
            bear += 1
    return {"msgs_24h": n, "bull": bull, "bear": bear,
            "ratio": round((bull - bear) / (bull + bear), 3) if (bull + bear) else None}


def _parse_iso(s: str) -> float:
    from datetime import datetime, timezone
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def fetch_binance_announcements() -> List[Dict]:
    items: List[Dict] = []
    for cat, kind in BINANCE_CATS.items():
        r = _get(BINANCE_ANN.format(cat=cat), headers={**UA, "Accept": "application/json"})
        if r is None:
            continue
        try:
            arts = ((r.json().get("data") or {}).get("catalogs") or [{}])[0].get("articles") or []
        except Exception:
            continue
        for a in arts:
            title = (a.get("title") or "").strip()
            ts = float(a.get("releaseDate") or 0) / 1000.0 or time.time()
            items.append({"source": f"binance_{kind}", "weight": 1.5, "title": title, "ts": ts,
                          "url": "https://www.binance.com/en/support/announcement/" + str(a.get("code") or "")})
    return items


# ═══════════════════════════════════════════════════════════════════════════
# tarama
# ═══════════════════════════════════════════════════════════════════════════
def _base(symbol: str) -> str:
    return symbol.split("/")[0].upper()


def scan(symbols: List[str], fetch_ohlcv: Optional[Callable[[str, str, int], object]] = None,
         with_social: bool = True, with_google: bool = True, now: Optional[float] = None,
         items_override: Optional[List[Dict]] = None) -> Dict:
    """Tam tarama. `fetch_ohlcv(symbol, tf, limit)` verilirse hareketlilik doğrulaması yapılır."""
    now = time.time() if now is None else now
    bases = [_base(s) for s in symbols]
    items = list(items_override) if items_override is not None else fetch_rss_all() + fetch_reddit() + fetch_binance_announcements()
    if items_override is None and with_google:
        for b in bases:
            items += fetch_google_news(b)
    # eşleştir + puanla
    per: Dict[str, Dict] = {b: {"items": [], "wsum": 0.0, "ssum": 0.0, "bull": 0, "bear": 0,
                                "catalysts": {}, "risks": {}, "buzz": 0.0} for b in bases}
    market_risk = 0.0
    market_items: List[Dict] = []
    seen = set()
    for it in items:
        key = _dedup_key(it["title"])
        if not key or key in seen:
            continue
        seen.add(key)
        it["tier"] = SOURCE_TIER.get(it["source"], 3)
        it.update(classify_event(it["title"]))
        w = _decay(it["ts"], now) * float(it.get("weight", 1.0))
        if w <= 0:
            continue
        sc = score_text(it["title"])
        tl = it["title"].lower()
        speculative = any(m in (" " + tl + " ") for m in MARKET_SPECULATIVE)
        strong = any(m in tl for m in MARKET_RISK_STRONG)
        severe_tag = any(k in sc["risks"] for k in ("hack/exploit", "insolvency", "halt/outage"))
        if not speculative and (strong or (severe_tag and ("exchange" in tl or "protocol" in tl))):
            market_risk += w * (1.0 if sc["score"] <= 0 else 0.3)
            market_items.append({"title": it["title"], "source": it["source"], "ts": it["ts"]})
        hits = [it["forced_asset"]] if it.get("forced_asset") else match_assets(it["title"], bases)
        for b in hits:
            if b not in per:
                continue
            p = per[b]
            p["items"].append({"title": it["title"], "source": it["source"], "ts": it["ts"],
                               "score": sc["score"], "url": it.get("url", ""), "tier": it.get("tier", 3),
                               "event_type": it.get("event_type", "OTHER")})
            p.setdefault("events", {})
            p["events"][it.get("event_type", "OTHER")] = p["events"].get(it.get("event_type", "OTHER"), 0) + 1
            if it.get("tier", 3) <= 2:
                p["tier12"] = p.get("tier12", 0) + 1
            p["wsum"] += w
            p["ssum"] += w * sc["score"]
            p["buzz"] += w
            if sc["score"] > 0.1:
                p["bull"] += 1
            elif sc["score"] < -0.1:
                p["bear"] += 1
            for c in sc["catalysts"]:
                p["catalysts"][c] = p["catalysts"].get(c, 0) + 1
            for r in sc["risks"]:
                p["risks"][r] = p["risks"].get(r, 0) + 1
    out: Dict[str, Dict] = {}
    for s in symbols:
        b = _base(s)
        p = per[b]
        n = len(p["items"])
        score = (p["ssum"] / p["wsum"]) if p["wsum"] > 0 else 0.0
        social = fetch_stocktwits(b) if (with_social and items_override is None) else None
        rec = {"symbol": s, "base": b, "n_items": n, "score": round(score, 3),
               "buzz": round(p["buzz"], 3), "bull": p["bull"], "bear": p["bear"],
               "catalysts": p["catalysts"], "risks": p["risks"],
               "severe_risk": any(k in p["risks"] for k in ("hack/exploit", "delist", "insolvency", "halt/outage")),
               "social": social, "data_ok": n > 0 or bool(social and social.get("msgs_24h")),
               "events": p.get("events", {}), "tier12_items": int(p.get("tier12", 0)),
               "top_event": (max(p.get("events", {}).items(), key=lambda kv: kv[1])[0] if p.get("events") else None),
               "headlines": sorted(p["items"], key=lambda x: -x["ts"])[:5],
               "confirmed": None, "move_pct_4h": None, "vol_ratio": None}
        if fetch_ohlcv is not None:
            rec.update(confirm_move(s, score, fetch_ohlcv))
        out[s] = rec
    return {"generated_at": now, "symbols": out,
            "market": {"risk_off_score": round(market_risk, 3),
                       # seviye 2 = NAKİT MODU: en az İKİ ayrı sistemik başlık ve yüksek puan
                       "level": (2 if (market_risk >= 3.0 and len(market_items) >= 2)
                                 else 1 if market_risk >= 1.0 else 0),
                       "items": sorted(market_items, key=lambda x: -x["ts"])[:6]},
            "sources": {"n_items": len(items), "n_unique": len(seen), "rss": len(RSS_SOURCES), "reddit": len(REDDIT),
                        "google": with_google, "stocktwits": with_social,
                        "note": "sözlük tabanlı duygu — ölçülmedi, kaba; rol ağırlığı ders motoruyla güncellenir"}}


def confirm_move(symbol: str, news_score: float, fetch_ohlcv) -> Dict:
    """Haberin yönünde HAREKET var mı? (hacim ≥1,5× ve 4 sa hareket ≥ 0,5 ATR)"""
    try:
        df = fetch_ohlcv(symbol, "1h", 60)
        c = df["close"].astype(float)
        v = df["volume"].astype(float)
        h = df["high"].astype(float); l = df["low"].astype(float)
        tr = (h - l).rolling(14).mean()
        atr_pct = float(tr.iloc[-1] / c.iloc[-1] * 100.0)
        move = float((c.iloc[-1] / c.iloc[-5] - 1.0) * 100.0) if len(c) >= 5 else 0.0
        vol_ratio = float(v.tail(3).mean() / max(1e-9, v.tail(27).head(24).mean())) if len(v) >= 27 else 1.0
        direction = 1.0 if news_score >= 0 else -1.0
        confirmed = bool(vol_ratio >= 1.5 and move * direction >= 0.5 * atr_pct and abs(news_score) >= 0.3)
        return {"confirmed": confirmed, "move_pct_4h": round(move, 3), "vol_ratio": round(vol_ratio, 2),
                "atr_pct_1h": round(atr_pct, 3)}
    except Exception as e:
        return {"confirmed": None, "move_pct_4h": None, "vol_ratio": None, "confirm_error": type(e).__name__}


# ═══════════════════════════════════════════════════════════════════════════
# arka plan tarayıcı (süreç içi)
# ═══════════════════════════════════════════════════════════════════════════
class NewsScanner:
    def __init__(self, symbols_fn: Callable[[], List[str]], fetch_ohlcv=None,
                 interval_sec: int = 600, out_path: Optional[Path] = None):
        self.symbols_fn = symbols_fn
        self.fetch_ohlcv = fetch_ohlcv
        self.interval = max(120, int(interval_sec))
        self.out_path = Path(out_path) if out_path else None
        self.latest: Dict = {}
        self.last_run: Optional[float] = None
        self.last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = False
        if self.out_path and self.out_path.exists():
            try:
                self.latest = json.loads(self.out_path.read_text(encoding="utf-8"))
                self.last_run = float(self.latest.get("generated_at") or 0) or None
            except Exception:
                self.latest = {}

    def run_once(self) -> Dict:
        try:
            self.latest = scan(list(self.symbols_fn()), fetch_ohlcv=self.fetch_ohlcv)
            import gc; gc.collect()                      # tarama artıkları (feed/XML ağaçları) hemen bırakılsın
            self.last_run = time.time()
            self.last_error = None
            if self.out_path:
                self.out_path.parent.mkdir(parents=True, exist_ok=True)
                self.out_path.write_text(json.dumps(self.latest, ensure_ascii=False, default=str),
                                         encoding="utf-8")
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
        return self.latest

    def _loop(self):
        while not self._stop:
            self.run_once()
            for _ in range(self.interval):
                if self._stop:
                    return
                time.sleep(1)

    def start(self) -> None:
        if self._thread:
            return
        self._thread = threading.Thread(target=self._loop, name="cm-news", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def for_symbol(self, symbol: str) -> Optional[Dict]:
        d = (self.latest.get("symbols") or {}).get(symbol)
        if d is None:
            return None
        return {**d, "age_sec": (time.time() - float(self.latest.get("generated_at") or time.time()))}

    def market(self) -> Dict:
        m = dict(self.latest.get("market") or {"risk_off_score": 0.0, "level": 0, "items": []})
        m["age_sec"] = (time.time() - float(self.latest.get("generated_at") or time.time())) if self.latest else None
        m["last_error"] = self.last_error
        return m


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="CryptoMind haber/sosyal tarayıcı")
    ap.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--out", default="runs/news/news_scan.json")
    ap.add_argument("--no-social", action="store_true")
    ap.add_argument("--confirm", action="store_true", help="MEXC 1h verisiyle hareketlilik doğrulaması")
    a = ap.parse_args()
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    fetch = None
    if a.confirm:
        import ccxt
        import pandas as pd
        ex = ccxt.mexc({"enableRateLimit": True})

        def fetch(symbol, tf, limit):
            rows = ex.fetch_ohlcv(symbol, tf, limit=limit)
            return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    sc = NewsScanner(lambda: syms, fetch_ohlcv=fetch, interval_sec=a.interval, out_path=Path(a.out))
    while True:
        d = sc.run_once()
        print(json.dumps({"generated_at": d.get("generated_at"), "market": d.get("market"),
                          "sources": d.get("sources")}, ensure_ascii=False))
        for s, r in (d.get("symbols") or {}).items():
            print(f"{s:12s} skor {r['score']:+.2f} n={r['n_items']:2d} boğa {r['bull']} ayı {r['bear']} "
                  f"katalizör {list(r['catalysts'])} risk {list(r['risks'])} "
                  f"sosyal {(r.get('social') or {}).get('ratio')} doğrulandı {r.get('confirmed')}")
            for hl in r["headlines"][:2]:
                print(f"    · [{hl['source']}] {hl['title'][:100]}")
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
