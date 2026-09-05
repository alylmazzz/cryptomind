"""
Nedensel Etki Motoru (Event → Yön → Gerekçe).

Bir tweet'in NEDEN artışa ya da azalışa sebep olacağını net belirler.
Her "olay türü" için: çok-dilli anahtar kelimeler, beklenen yön (boğa/ayı),
temel etki büyüklüğü (0-1) ve insan-okur gerekçe şablonu tanımlıdır.

analyze_causal(text, account_weight) → tespit edilen olaylar + net yön +
büyüklük + gerekçeler. Bu, "sentiment pozitif/negatif" demekten farklıdır:
SEBEBİ söyler (ör. "ETF onayı → kurumsal giriş beklentisi → yükseliş").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

# (yön: +1 boğa / -1 ayı, taban_büyüklük 0-1, anahtarlar, gerekçe)
EVENT_TAXONOMY: Dict[str, Dict] = {
    "ETF_APPROVAL":      {"dir": +1, "mag": 0.95, "kw": ["etf approv", "etf onay", "spot etf", "etf greenlight", "etf launch"],
                          "why": "ETF onayı → kurumsal sermaye girişi beklentisi → güçlü yükseliş"},
    "ETF_REJECTION":     {"dir": -1, "mag": 0.85, "kw": ["etf reject", "etf red", "etf delay", "etf ertelendi", "etf denied"],
                          "why": "ETF reddi/ertelemesi → kurumsal giriş gecikmesi → düşüş"},
    "LISTING":           {"dir": +1, "mag": 0.8, "kw": ["will list", "listing", "now trading", "listeleme", "lists $", "added to"],
                          "why": "Borsa listeleme → likidite ve talep artışı → yükseliş"},
    "DELISTING":         {"dir": -1, "mag": 0.85, "kw": ["delist", "delisting", "remove", "kaldırıl", "suspend trading"],
                          "why": "Delist → likidite kaybı → sert düşüş"},
    "HACK_EXPLOIT":      {"dir": -1, "mag": 0.95, "kw": ["hack", "exploit", "drained", "breach", "stolen", "saldırı", "çalındı", "rug"],
                          "why": "Hack/exploit → güven kaybı ve fon çıkışı → sert düşüş"},
    "PARTNERSHIP":       {"dir": +1, "mag": 0.6, "kw": ["partnership", "partner with", "integration", "collab", "ortaklık", "iş birliği"],
                          "why": "Ortaklık/entegrasyon → benimseme artışı → yükseliş"},
    "REG_NEGATIVE":      {"dir": -1, "mag": 0.85, "kw": ["lawsuit", "sue", "sec charge", "ban", "crackdown", "dava", "yasak", "soruşturma", "investigation"],
                          "why": "Olumsuz regülasyon/dava → belirsizlik ve satış → düşüş"},
    "REG_POSITIVE":      {"dir": +1, "mag": 0.7, "kw": ["approved", "regulation clarity", "legal", "license granted", "lisans", "düzenleme netliği", "favorable ruling"],
                          "why": "Olumlu regülasyon/lisans → meşruiyet artışı → yükseliş"},
    "RATE_HIKE":         {"dir": -1, "mag": 0.8, "kw": ["rate hike", "raise rates", "hawkish", "faiz artır", "tightening"],
                          "why": "Faiz artışı/şahin Fed → riskli varlıklardan çıkış → düşüş"},
    "RATE_CUT":          {"dir": +1, "mag": 0.8, "kw": ["rate cut", "cut rates", "dovish", "faiz indir", "easing", "qe", "pivot"],
                          "why": "Faiz indirimi/güvercin Fed → likidite bolluğu → yükseliş"},
    "CPI_HOT":           {"dir": -1, "mag": 0.6, "kw": ["cpi hot", "inflation higher", "enflasyon yüksek", "hot inflation"],
                          "why": "Yüksek enflasyon → sıkı politika beklentisi → düşüş"},
    "CPI_COOL":          {"dir": +1, "mag": 0.6, "kw": ["cpi cool", "inflation lower", "enflasyon düş", "disinflation"],
                          "why": "Düşük enflasyon → gevşeme beklentisi → yükseliş"},
    "TOKEN_UNLOCK":      {"dir": -1, "mag": 0.55, "kw": ["unlock", "vesting", "cliff", "kilit açıl", "token release"],
                          "why": "Token kilidi açılışı → arz baskısı → düşüş"},
    "BURN":              {"dir": +1, "mag": 0.5, "kw": ["burn", "buyback", "yakım", "geri alım", "token burn"],
                          "why": "Burn/geri alım → arz azalması → yükseliş"},
    "WHALE_ACCUMULATE":  {"dir": +1, "mag": 0.7, "kw": ["whale buy", "accumulat", "whale bought", "balina alım", "withdrew from exchange", "exchange outflow"],
                          "why": "Whale birikimi/borsadan çıkış → arz daralması → yükseliş"},
    "WHALE_DISTRIBUTE":  {"dir": -1, "mag": 0.75, "kw": ["whale sell", "whale dump", "deposit to exchange", "balina satış", "exchange inflow", "moved to binance"],
                          "why": "Whale dağıtımı/borsaya giriş → satış baskısı → düşüş"},
    "HALVING":           {"dir": +1, "mag": 0.6, "kw": ["halving", "halvening", "yarılanma"],
                          "why": "Halving → arz şoku → orta vade yükseliş"},
    "ENDORSEMENT":       {"dir": +1, "mag": 0.6, "kw": ["i bought", "i'm buying", "bullish on", "accumulating", "long ", "buying the dip"],
                          "why": "Etkili hesabın açık alımı → takipçi talebi → yükseliş"},
    "WARNING_BEARISH":   {"dir": -1, "mag": 0.65, "kw": ["sell everything", "top is in", "bear market", "crash incoming", "short ", "exit now", "tepe yaptı"],
                          "why": "Etkili hesabın satış uyarısı → panik satışı → düşüş"},
    "DEPEG":             {"dir": -1, "mag": 0.9, "kw": ["depeg", "lost peg", "depegged", "stablecoin collapse"],
                          "why": "Stablecoin depeg → sistemik güvensizlik → sert düşüş"},
    "BANKRUPTCY":        {"dir": -1, "mag": 0.95, "kw": ["bankrupt", "insolvent", "halt withdrawals", "iflas", "ödeme durdur", "chapter 11"],
                          "why": "İflas/çekim durdurma → bulaşma riski → sert düşüş"},
}


@dataclass
class CausalResult:
    events: List[str] = field(default_factory=list)
    net_direction: int = 0          # +1 boğa, -1 ayı, 0 nötr
    magnitude: float = 0.0          # 0..1 beklenen etki şiddeti
    reasons: List[str] = field(default_factory=list)


def _combination_rules(t: str) -> List[str]:
    """Kelime sırasından bağımsız bileşik olay tespiti."""
    found = []
    if "etf" in t and any(w in t for w in ["approv", "greenlight", "launch", "go live", "onay"]):
        found.append("ETF_APPROVAL")
    if "etf" in t and any(w in t for w in ["reject", "deny", "denied", "delay", "ertele", "red"]):
        found.append("ETF_REJECTION")
    if any(w in t for w in ["rate", "faiz", "fed"]) and any(w in t for w in ["cut", "indir", "lower", "dovish", "pivot"]):
        found.append("RATE_CUT")
    if any(w in t for w in ["rate", "faiz", "fed"]) and any(w in t for w in ["hike", "raise", "artır", "hawkish", "higher"]):
        found.append("RATE_HIKE")
    return found


def analyze_causal(text: str, account_weight: float = 5.0) -> CausalResult:
    t = text.lower()
    hits: List[tuple] = []   # (dir, mag, name, why)
    matched = set()
    for name, ev in EVENT_TAXONOMY.items():
        for kw in ev["kw"]:
            if kw in t:
                hits.append((ev["dir"], ev["mag"], name, ev["why"]))
                matched.add(name)
                break
    # bileşik kurallar
    for name in _combination_rules(t):
        if name not in matched:
            ev = EVENT_TAXONOMY[name]
            hits.append((ev["dir"], ev["mag"], name, ev["why"]))
            matched.add(name)

    if not hits:
        return CausalResult()

    # hesap ağırlığı etkisi (0-10 → 0.5-1.5 çarpan)
    w_mult = 0.5 + account_weight / 10.0
    score = sum(d * m for d, m, _, _ in hits)
    mag = min(1.0, (sum(m for _, m, _, _ in hits) / len(hits)) * w_mult)
    net = 1 if score > 0 else -1 if score < 0 else 0

    reasons = []
    for d, m, name, why in sorted(hits, key=lambda x: -x[1])[:3]:
        arrow = "📈 YÜKSELİŞ" if d > 0 else "📉 DÜŞÜŞ"
        reasons.append(f"{arrow} [{name}] {why}")

    return CausalResult(events=[h[2] for h in hits], net_direction=net,
                        magnitude=float(mag), reasons=reasons)
