"""MODEL KAYIT DEFTERİ VE MODEL KARTI (§XVI, XXIX, CII).

KURUMSAL MODEL RİSKİ YÖNETİMİNİN ÇEKİRDEĞİ

Bir modelin production'da olması, birinin onu onayladığı anlamına gelmelidir.
Bu dosya "hangi model, ne için, hangi veriyle, hangi kısıtlarla, kim tarafından
doğrulandı" sorusunu kayıt altına alır.

⚠️ AYRILIK İLKESİ (§LVI): modeli geliştiren kişi kendi modelinin son onay
otoritesi OLAMAZ. `approve()` bunu kodla zorlar — `validated_by == built_by`
ise onay reddedilir.

BU SİSTEMDE ŞU AN DURUM
Modeller `owner="cryptomind-research"`, `validated_by=None` ile kayıtlıdır ve
bu yüzden hiçbiri `APPROVED` değildir. Bu bir eksiklik değil, gerçeğin kaydıdır:
bağımsız bir doğrulayıcı yoksa "onaylandı" yazmak, onayın kendisini anlamsız
kılar.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

# §XXIV — kanıt durum makinesi
RESEARCH = "RESEARCH"
BACKTESTED = "BACKTESTED"
OOS_VALIDATED = "OOS_VALIDATED"
SHADOW = "SHADOW"
PAPER = "PAPER"
TESTNET = "TESTNET"
LIMITED_LIVE = "LIMITED_LIVE"
PRODUCTION = "PRODUCTION"
DEGRADED = "DEGRADED"
RETIRED = "RETIRED"

LIFECYCLE = (RESEARCH, BACKTESTED, OOS_VALIDATED, SHADOW, PAPER, TESTNET,
             LIMITED_LIVE, PRODUCTION, DEGRADED, RETIRED)
# Canlı paraya dokunan durumlar — onay ZORUNLU
REQUIRES_APPROVAL = (TESTNET, LIMITED_LIVE, PRODUCTION)

# §XXIX — risk katmanı. Yönlü ML en yüksek katmandadır.
RISK_TIER = {"DIRECTIONAL_ML": 1, "DIRECTIONAL_RULE": 2,
             "MARKET_NEUTRAL": 3, "DESCRIPTIVE": 4}


@dataclass
class ModelCard:
    """§CII — model kartı. Eksik alan `None` kalır, uydurulmaz."""
    model_id: str
    purpose: str
    built_by: str
    asset_universe: List[str]
    horizon: str
    direction: str
    inputs: List[str]
    training_period: str
    validation_period: str
    locked_test_period: str
    version: str
    risk_tier: int
    status: str = RESEARCH
    validated_by: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    prohibited_uses: List[str] = field(default_factory=list)
    known_failure_modes: List[str] = field(default_factory=list)
    retirement_criteria: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    provenance: Dict = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime()))

    def to_dict(self) -> Dict:
        return asdict(self)


class ApprovalError(RuntimeError):
    pass


def approve(card: ModelCard, validated_by: str, approved_by: str,
            target_status: str) -> ModelCard:
    """Onay — ayrılık ilkesini KODLA zorlar.

    Üç kural:
      1. Geliştirici kendi modelini doğrulayamaz.
      2. Geliştirici kendi modelini onaylayamaz.
      3. Canlı paraya dokunan durum için doğrulama ZORUNLU.
    """
    if target_status not in LIFECYCLE:
        raise ApprovalError(f"bilinmeyen durum: {target_status}")
    if validated_by == card.built_by:
        raise ApprovalError(
            "modeli geliştiren kişi kendi modelini DOĞRULAYAMAZ (§LVI)")
    if approved_by == card.built_by:
        raise ApprovalError(
            "modeli geliştiren kişi kendi modelini ONAYLAYAMAZ (§LVI)")
    if target_status in REQUIRES_APPROVAL and not validated_by:
        raise ApprovalError(
            f"{target_status} bağımsız doğrulama olmadan verilemez")
    card.validated_by = validated_by
    card.approved_by = approved_by
    card.approved_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    card.status = target_status
    return card


class Registry:
    """Model envanteri. Ekleme-yalnız değil; durum değişir ama GEÇMİŞ yazılır."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> Dict:
        if not self.path.exists():
            return {"models": {}, "history": []}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"models": {}, "history": []}

    def _write(self, d: Dict) -> None:
        self.path.write_text(json.dumps(d, ensure_ascii=False, indent=1),
                             encoding="utf-8")

    def register(self, card: ModelCard) -> ModelCard:
        d = self._read()
        onceki = d["models"].get(card.model_id)
        if onceki and onceki.get("status") != card.status:
            d["history"].append({
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model_id": card.model_id,
                "from": onceki.get("status"), "to": card.status})
        d["models"][card.model_id] = card.to_dict()
        self._write(d)
        return card

    def get(self, model_id: str) -> Optional[Dict]:
        return self._read()["models"].get(model_id)

    def all(self) -> List[Dict]:
        return sorted(self._read()["models"].values(),
                      key=lambda m: (m.get("risk_tier", 9), m["model_id"]))

    def history(self) -> List[Dict]:
        return self._read().get("history", [])

    def production_ready(self) -> List[Dict]:
        return [m for m in self.all()
                if m.get("status") == PRODUCTION and m.get("approved_by")]


def card_for_softmax(horizon: str, direction: str, symbols: List[str],
                     metrics: Dict, provenance: Dict,
                     train_end: str, valid_end: str) -> ModelCard:
    """Bu programın ürettiği softmax modeli için standart kart.

    Sınırlar ve yasak kullanımlar ÖLÇÜLMÜŞ bulgulardan gelir — temenni değil."""
    return ModelCard(
        model_id=f"net1_softmax_{horizon}_{direction}",
        purpose=("Net +%1 hedefin stop bariyerinden önce görülme olasılığı "
                 "(üç sınıf: hedef / stop / zaman aşımı)"),
        built_by="cryptomind-research",
        asset_universe=list(symbols),
        horizon=horizon, direction=direction,
        inputs=["technical", "flow", "geometry", "cross_asset", "time"],
        training_period=f"< {train_end}",
        validation_period=f"{train_end} → {valid_end}",
        locked_test_period=f"≥ {valid_end}",
        version="softmax_l2/1", risk_tier=RISK_TIER["DIRECTIONAL_ML"],
        status=RESEARCH,
        assumptions=[
            "5 dakikalık bar, hedef ve stop aynı barda vurulursa örnek düşer",
            "maliyet gidiş-dönüş ve giriş anında ölçülen L2 eğrisinden",
            "stop k·σ(H) ile ufka göre ölçeklenir",
        ],
        limitations=[
            "ÖLÇÜLDÜ: model oynaklığı tahmin ediyor, YÖNÜ değil",
            "ÖLÇÜLDÜ: kilitli testte üst desil dev'e göre ~5 puan düşüyor",
            "kör taban rejime koşullu (BTC 4h: %8 → %48)",
            "tek venue (Binance USD-M vadeli)",
        ],
        prohibited_uses=[
            "tek başına emir üretmek",
            "gösterge konsensüsü ya da formasyon yönüyle birleştirmek "
            "(ikisinin de OOS yön kenarı ölçüldü ve çıkmadı)",
            "kalibre edilmemiş eşikle sıralama",
        ],
        known_failure_modes=[
            "rejim değişiminde kalibrasyon kayması",
            "ince defterli paritede maliyetin hedefi yemesi",
        ],
        retirement_criteria=[
            "PSI > 0,25", "ECE > 0,05", "kalibrasyon eğimi [0,7 – 1,4] dışı",
            "gerçekleşen/tahmin oranı 30 gün boyunca < 0,5",
        ],
        metrics=dict(metrics), provenance=dict(provenance))


STATUS_TR = {
    RESEARCH: "araştırma", BACKTESTED: "geriye dönük test",
    OOS_VALIDATED: "örneklem dışı doğrulandı", SHADOW: "gölge",
    PAPER: "kâğıt", TESTNET: "test ağı", LIMITED_LIVE: "sınırlı canlı",
    PRODUCTION: "üretim", DEGRADED: "bozulmuş", RETIRED: "emekli",
}
