"""Sistem genelinde kullanılan veri modelleri ve enum'lar."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"          # işlem yok


class Bias(str, Enum):
    BULLISH = "BOĞA"       # yükseliş
    BEARISH = "AYI"        # düşüş
    NEUTRAL = "NÖTR"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class LayerVote:
    """
    Tek bir analiz katmanının çıktısı.

    score: -1 (kuvvetli ayı) .. +1 (kuvvetli boğa)
    confidence: 0..1  bu katmanın kendi kararına ne kadar güvendiği
    weight: bu katmanın karar motorundaki ağırlığı (dinamik)
    reasons: insan-okur gerekçeler (Explainable AI)
    """
    name: str
    score: float
    confidence: float
    weight: float = 0.0
    reasons: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def bias(self) -> Bias:
        if self.score > 0.15:
            return Bias.BULLISH
        if self.score < -0.15:
            return Bias.BEARISH
        return Bias.NEUTRAL

    @property
    def weighted_score(self) -> float:
        return self.score * self.confidence * self.weight


@dataclass
class PatternMatch:
    """Tespit edilen formasyon."""
    name: str                      # "Gartley", "Head & Shoulders", "FVG" ...
    family: str                    # "harmonic" | "classic" | "smc"
    direction: Direction           # tamamlanınca beklenen yön
    completion: float              # tamamlanma yüzdesi 0..1
    quality: float                 # 0..1 uyum/kalite skoru
    pivot_index: int               # grafik üzerinde konum (bar index)
    points: Dict[str, float] = field(default_factory=dict)   # X,A,B,C,D fiyatları
    indices: Dict[str, int] = field(default_factory=dict)    # aynı noktaların bar indeksleri (çizim için)
    target: Optional[float] = None
    invalidation: Optional[float] = None
    note: str = ""
    ratios: Dict[str, float] = field(default_factory=dict)   # sayısal oranlar (Fib vb.) — etikette gösterilir


@dataclass
class RiskAnalysis:
    symbol: str
    recommended_position_size: float      # USDT cinsi
    position_pct: float                   # portföyün %'si
    kelly_fraction: float
    risk_reward: float
    expected_value_pct: float
    value_at_risk_95: float
    value_at_risk_99: float
    conditional_var: float
    stop_loss: float
    take_profits: List[float] = field(default_factory=list)
    mc_win_probability: float = 0.0
    mc_expected_return: float = 0.0
    mc_max_drawdown: float = 0.0
    portfolio_heat: float = 0.0
    expected_sharpe: float = 0.0


@dataclass
class AnalysisResult:
    """Tek parite + zaman dilimi için tüm katman çıktıları."""
    symbol: str
    timeframe: str
    last_price: float
    timestamp: datetime = field(default_factory=_now)
    votes: List[LayerVote] = field(default_factory=list)
    patterns: List[PatternMatch] = field(default_factory=list)
    indicators: Dict[str, float] = field(default_factory=dict)
    extremes: Dict[str, Any] = field(default_factory=dict)   # maks/min noktalar


@dataclass
class TradeSignal:
    """Karar motorunun nihai, açıklanabilir çıktısı."""
    symbol: str
    direction: Direction
    bias: Bias
    confidence: float                     # 0..1 (>=0.90 ise işleme uygun)
    entry: float
    stop_loss: float
    take_profits: List[float]
    risk_reward: float
    success_probability: float
    expected_return_pct: float
    expected_loss_pct: float
    invalidation: float
    timeframe: str
    timestamp: datetime = field(default_factory=_now)

    # Alış / satış baskısı (0..100, toplamı 100)
    buy_pressure_pct: float = 50.0
    sell_pressure_pct: float = 50.0
    pressure_label: str = "DENGELİ"

    # Sonraki periyodun beklenen maks/min tahmini
    forecast: Dict[str, Any] = field(default_factory=dict)

    # Açıklanabilirlik
    layer_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    alternative_scenario: str = ""
    risk: Optional[RiskAnalysis] = None
    actionable: bool = False              # min_confidence eşiğini geçti mi
    signal_class: str = ""                 # kesin_al / zayif_al / notr / zayif_sat / kesin_sat / acil_cikis
    momentum_score: int = 50               # 0-100 momentum skoru
    volatility: str = "medium"             # low / medium / high / extreme
    correlation_badge: Optional[Dict[str, Any]] = None  # örn: {"symbol": "EUR/USD", "value": 0.87}

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        d = asdict(self)
        d["direction"] = self.direction.value
        d["bias"] = self.bias.value
        d["timestamp"] = self.timestamp.isoformat()
        return d
