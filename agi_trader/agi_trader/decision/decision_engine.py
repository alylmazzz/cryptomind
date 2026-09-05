"""
Karar Motoru (Multi-Agent System + Quant rolü).

Spec:
  - Her analiz katmanına dinamik ağırlık ver (teknik, order flow, likidite,
    on-chain, makro, sentiment, twitter, whale, funding, OI, volatilite, AI).
  - Sistem tek sinyalle işlem AÇMAMALI.
  - En az %90 güven oluşmadan sinyal üretmemeli.
  - Çıktı: LONG/SHORT, entry, SL, TP1/2/3, R/R, başarı olasılığı, beklenen
    getiri/kayıp, iptal seviyesi, alternatif senaryo, confidence score — ve
    her kararın GEREKÇESİYLE (Explainable AI).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..core.models import (
    AnalysisResult, Bias, Direction, LayerVote, RiskAnalysis, TradeSignal,
)
from ..risk.risk_engine import RiskEngine


class DecisionEngine:
    def __init__(self, config, risk_engine: RiskEngine):
        self.config = config
        self.risk = risk_engine
        dc = config.get("decision", {})
        self.min_confidence = float(dc.get("min_confidence", 0.62))
        self.min_rr = float(dc.get("min_risk_reward", 1.3))
        # Konsensüs kapısı: "her şey aynı sinyali gösteriyorsa gir"
        self.consensus_min_agreement = float(dc.get("consensus_min_agreement", 0.70))
        self.consensus_min_layers = int(dc.get("consensus_min_layers", 3))
        self.news_veto = bool(dc.get("news_veto", True))
        self.news_veto_min_conf = float(dc.get("news_veto_min_conf", 0.45))
        self.trend_gate_min_adx = float(dc.get("trend_gate_min_adx", 0.0))  # 0 = kapalı
        # Parite-bazlı trend kapısı (kalibrasyondan): {symbol: min_adx}
        self.pair_trend_gate: Dict[str, float] = {
            k: float(v) for k, v in dict(dc.get("pair_trend_gate", {})).items()}
        self.base_weights: Dict[str, float] = dict(dc.get("layer_weights", {}))

    def _effective_weights(self, votes: List[LayerVote]) -> Dict[str, float]:
        """Katman ağırlıklarını normalize et (yalnızca mevcut katmanlar)."""
        w = {v.name: self.base_weights.get(v.name, 0.05) for v in votes}
        total = sum(w.values()) + 1e-12
        return {k: val / total for k, val in w.items()}

    def decide(self, symbol: str, timeframe: str, last_price: float, atr: float,
               votes: List[LayerVote], analysis: AnalysisResult,
               df=None, p_up: float = 0.5) -> TradeSignal:
        weights = self._effective_weights(votes)
        for v in votes:
            v.weight = weights.get(v.name, 0.0)

        # Ağırlıklı toplam skor (katman güveniyle modüle)
        num = sum(v.score * v.confidence * v.weight for v in votes)
        den = sum(v.confidence * v.weight for v in votes) + 1e-12
        agg_score = num / den                          # -1..1

        direction = (Direction.LONG if agg_score > 0.12
                     else Direction.SHORT if agg_score < -0.12
                     else Direction.FLAT)
        bias = (Bias.BULLISH if agg_score > 0.12
                else Bias.BEARISH if agg_score < -0.12 else Bias.NEUTRAL)

        # Hizalanma (agreement): katmanların ağırlıklı yön uyumu
        if direction != Direction.FLAT:
            sign = 1 if agg_score > 0 else -1
            agree_w = sum(v.weight for v in votes if np.sign(v.score) == sign and abs(v.score) > 0.1)
            total_active = sum(v.weight for v in votes if abs(v.score) > 0.1) + 1e-12
            agreement = agree_w / total_active
        else:
            agreement = 0.0

        avg_conf = float(np.mean([v.confidence for v in votes])) if votes else 0.0

        # Nihai güven: skor büyüklüğü + hizalanma + ortalama katman güveni
        confidence = (0.45 * abs(agg_score) + 0.35 * agreement + 0.20 * avg_conf)
        confidence = float(np.clip(confidence, 0.0, 0.99))

        # Başarı olasılığı (confidence'tan kalibre)
        success_prob = float(np.clip(0.45 + 0.45 * confidence, 0.45, 0.92))

        # Risk analizi + seviyeler
        # Hibrit mod için ADX (güçlü trend → ATR-R, zayıf → scalp)
        adx_val = None
        if analysis and analysis.indicators:
            adx_val = analysis.indicators.get("adx_14")
        tp_mode_used = self.risk.resolve_mode(adx_val)

        risk: Optional[RiskAnalysis] = None
        entry = last_price
        if direction != Direction.FLAT and atr > 0:
            risk = self.risk.analyze(symbol, direction, entry, atr, success_prob,
                                     mode=tp_mode_used)
            stop = risk.stop_loss
            tps = risk.take_profits
            rr = risk.risk_reward
        else:
            stop = entry
            tps = [entry, entry, entry]
            rr = 0.0

        expected_return_pct = abs(tps[0] - entry) / (entry + 1e-12) * 100 if direction != Direction.FLAT else 0.0
        expected_loss_pct = abs(entry - stop) / (entry + 1e-12) * 100 if direction != Direction.FLAT else 0.0
        invalidation = stop

        # --- KONSENSÜS KAPISI: "her şey aynı sinyali gösteriyorsa gir" ---
        # Aktif (kararlı) katman = belirgin skoru + bir miktar güveni olan.
        active = [v for v in votes if abs(v.score) > 0.1 and v.confidence > 0.05]
        n_active = len(active)
        consensus_ok = (agreement >= self.consensus_min_agreement
                        and n_active >= self.consensus_min_layers)

        # --- HABER VETOSU: haber ters yönde + yeterince güvenliyse işlemi engelle ---
        veto = False
        veto_reason = ""
        if self.news_veto and direction != Direction.FLAT:
            sgn = 1 if agg_score > 0 else -1
            nv = next((v for v in votes if v.name == "news"), None)
            if (nv and nv.confidence >= self.news_veto_min_conf
                    and np.sign(nv.score) == -sgn and abs(nv.score) > 0.15):
                veto = True
                veto_reason = f"Haber katmanı ters yönde (skor {nv.score:+.2f}) — işlem VETO."

        # NOT: rr karşılaştırmasında küçük tolerans — ATR modunda rr tam olarak
        # hedef katına (örn. 1.5) eşittir ama risk_engine paydadaki 1e-12 yüzünden
        # rr'yi 1.4999…'a düşürür; bu, fiyat ölçeğine bağlı knife-edge hatasıdır
        # (BTC geçer, ETH kıl payı kalır). Tolerans bunu giderir.
        rr_ok = rr >= (self.min_rr - 1e-6)
        # --- TREND KAPISI: choppy (zayıf ADX) piyasada işlem açma ---
        # (alt-coin'lerin yatay/choppy dönemlerde verdiği whipsaw kayıpları eler)
        trend_ok = True
        eff_min_adx = self.pair_trend_gate.get(symbol, self.trend_gate_min_adx)
        if eff_min_adx >= 999:
            trend_ok = False           # 999 = bu pariteyi hiç işleme (kalibrasyon eledi)
        elif eff_min_adx > 0 and adx_val is not None:
            trend_ok = adx_val >= eff_min_adx
        actionable = (
            direction != Direction.FLAT
            and consensus_ok
            and confidence >= self.min_confidence
            and rr_ok
            and trend_ok
            and not veto
        )
        gate = {"consensus_ok": consensus_ok, "agreement": round(agreement, 3),
                "n_active": n_active, "veto": veto, "veto_reason": veto_reason,
                "min_agreement": self.consensus_min_agreement,
                "min_layers": self.consensus_min_layers,
                "tp_mode": tp_mode_used, "adx": round(adx_val, 1) if adx_val is not None else None,
                "trend_ok": trend_ok, "trend_gate_min_adx": eff_min_adx}

        # Açıklanabilirlik
        breakdown = [{
            "layer": v.name,
            "bias": v.bias.value,
            "score": round(v.score, 3),
            "confidence": round(v.confidence, 3),
            "weight": round(v.weight, 3),
            "contribution": round(v.score * v.confidence * v.weight, 4),
            "reasons": v.reasons,
        } for v in sorted(votes, key=lambda x: -abs(x.score * x.confidence * x.weight))]

        reasons = self._summary_reasons(votes, agg_score, agreement, confidence,
                                        actionable, gate)
        alt = self._alternative_scenario(direction, entry, stop, tps, invalidation)

        # Sonraki periyot maks/min tahmini + alış/satış baskısı
        forecast: Dict = {}
        buy_pct, sell_pct, pressure_label = 50.0, 50.0, "DENGELİ"
        if df is not None and len(df) > 5:
            try:
                from ..analysis.forecast import forecast_next, buy_sell_pressure
                forecast = forecast_next(df, agg_score, p_up, confidence)
                buy_pct, sell_pct, pressure_label = buy_sell_pressure(df, agg_score)
            except Exception:
                pass

        # AL / SAT eşikleri (destek/direnç) — sağ panel için sayısal seviyeler
        buy_th, sell_th = _buy_sell_thresholds(last_price, analysis.indicators, analysis.extremes)
        forecast["buy_threshold"] = buy_th
        forecast["sell_threshold"] = sell_th
        forecast["price"] = float(last_price)
        forecast["gate"] = gate

        # 6-tier sinyal sınıflandırması
        signal_class = cls = _classify_signal(direction, confidence)
        momentum = _calc_momentum(votes, agg_score)
        vol = _calc_volatility(df, last_price, atr) if df is not None and len(df) > 5 else "medium"

        return TradeSignal(
            symbol=symbol,
            direction=direction,
            bias=bias,
            confidence=round(confidence, 4),
            entry=float(entry),
            stop_loss=float(stop),
            take_profits=[float(t) for t in tps],
            risk_reward=round(float(rr), 2),
            success_probability=round(success_prob, 3),
            expected_return_pct=round(expected_return_pct, 2),
            expected_loss_pct=round(expected_loss_pct, 2),
            invalidation=float(invalidation),
            timeframe=timeframe,
            buy_pressure_pct=buy_pct,
            sell_pressure_pct=sell_pct,
            pressure_label=pressure_label,
            forecast=forecast,
            layer_breakdown=breakdown,
            reasons=reasons,
            alternative_scenario=alt,
            risk=risk,
            actionable=actionable,
            signal_class=signal_class,
            momentum_score=momentum,
            volatility=vol,
        )

    def _summary_reasons(self, votes, agg_score, agreement, confidence, actionable,
                         gate=None) -> List[str]:
        gate = gate or {}
        out = [
            f"Birleşik skor: {agg_score:+.3f} | Katman hizalanması: {agreement*100:.0f}% "
            f"({gate.get('n_active', 0)} aktif katman) | Güven: {confidence*100:.1f}%",
        ]
        if not actionable:
            if gate.get("veto"):
                out.append(f"⛔ İŞLEM YOK — {gate.get('veto_reason', 'haber vetosu')}")
            elif not gate.get("trend_ok", True):
                out.append(f"⛔ İŞLEM YOK — trend zayıf (ADX {gate.get('adx')} < "
                           f"{gate.get('trend_gate_min_adx')}); choppy piyasada işlem açılmaz.")
            elif not gate.get("consensus_ok", True):
                out.append(f"⛔ İŞLEM YOK — konsensüs yetersiz: hizalanma "
                           f"%{agreement*100:.0f} < %{self.consensus_min_agreement*100:.0f} "
                           f"ya da aktif katman {gate.get('n_active',0)} < {self.consensus_min_layers}.")
            elif confidence < self.min_confidence:
                out.append(f"⛔ İŞLEM YOK — güven %{confidence*100:.0f} < %{self.min_confidence*100:.0f}.")
            else:
                out.append(f"⛔ İŞLEM YOK — R/R {self.min_rr} altında.")
        else:
            out.append(f"✅ İŞLEM ADAYI — {gate.get('n_active',0)} katmanın %{agreement*100:.0f}'i "
                       f"aynı yönde + güven %{confidence*100:.0f} + eşik aşıldı.")
        # en güçlü 3 katkı
        top = sorted(votes, key=lambda x: -abs(x.score * x.confidence * x.weight))[:3]
        for v in top:
            if v.reasons:
                out.append(f"[{v.name}] {v.reasons[0]}")
        return out

    def _alternative_scenario(self, direction, entry, stop, tps, invalidation) -> str:
        if direction == Direction.FLAT:
            return "Net yön yok; fiyat range içinde — kırılım beklenmeli."
        if direction == Direction.LONG:
            return (f"Fiyat {invalidation:.4f} altında kapanırsa boğa senaryosu iptal; "
                    f"yapı ayıya döner, short fırsatı aranır.")
        return (f"Fiyat {invalidation:.4f} üstünde kapanırsa ayı senaryosu iptal; "
                f"yapı boğaya döner, long fırsatı aranır.")
# ---------------------------------------------------------------------------
# Sinyal sınıflandırma yardımcıları
# ---------------------------------------------------------------------------
def _buy_sell_thresholds(price: float, indicators: Dict, extremes: Dict):
    """AL eşiği (en yakın güçlü destek ≤ fiyat) ve SAT eşiği (en yakın direnç ≥ fiyat).
    Pivot + Fibonacci + son-20 swing seviyelerinden türetilir."""
    ind = indicators or {}
    ex = extremes or {}
    supports = [ind.get("pp_s1"), ind.get("pp_s2"), ind.get("fib_618"), ind.get("fib_786"),
                ex.get("recent_low_20"), ind.get("bb_lower")]
    resists = [ind.get("pp_r1"), ind.get("pp_r2"), ind.get("fib_382"), ind.get("fib_236"),
               ex.get("recent_high_20"), ind.get("bb_upper")]
    below = [s for s in supports if isinstance(s, (int, float)) and 0 < s < price]
    above = [r for r in resists if isinstance(r, (int, float)) and r > price]
    buy_th = max(below) if below else round(price * 0.985, 8)      # en yakın destek
    sell_th = min(above) if above else round(price * 1.015, 8)     # en yakın direnç
    return float(buy_th), float(sell_th)


def _classify_signal(direction: Direction, confidence: float) -> str:
    """6-tier sinyal sınıflandırması (prompt.txt spec).
    🟢 KESİN AL/SAT: >= 0.85 | 🟡 ZAYIF: >= 0.40 | ⚪ NÖTR: FLAT veya < 0.40"""
    if confidence >= 0.85:
        return "kesin_al" if direction == Direction.LONG else "kesin_sat"
    if confidence >= 0.40:
        return "zayif_al" if direction == Direction.LONG else "zayif_sat"
    if direction == Direction.FLAT:
        return "notr"
    return "notr"


def _calc_momentum(votes: List[LayerVote], agg_score: float) -> int:
    """0-100 momentum skoru."""
    tech_votes = [v for v in votes if v.name in ("technical", "ai_ensemble", "multi_timeframe")]
    if tech_votes:
        raw = float(np.mean([abs(v.score) for v in tech_votes]))
    else:
        raw = abs(agg_score)
    momentum = int(np.clip(raw * 85 + 15, 0, 100))
    return momentum


def _calc_volatility(df, price: float, atr: float) -> str:
    """Volatilite sınıflandırması: low / medium / high / extreme."""
    if df is None or len(df) < 20:
        return "medium"
    atr_pct = atr / (price + 1e-9)
    if atr_pct < 0.01:
        return "low"
    if atr_pct < 0.03:
        return "medium"
    if atr_pct < 0.06:
        return "high"
    return "extreme"
