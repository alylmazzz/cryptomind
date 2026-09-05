"""
Multi-Agent Orchestrator (Multi-Agent Sistem Mimarisi rolü).

Her analiz katmanı bir "uzman ajan" gibi davranır ve bir LayerVote döndürür.
Orchestrator bu ajanları koordine eder, çıktılarını karar motoruna iletir,
sinyali execution'a ve journal'a yönlendirir. Hata izolasyonu vardır: bir ajan
çökerse diğerleri çalışmaya devam eder (graceful degradation).

Pipeline (her parite için):
  veri (MTF) -> [technical, pattern, smc, multi_timeframe, ai_ensemble,
                 sentiment, onchain, macro] -> decision -> execution -> journal
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ..core.models import AnalysisResult, LayerVote, TradeSignal
from ..data import ExchangeManager
from ..analysis import (
    compute_all_indicators, technical_vote, detect_patterns, smc_vote,
    multi_timeframe_vote, trendline_vote,
)
from ..analysis.indicators import atr as atr_fn
from ..analysis.patterns import extreme_analysis
from ..ai import ensemble_vote
from ..sentiment import TwitterIntelligence, sentiment_vote
from ..sentiment.fear_greed import fear_greed_vote
from ..sentiment.news import news_vote
from ..risk import RiskEngine
from ..decision import DecisionEngine
from ..execution import ExecutionEngine
from ..journal import TradeJournal
from ..onchain import WhaleFlowEngine, onchain_vote
from ..notify import Notifier
from ..notify.alarms import AlarmEngine
from ..analysis.patterns import find_confluence
from ..core.models import Direction
from ..learn import load_learned_weights
from .extra_layers import macro_vote


class Orchestrator:
    def __init__(self, config):
        self.config = config
        self.data = ExchangeManager(config)
        self.whale = WhaleFlowEngine(config, self.data)
        self.twitter = TwitterIntelligence(config)
        self.risk = RiskEngine(config)
        self.decision = DecisionEngine(config, self.risk)
        self.execution = ExecutionEngine(config)
        self.journal = TradeJournal(config.get("output_dir", "runs"))
        self.notifier = Notifier(config)
        self.alarms = AlarmEngine(self.notifier)
        self.primary_tf = config.get("primary_timeframe", "4h")
        # öğrenilmiş ağırlıklar varsa uygula (#7 sürekli öğrenme)
        learned = load_learned_weights(config.get("output_dir", "runs"))
        if learned:
            self.decision.base_weights = learned

    def _safe(self, fn, fallback_name: str) -> LayerVote:
        try:
            return fn()
        except Exception as e:
            return LayerVote(name=fallback_name, score=0.0, confidence=0.0,
                             reasons=[f"Katman hatası: {type(e).__name__}: {e}"])

    def analyze_symbol(self, symbol: str) -> TradeSignal:
        # 1) Veri (çok zaman dilimli)
        mtf = self.data.fetch_multi_timeframe(symbol)
        if self.primary_tf not in mtf:
            # birincil TF yoksa eldeki en yüksek TF'yi kullan
            mtf[self.primary_tf] = self.data.fetch_ohlcv(symbol, self.primary_tf)
        df = mtf[self.primary_tf]
        last_price = float(df["close"].iloc[-1])
        atr_val = float(atr_fn(df, 14).iloc[-1])

        # 2) İndikatörler (birincil TF)
        indicators = compute_all_indicators(df)

        # 3) Ajan oyları (hata izolasyonlu)
        votes: List[LayerVote] = []
        patterns_list = []

        votes.append(self._safe(lambda: technical_vote(df, indicators), "technical"))

        def _pattern_vote():
            pats, v = detect_patterns(df)
            patterns_list.extend(pats)
            return v
        votes.append(self._safe(_pattern_vote, "pattern"))

        votes.append(self._safe(lambda: trendline_vote(df), "trendline"))
        votes.append(self._safe(lambda: smc_vote(df), "smc"))
        votes.append(self._safe(lambda: multi_timeframe_vote(mtf), "multi_timeframe"))
        votes.append(self._safe(lambda: ensemble_vote(df, symbol, self.primary_tf), "ai_ensemble"))
        votes.append(self._safe(lambda: sentiment_vote(self.twitter, symbol), "sentiment"))
        votes.append(self._safe(lambda: fear_greed_vote(), "fear_greed"))
        votes.append(self._safe(lambda: news_vote(symbol, self.config), "news"))
        votes.append(self._safe(lambda: onchain_vote(self.whale, symbol, df), "onchain"))
        votes.append(self._safe(lambda: macro_vote(df, self.config), "macro"))

        # 4) Analiz sonucunu paketle
        analysis = AnalysisResult(
            symbol=symbol, timeframe=self.primary_tf, last_price=last_price,
            votes=votes, patterns=patterns_list, indicators=indicators,
            extremes=extreme_analysis(df),
        )

        # 5) Karar (df + ensemble p_up -> tahmin & baskı için)
        p_up = next((v.detail.get("p_up", 0.5) for v in votes if v.name == "ai_ensemble"), 0.5)
        signal = self.decision.decide(symbol, self.primary_tf, last_price, atr_val,
                                      votes, analysis, df=df, p_up=p_up)
        # extremes'i sinyale iliştir (rapor için)
        signal.layer_breakdown.append({"layer": "_extremes", "detail": analysis.extremes})

        # formasyon + birleşim özetini sinyale iliştir (alarm + rapor için)
        last_idx = len(df) - 1
        conf = find_confluence(patterns_list)
        pat_summary = []
        for p in patterns_list:
            apex = p.points.get("D") or p.points.get("top2") or p.points.get("bottom2") \
                or p.points.get("right") or (next(iter(p.points.values())) if p.points else 0.0)
            pat_summary.append({
                "name": p.name, "family": p.family, "direction": p.direction.value,
                "quality": p.quality, "apex_price": float(apex),
                "near_last": bool(p.pivot_index >= last_idx - 2),
            })
        signal.layer_breakdown.append({"layer": "_formations",
                                       "detail": {"patterns": pat_summary, "confluence": conf}})

        # piyasa rejimi (HMM) — dinamik pozisyon boyutlama için sinyale iliştir
        try:
            from ..analysis.regime import detect_regime
            signal.layer_breakdown.append({"layer": "_regime", "detail": detect_regime(df)})
        except Exception:
            pass
        return signal

    def run(self, symbols: Optional[List[str]] = None) -> List[TradeSignal]:
        symbols = symbols or self.config.symbols
        results: List[TradeSignal] = []
        for sym in symbols:
            try:
                sig = self.analyze_symbol(sym)
            except Exception as e:
                continue
            exec_res = self.execution.execute(sig)
            self.journal.record(sig, exec_res)
            sig.layer_breakdown.append({"layer": "_execution", "detail": exec_res})
            # alarm kontrolü (formasyon / birleşim / seviye kırılımı / işlem adayı)
            try:
                self.alarms.check_signal(sig)
            except Exception:
                pass
            results.append(sig)
        # korelasyon rozeti: her pariteyi referans (BTC) ile ilişkilendir (contagion)
        try:
            self._attach_correlation(results)
        except Exception:
            pass

        # #4 işlem adayı varsa bildirim gönder
        try:
            if self.notifier.enabled:
                self.notifier.notify_actionable(results)
        except Exception:
            pass
        return results

    def _attach_correlation(self, signals: List[TradeSignal]) -> None:
        """Pariteler arası getiri korelasyonu → her sinyale referansla rozet ekler."""
        if len(signals) < 2:
            return
        ref = next((s.symbol for s in signals if s.symbol.upper().startswith("BTC")), signals[0].symbol)
        rets: Dict[str, "pd.Series"] = {}
        for s in signals:
            df = self.data.fetch_ohlcv(s.symbol, self.primary_tf)
            if df is not None and len(df) > 30:
                rets[s.symbol] = df["close"].pct_change().dropna()
        if ref not in rets:
            return
        n = min(min(len(v) for v in rets.values()), 200)
        ref_r = rets[ref].tail(n).reset_index(drop=True)
        for s in signals:
            if s.symbol not in rets:
                continue
            if s.symbol == ref:
                s.correlation_badge = {"symbol": ref, "value": 1.0, "self": True}
                continue
            try:
                corr = float(ref_r.corr(rets[s.symbol].tail(n).reset_index(drop=True)))
                s.correlation_badge = {"symbol": ref, "value": round(corr, 2)}
            except Exception:
                pass

    def describe_environment(self) -> Dict:
        from ..sentiment.accounts import account_stats
        from ..sentiment.fear_greed import get_fear_greed
        stats = account_stats()
        wh = self.whale
        fg = get_fear_greed()
        return {
            "data_source": self.data.describe(),
            "twitter_live": self.twitter.live,
            "fear_greed": fg,  # {value,label,score} veya None
            "news_live": bool(self.config.secret("CRYPTOPANIC_API_KEY") or self.config.secret("NEWSAPI_API_KEY")),
            "macro_live": bool(self.config.secret("FRED_API_KEY")),
            "execution": self.execution.status(),
            "tracked_accounts": stats["total"],
            "accounts_crypto": stats["crypto_exchange"],
            "accounts_political": stats["political_macro"],
            "onchain_live": self.data.live_enabled,
            "deep_learning": _torch_available(),
            # 3-tier on-chain durumu
            "onchain_tiers": {
                "tier1_always": bool(wh._spot_client()),  # ccxt var mı?
                "tier2_free": {
                    "etherscan": bool(wh.etherscan_key),
                    "coingecko": bool(wh.coingecko_key),
                    "coinmarketcap": bool(wh.coinmarketcap_key),
                    "blockchaincom": bool(wh.blockchaincom_key),
                },
                "tier3_premium": {
                    "whale_alert": bool(wh.whale_alert_key),
                },
                "tiers_active": [
                    t for t in ["T1"] +
                    (["T2"] if any([wh.etherscan_key, wh.coingecko_key,
                                    wh.coinmarketcap_key, wh.blockchaincom_key]) else []) +
                    (["T3"] if wh.whale_alert_key else [])
                ],
            },
        }


def _torch_available() -> bool:
    from ..core.light import LIGHT_MODE
    if LIGHT_MODE:          # hafif modda torch'u import bile etme (bellek)
        return False
    try:
        import torch  # noqa
        return True
    except Exception:
        return False
