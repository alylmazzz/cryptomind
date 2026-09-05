"""Kural-tabanlı strateji kütüphanesi + ATR iz-süren stop simülatörü + fırsat tarayıcı.

Derin-öğrenme hattından bağımsız, vektörize ve hızlı stratejiler. Amaç: çok-parite ×
çok-zaman-dilimi tarayarak BELİRLİ bir pencerede gerçek (kural-tabanlı, genellenebilir)
bir kenarın NEREDE olduğunu bulmak — overfitting yapmadan, in-sample/out-of-sample
ayrımıyla.
"""
from .library import (
    STRATEGIES, strat_trend, strat_breakout, strat_meanrev,
    simulate_trailing, scan_opportunities,
)
from .freqtrade import (
    FREQTRADE_STRATEGIES, simulate_freqtrade, run_strategy,
)

__all__ = [
    "STRATEGIES", "strat_trend", "strat_breakout", "strat_meanrev",
    "simulate_trailing", "scan_opportunities",
    "FREQTRADE_STRATEGIES", "simulate_freqtrade", "run_strategy",
]
