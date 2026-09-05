"""
Araştırma / doğrulama katmanı.

Bu paket STRATEJİ ÜRETMEZ — üretilen stratejilerin gerçek mi yoksa aşırı uyum
(overfit) mu olduğunu ölçer. Projede bir kez "+%61,8" sonucu üretilip sonradan
%100 overfit çıktığı için, yeni her sleeve/özellik buradaki kapıdan geçmek
zorundadır (bkz. `validation.acceptance_gate`).
"""
from .validation import (
    psr,
    deflated_sharpe,
    expected_max_sharpe,
    min_backtest_length,
    purged_kfold_splits,
    combinatorial_purged_splits,
    pbo,
    sharpe,
    trial_log,
    trial_count,
    acceptance_gate,
    AcceptanceResult,
)

__all__ = [
    "psr", "deflated_sharpe", "expected_max_sharpe", "min_backtest_length",
    "purged_kfold_splits", "combinatorial_purged_splits", "pbo", "sharpe",
    "trial_log", "trial_count", "acceptance_gate", "AcceptanceResult",
]
