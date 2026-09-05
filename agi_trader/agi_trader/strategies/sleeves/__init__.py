"""
Sleeve'ler — birbirinden bağımsız getiri akışları.

Kullanım:
    from agi_trader.strategies.sleeves import build_sleeve, ALL_SLEEVES
    s = build_sleeve("xsec_momentum", lookback=90)
    r = s.returns(price_frame)
"""
from .base import Sleeve, DEFAULT_COST, vol_target_scale, zscore, cross_sectional_rank
from .price_sleeves import (
    TrendSleeve, CrossSectionalSleeve, ShortReversalSleeve, TermStructureSleeve,
    PRICE_SLEEVES,
)
from .carry import CarrySleeve

ALL_SLEEVES = {**PRICE_SLEEVES, "carry": CarrySleeve}


def build_sleeve(name: str, **kwargs) -> Sleeve:
    if name not in ALL_SLEEVES:
        raise KeyError(f"bilinmeyen sleeve: {name} (mevcut: {sorted(ALL_SLEEVES)})")
    return ALL_SLEEVES[name](**kwargs)


__all__ = [
    "Sleeve", "DEFAULT_COST", "vol_target_scale", "zscore", "cross_sectional_rank",
    "TrendSleeve", "CrossSectionalSleeve", "ShortReversalSleeve",
    "TermStructureSleeve", "CarrySleeve", "ALL_SLEEVES", "build_sleeve",
]
