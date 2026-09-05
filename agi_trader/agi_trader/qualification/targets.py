"""NET +%1 hedef çözücü ve stop ızgarası — şartname 5, 6, 26, 27, 82, 83.

TEMEL AYRIM
"Piyasa +%1 seviyesi" ile "NET +%1 çıkış seviyesi" AYNI ŞEY DEĞİLDİR.
Kullanıcının amacı `net_profit ≥ %1`. Bunun için gereken BRÜT fiyat hareketi
maliyetlerden çözülür ve toplama ile değil ÇARPMA ile:

    (1+g)(1−c) = 1+n     →     g = (1+n)/(1−c) − 1

Maliyet %0,15 iken net %1 için brüt %1,152 gerekir; %1,15 demek bile eksik
kalır. Panelde gösterilen seviye HER ZAMAN bu g'den türetilir.

STOP DİNAMİKTİR (şartname 26)
Sabit −%0,5 stop yanlıştır: aynı yüzde BTC'de 2 sigma, DOGE'de 0,4 sigmadır.
Stop, o barın ATR'sinin katı olarak bar-başına hesaplanır. Bu, ilk-geçiş
motorunda bedavadır çünkü bariyerler zaten dizi olarak veriliyor.

STOP ÖLÇEĞİ UFKA BAĞLIDIR — ÖLÇEREK BULUNAN HATA
İlk uygulamada stop 24 saatlik ATR'nin katıydı ve bu ATR 5 dakikalık bar
cinsindendi. Sonuç: BTC'de stop tabana (%0,20) yapıştı, R/R 5,7 oldu ve stop
4 saatlik ufukta gözlemlerin %72'sinde önce vuruldu. Hata şuydu: bir barın
oynaklığı 4 saatlik bir kararın ölçeği değildir.

Doğrusu ufuk oynaklığıdır:  σ(H) = σ_bar · √H
Stop = k · σ(H). Böylece aynı k, her ufukta ve her paritede AYNI istatistiksel
sıkılığı ifade eder. R/R buradan TÜREYEN bir sonuçtur, girdi değil.

STOP GENİŞLETEREK KAZANMA YASAĞI (şartname 27)
Stop'u −%3'e açmak "hedef önce" oranını yapay olarak yükseltir. Bu yüzden
stop seçimi asla `argmax P(hedef önce)` ile yapılmaz; RobustEV ile yapılır ve
R/R, beklenen kayıp ve kuyruk riski AYRICA raporlanır.

MALİYET MODELİ BEYANI (şartname 6, 82)
  MEASURED_L2_VWAP : kaydedilmiş kümülatif derinlik eğrisinde gerçek yürüyüş
  ESTIMATED        : eğri yok — doğrusal defter yaklaşımı / sabit varsayım
Gerçek eğri mid'den ölçüldüğü için spread'i ZATEN içerir; o modda spread
ayrıca eklenmez (çift sayım). Bu kural `costs.estimate_costs` içinde ve
`test_spread_cift_sayilamaz` ile kilitlidir.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..opportunity.costs import (CostEstimate, estimate_costs,
                                 required_gross_move_pct, ladder_from_row,
                                 DEFAULT_TAKER_BPS, LATENCY_RESERVE_BPS,
                                 FAILURE_RESERVE_BPS)

TARGET_NET_PCT = 1.0
REFERENCE_NOTIONAL_USD = 1_000.0

# Stop adayları: ufuk oynaklığının katı — k·σ(H). k=1 "bir ufuk-sigması".
STOP_SIGMA_MULTS: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)
STOP_MIN_PCT, STOP_MAX_PCT = 0.15, 6.0
ATR_BARS = 288                      # 24 saat, 5m barda (yalnız giriş bandı için)
VOL_BARS = 288                      # σ_bar tahmini penceresi


@dataclass
class CostProfile:
    """Bir paritenin tarihsel etiketleme için maliyet varsayımı."""
    symbol: str
    spread_bps: float
    impact_bps_roundtrip: float
    fee_bps_roundtrip: float
    reserve_bps: float
    funding_rate_8h: float
    model: str                       # MEASURED_L2_VWAP | ESTIMATED
    source: str
    n_observations: int = 0

    @property
    def base_bps(self) -> float:
        """Ufuktan BAĞIMSIZ maliyet (funding hariç)."""
        return float(self.spread_bps + self.impact_bps_roundtrip
                     + self.fee_bps_roundtrip + self.reserve_bps)

    def total_bps(self, horizon_hours: float, direction: str) -> float:
        """Funding dahil toplam — ASİMETRİK (bkz. `costs.estimate_costs`).

        Ödenecek funding tam uygulanır; tahsil edilecek funding SIFIR sayılır.
        Gelecekteki funding bilinmediği için beklenen kazancı hedefe saymak,
        brüt hedefi net hedefin altına indirir ve bu iyimserliktir."""
        isaret = 1.0 if direction == "LONG" else -1.0
        fund = isaret * self.funding_rate_8h * (horizon_hours / 8.0) * 10_000.0
        return float(self.base_bps + max(0.0, fund))

    def funding_credit_bps(self, horizon_hours: float, direction: str) -> float:
        """Beklenen funding KAZANCI — toplama girmez, raporlanır."""
        isaret = 1.0 if direction == "LONG" else -1.0
        fund = isaret * self.funding_rate_8h * (horizon_hours / 8.0) * 10_000.0
        return float(max(0.0, -fund))

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["base_bps"] = round(self.base_bps, 3)
        return d


def gross_target_pct(profile: CostProfile, horizon_hours: float,
                     direction: str, net_pct: float = TARGET_NET_PCT) -> float:
    """NET hedefe ulaşmak için gereken BRÜT fiyat hareketi (%)."""
    c = CostEstimate()
    c.entry_fee_bps = profile.total_bps(horizon_hours, direction)
    return required_gross_move_pct(net_pct, c)


def atr_pct(df: pd.DataFrame, bars: int = ATR_BARS) -> np.ndarray:
    """Nedensel ATR (%) — Wilder true range, kaydırılmış.

    ⚠️ `.shift(1)` ŞART: i barındaki karar i barının kendi high/low'unu
    kullanamaz. Bu proje daha önce bu satırı unutup Sharpe 3,68 görmüş,
    düzeltince 0,80'e çökmüştü."""
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    onceki = np.empty_like(c)
    onceki[0] = c[0]
    onceki[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - onceki), np.abs(l - onceki)))
    a = pd.Series(tr).ewm(alpha=1.0 / bars, adjust=False,
                          min_periods=bars).mean().shift(1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(c > 0, a / c * 100.0, np.nan)


def sigma_bar_pct(df: pd.DataFrame, bars: int = VOL_BARS) -> np.ndarray:
    """Bar başına gerçekleşen oynaklık (%), NEDENSEL.

    i barında bilinen son getiri log(close[i]/close[i-1])'dir; pencere i'yi
    kapsar ve ileri bakmaz. Giriş de close[i]'de yapıldığı için bu meşrudur."""
    c = df["close"].to_numpy(dtype=float)
    lr = np.zeros(len(c))
    lr[1:] = np.log(c[1:] / c[:-1])
    s = pd.Series(lr).rolling(bars, min_periods=bars // 2).std().to_numpy()
    return s * 100.0


def horizon_sigma_pct(sigma_bar: np.ndarray, horizon_bars: int) -> np.ndarray:
    """σ(H) = σ_bar · √H — bağımsız artışlar varsayımı."""
    return np.asarray(sigma_bar, dtype=float) * np.sqrt(max(1, horizon_bars))


def stop_grid(sigma_bar: np.ndarray, horizon_bars: int,
              mults: Tuple[float, ...] = STOP_SIGMA_MULTS
              ) -> Dict[float, np.ndarray]:
    """k·σ(H) → bar-başına stop yüzdesi (kelepçeli)."""
    sh = horizon_sigma_pct(sigma_bar, horizon_bars)
    return {m: np.clip(sh * m, STOP_MIN_PCT, STOP_MAX_PCT) for m in mults}


def barrier_levels(close: np.ndarray, pct: np.ndarray, direction: str,
                   kind: str) -> np.ndarray:
    """Yüzdeyi mutlak fiyat bariyerine çevirir.

    LONG  hedef → yukarı,  LONG  stop → aşağı
    SHORT hedef → aşağı,   SHORT stop → yukarı
    """
    yukari = (direction == "LONG") == (kind == "target")
    p = np.asarray(pct, dtype=float) / 100.0
    return close * (1.0 + p) if yukari else close * (1.0 - p)


def profile_from_recorder(symbol: str, feats: Optional[pd.DataFrame],
                          notional: float = REFERENCE_NOTIONAL_USD
                          ) -> CostProfile:
    """Kaydedicinin GERÇEK ölçümlerinden maliyet profili.

    Spread ve etki gerçek L2 merdiveninden gelirse model MEASURED_L2_VWAP;
    merdiven yoksa spread yine ölçülmüştür ama etki tahmindir → ESTIMATED.
    """
    if feats is None or not len(feats):
        return CostProfile(symbol, spread_bps=2.0, impact_bps_roundtrip=2.0,
                           fee_bps_roundtrip=2 * DEFAULT_TAKER_BPS,
                           reserve_bps=LATENCY_RESERVE_BPS + FAILURE_RESERVE_BPS,
                           funding_rate_8h=0.0, model="ESTIMATED",
                           source="veri yok — varsayılan sabitler", n_observations=0)
    d = feats[feats["symbol"] == symbol] if "symbol" in feats else feats
    if not len(d):
        return profile_from_recorder(symbol, None, notional)

    spread = float(np.nanmedian(d["spread_bps"])) if "spread_bps" in d else 2.0
    fr = float(np.nanmedian(d["funding_rate"])) if "funding_rate" in d else 0.0

    etki, model, kaynak = None, "ESTIMATED", "spread ölçüldü, etki tahmin"
    merdiven = d[d.get("bid_cum_1bps", pd.Series(dtype=float)).notna()] \
        if "bid_cum_1bps" in d else d.iloc[0:0]
    if len(merdiven):
        from ..opportunity.costs import vwap_offset_bps
        gir, cik = [], []
        for _, row in merdiven.tail(200).iterrows():
            ask = ladder_from_row(row, "ask")
            bid = ladder_from_row(row, "bid")
            if not ask or not bid:
                continue
            a, _ = vwap_offset_bps(ask, notional, min_offset_bps=spread / 2.0)
            b, _ = vwap_offset_bps(bid, notional, min_offset_bps=spread / 2.0)
            if np.isfinite(a) and np.isfinite(b):
                gir.append(a); cik.append(b)
        if gir:
            etki = float(np.median(gir) + np.median(cik))
            model = "MEASURED_L2_VWAP"
            kaynak = (f"{len(gir)} gerçek defter anlık görüntüsünde "
                      f"{notional:,.0f} $ VWAP yürüyüşü")
            spread = 0.0            # ÇİFT SAYIM: gerçek eğri spread'i içerir
    if etki is None:
        etki = 2.0

    return CostProfile(symbol, spread_bps=spread, impact_bps_roundtrip=etki,
                       fee_bps_roundtrip=2 * DEFAULT_TAKER_BPS,
                       reserve_bps=LATENCY_RESERVE_BPS + FAILURE_RESERVE_BPS,
                       funding_rate_8h=fr, model=model, source=kaynak,
                       n_observations=int(len(d)))


def rr_ratio(target_pct: float, stop_pct: float) -> float:
    return float(target_pct / stop_pct) if stop_pct > 0 else float("inf")
