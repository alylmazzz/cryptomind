"""Taban oran motoru — şartname 8, 16, 21, 22, 28, 42.

BU MOTOR MODEL DEĞİLDİR. KÖR TABANDIR.

"Rastgele bir anda girsem, net +%1 hedef stop'tan önce ne sıklıkla gelir?"
Bir modelin başarısı ancak BU sayıya göre ölçülebilir. %75 hedef-önce oranı,
taban %73 ise sıfır bilgidir. Ham doğruluk göstermek yasaktır (şartname 8).

ÖLÇÜLENLER — hepsi gerçek 5 dakikalık vadeli barlardan
  • TP-first / SL-first / timeout oranı
  • BELİRSİZ (aynı bar) oranı — ölçümden düşer, ayrıca raporlanır
  • MFE / MAE dağılımı (P10…P90) — hedef, dağılımın gerçekçi olmayan
    kuyruğundaysa sinyal üretilmemeli (şartname 22)
  • medyan hedefe/stopa varış süresi (şartname 95, 96)
  • ZAMAN AŞIMI GETİRİSİ — sıfır VARSAYILMAZ, ölçülür (şartname 28)
  • boşluklu açılışta stop kayması — kuyruk cezasının ölçülen tarafı
  • hedef mesafesi (sigma) — fiziksel ulaşılabilirlik (şartname 21)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .firstpassage import (LABEL_AMBIGUOUS, LABEL_STOP, LABEL_TARGET,
                           first_passage_times, label_cell, running_extremes)
from .horizons import BASE_TF_MIN, HORIZON_MIN, all_horizons, horizon_bars
from .regime import classify, target_distance_sigma
from .convergence import shrink_curve
from .stats import effective_sample_size, proportion_with_ci
from .targets import (CostProfile, STOP_SIGMA_MULTS, TARGET_NET_PCT,
                      barrier_levels, gross_target_pct, rr_ratio,
                      sigma_bar_pct, stop_grid)

DIRECTIONS = ("LONG", "SHORT")
MFE_QUANTILES = (10, 25, 50, 75, 90)


@dataclass
class SymbolMeasurement:
    symbol: str
    n_bars: int
    first_ts: str
    last_ts: str
    profile: Dict
    cells: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


QUANTILE_SAMPLE_CAP = 60_000


def _subsample(x: np.ndarray, cap: int = QUANTILE_SAMPLE_CAP) -> np.ndarray:
    """Kantil/otokorelasyon için DETERMİNİSTİK seyreltme.

    Rastgele örnekleme kullanılmaz: `Math.random`/tohumsuz RNG sonucu
    tekrarlanamaz yapar (şartname 84). Sabit adımlı seyreltme hem
    tekrarlanabilir hem de zaman sırasını korur."""
    if len(x) <= cap:
        return x
    adim = int(np.ceil(len(x) / cap))
    return x[::adim]


def _quantiles(x: np.ndarray) -> Dict[str, Optional[float]]:
    x = x[np.isfinite(x)]
    if not len(x):
        return {f"p{q}": None for q in MFE_QUANTILES}
    v = np.percentile(_subsample(x), MFE_QUANTILES)
    return {f"p{q}": float(round(val, 4)) for q, val in zip(MFE_QUANTILES, v)}


def measure_symbol(symbol: str,
                   df: pd.DataFrame,
                   profile: CostProfile,
                   horizons: Optional[Sequence[str]] = None,
                   stop_mults: Sequence[float] = STOP_SIGMA_MULTS,
                   periods: Optional[Dict[str, np.ndarray]] = None,
                   net_pct: float = TARGET_NET_PCT,
                   regimes: Optional[pd.DataFrame] = None,
                   tf_min: int = BASE_TF_MIN,
                   regime_names: Sequence[str] = ("ALL", "LOW_VOL",
                                                  "NORMAL_VOL", "HIGH_VOL",
                                                  "PANIC")) -> SymbolMeasurement:
    """Bir paritenin bütün (dönem × ufuk × yön × stop × rejim) hücrelerini ölçer.

    Dönemler (train/validation/test) TEK hesaptan maskeyle ayrılır — aynı
    ilk-geçiş taraması üç kez yapılmaz, üç kez farklı okunur.
    """
    horizons = list(horizons or all_horizons())
    periods = periods or {"full": np.ones(len(df), dtype=bool)}
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(c)

    reg = regimes if regimes is not None else classify(df)
    vol_reg = reg["vol_regime"].to_numpy()
    rv = reg["rv_24h_pct"].to_numpy(dtype=float)
    sig_bar = sigma_bar_pct(df)

    hb = {hz: horizon_bars(hz, tf_min) for hz in horizons}
    ext = running_extremes(h, l, c, list(hb.values()))
    temel0 = np.isfinite(sig_bar) & np.isfinite(c) & np.isfinite(rv)

    olcum = SymbolMeasurement(
        symbol=symbol, n_bars=n,
        first_ts=str(df.index[0]), last_ts=str(df.index[-1]),
        profile=profile.to_dict())

    for hz in horizons:
        H = hb[hz]
        stops = stop_grid(sig_bar, H, tuple(stop_mults))
        t_long = gross_target_pct(profile, HORIZON_MIN[hz] / 60.0, "LONG", net_pct)
        t_short = gross_target_pct(profile, HORIZON_MIN[hz] / 60.0, "SHORT", net_pct)

        up_list = [c * (1.0 + t_long / 100.0)]                       # LONG hedef
        up_key = {"T": 0}
        for m in stop_mults:                                          # SHORT stop
            up_key[f"S{m}"] = len(up_list)
            up_list.append(barrier_levels(c, stops[m], "SHORT", "stop"))
        dn_list = [c * (1.0 - t_short / 100.0)]                       # SHORT hedef
        dn_key = {"T": 0}
        for m in stop_mults:                                          # LONG stop
            dn_key[f"S{m}"] = len(dn_list)
            dn_list.append(barrier_levels(c, stops[m], "LONG", "stop"))

        T_up = first_passage_times(h, np.vstack(up_list), H, "up")
        T_dn = first_passage_times(l, np.vstack(dn_list), H, "dn")

        mfe_l, mae_l = ext[H]
        for d in DIRECTIONS:
            tT = T_up[up_key["T"]] if d == "LONG" else T_dn[dn_key["T"]]
            tgt_pct = t_long if d == "LONG" else t_short
            mfe_all = mfe_l if d == "LONG" else -mae_l
            mae_all = mae_l if d == "LONG" else -mfe_l
            for m in stop_mults:
                tS = (T_dn[dn_key[f"S{m}"]] if d == "LONG"
                      else T_up[up_key[f"S{m}"]])
                ls = label_cell(tT, tS, H, n_total=n)
                gecerli0 = temel0 & ls.valid
                for pname, pmask in periods.items():
                    gecerli = gecerli0 & pmask
                    for rname in regime_names:
                        msk = (gecerli if rname == "ALL"
                               else (gecerli & (vol_reg == rname)))
                        hucre = _cell_stats(
                            symbol, d, hz, H, m, rname, pname,
                            msk, ls, o, c, stops[m], rv, tgt_pct,
                            mfe_all, mae_all, tf_min)
                        if hucre is not None:
                            olcum.cells.append(hucre)
    return olcum


def _cell_stats(symbol, direction, hz, H, stop_mult, regime, period,
                msk, ls, o, c, stop_pct_arr, rv, tgt_pct,
                mfe_all, mae_all, tf_min) -> Optional[Dict]:
    n_ham = int(msk.sum())
    if n_ham < 30:
        return None
    lab = ls.label[msk]
    belirsiz = int((lab == LABEL_AMBIGUOUS).sum())
    kesin_m = lab != LABEL_AMBIGUOUS
    nk = int(kesin_m.sum())
    if nk < 30:
        return None
    labk = lab[kesin_m]
    tp = int((labk == LABEL_TARGET).sum())
    sl = int((labk == LABEL_STOP).sum())
    to = int(nk - tp - sl)

    ess = effective_sample_size(labk.astype(float), H)
    ci = proportion_with_ci(tp, nk, ess["used"])
    ci_sl = proportion_with_ci(sl, nk, ess["used"])

    idx = np.flatnonzero(msk)
    idx_k = idx[kesin_m]
    t_hit = ls.t_hit[msk][kesin_m]

    bars_tp = t_hit[labk == LABEL_TARGET]
    bars_sl = t_hit[labk == LABEL_STOP]

    # ZAMAN AŞIMI GETİRİSİ — ölçülür, sıfır varsayılmaz (şartname 28)
    to_idx = idx_k[labk == 0]
    if len(to_idx):
        son = np.minimum(to_idx + H, len(c) - 1)
        ham = (c[son] / c[to_idx] - 1.0) * 100.0
        if direction == "SHORT":
            ham = -ham
        timeout_ret = float(np.mean(ham))
        timeout_ret_p10 = float(np.percentile(ham, 10))
    else:
        timeout_ret = timeout_ret_p10 = None

    # BOŞLUKLU AÇILIŞTA STOP KAYMASI — kuyruk cezasının ölçülebilen kısmı.
    # Bar içi kayma 5m OHLC'den bilinemez; yalnız açılışın stop'u aşmış
    # olduğu vakalar sayılır. Bu ALT SINIRDIR, tam kayma değildir.
    sl_idx = idx_k[labk == LABEL_STOP]
    tail_excess = 0.0
    gap_n = 0
    if len(sl_idx):
        hit = sl_idx + t_hit[labk == LABEL_STOP]
        hit = np.minimum(hit, len(c) - 1)
        giris = c[sl_idx]
        sp = stop_pct_arr[sl_idx]
        if direction == "LONG":
            stop_lvl = giris * (1.0 - sp / 100.0)
            fazla = np.maximum(0.0, (stop_lvl - o[hit]) / giris * 100.0)
        else:
            stop_lvl = giris * (1.0 + sp / 100.0)
            fazla = np.maximum(0.0, (o[hit] - stop_lvl) / giris * 100.0)
        gap_n = int((fazla > 0).sum())
        tail_excess = float(np.mean(fazla))

    # YAKINSAMA — güven aralığı örneklemle gerçekten daralıyor mu?
    # Etiket dizisi yalnız burada elimizde; daralma eğrisi burada hesaplanır.
    egri, daralma = shrink_curve((labk == LABEL_TARGET).astype(int), 1, H)

    sig = target_distance_sigma(tgt_pct, rv[msk], H)
    sig = sig[np.isfinite(sig)]

    stop_med = float(np.nanmedian(stop_pct_arr[msk]))
    dk = HORIZON_MIN[hz]
    saat = dk / 60.0
    return {
        "symbol": symbol, "direction": direction, "horizon": hz,
        "horizon_minutes": dk, "horizon_bars": H,
        "stop_sigma_mult": float(stop_mult), "regime": regime, "period": period,
        "target_gross_pct": round(float(tgt_pct), 4),
        "target_net_pct": TARGET_NET_PCT,
        "stop_pct_median": round(stop_med, 4),
        "rr_median": round(rr_ratio(tgt_pct, stop_med), 3),
        "n_raw": nk, "n_ambiguous": belirsiz,
        "ambiguous_pct": round(belirsiz / max(1, nk + belirsiz) * 100.0, 3),
        "n_eff_used": round(ess["used"], 1),
        "n_eff_non_overlap": round(ess["non_overlap"], 1),
        "n_eff_autocorr": round(ess["autocorr"], 1),
        "tp_events": tp, "sl_events": sl, "timeout_events": to,
        "p_target_first": round(ci["p"], 5),
        "p_target_lower95": round(ci["lower95"], 5),
        "p_target_upper95": round(ci["upper95"], 5),
        "p_stop_first": round(ci_sl["p"], 5),
        "p_timeout": round(to / nk, 5),
        "median_bars_to_tp": (float(np.median(bars_tp)) if len(bars_tp) else None),
        "median_hours_to_tp": (float(np.median(bars_tp)) * tf_min / 60.0
                               if len(bars_tp) else None),
        "hours_to_tp_p25": (float(np.percentile(bars_tp, 25)) * tf_min / 60.0
                            if len(bars_tp) else None),
        "hours_to_tp_p75": (float(np.percentile(bars_tp, 75)) * tf_min / 60.0
                            if len(bars_tp) else None),
        "median_hours_to_sl": (float(np.median(bars_sl)) * tf_min / 60.0
                               if len(bars_sl) else None),
        "mfe": _quantiles(mfe_all[msk]),
        "mae": _quantiles(mae_all[msk]),
        "timeout_return_pct": (round(timeout_ret, 5) if timeout_ret is not None else None),
        "timeout_return_p10": (round(timeout_ret_p10, 5)
                               if timeout_ret_p10 is not None else None),
        "stop_gap_excess_pct": round(tail_excess, 5),
        "stop_gap_events": gap_n,
        "target_distance_sigma_median": (round(float(np.median(sig)), 3)
                                         if len(sig) else None),
        "shrink_ratio": daralma,
        "shrink_curve": egri,
        "horizon_hours": saat,
    }
