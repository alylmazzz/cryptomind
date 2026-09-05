"""Araştırma koşucusu — taban oranlarından nitelendirilmiş matrise.

AKIŞ
  1. 5m vadeli barlar + kaydediciden gerçek maliyet profili
  2. Bütün (dönem × ufuk × yön × stop × rejim) hücrelerinin taban oranları
  3. Stop seçimi — YALNIZ TRAIN döneminde, RobustEV ile (her seçim bir deneme)
  4. Özellikler + softmax model, purged walk-forward (train+validation)
  5. KİLİTLİ TEST bir kez açılır (2026-01-01→)
  6. Kalibrasyon, lift, DSR/PBO → durum makinesi → matrix.json

BU KOŞU HİÇBİR ŞEY VAAT ETMEZ. Çıktı "kenar bulundu" da olabilir "hiçbir
ufukta doğrulanmış fırsat yok" da. İkincisi bir başarısızlık değil, ölçümün
kendisidir (şartname 59, 116).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..research import validation as V
from . import features as FT
from .baserate import measure_symbol
from .firstpassage import (LABEL_AMBIGUOUS, LABEL_STOP, LABEL_TARGET,
                           first_passage_times, label_cell)
from .horizons import BASE_TF_MIN, HORIZON_MIN, horizon_bars, primary_horizons
from .lift import Payoff, evaluate_lift
from .model import (CLASS_INDEX, PlattHead, SoftmaxModel, decile_table,
                    fit_platt, fit_softmax, purged_walk_forward, top_decile_rate)
from .regime import classify, regime_note
from .robust import expected_holding_hours, robust_expected_value
from .state import CellEvidence, EvidenceGates, decide_state
from .stats import brier_score, calibration_slope_intercept, ece, log_loss, \
    reliability_curve
from .targets import (STOP_SIGMA_MULTS, TARGET_NET_PCT, CostProfile,
                      barrier_levels, gross_target_pct, profile_from_recorder,
                      sigma_bar_pct, stop_grid)

TRAIN_END = "2025-01-01"
VALID_END = "2026-01-01"
MODEL_STRIDE = 12                # eğitim örneklemi: saatte bir bar
SEED = 20260818                  # sabit ve açık (şartname 84)


def _periods(idx: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    t = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    tr = np.asarray(t < pd.Timestamp(TRAIN_END, tz="UTC"))
    va = np.asarray((t >= pd.Timestamp(TRAIN_END, tz="UTC")) &
                    (t < pd.Timestamp(VALID_END, tz="UTC")))
    te = np.asarray(t >= pd.Timestamp(VALID_END, tz="UTC"))
    return {"train": tr, "validation": va, "test": te,
            "full": np.ones(len(t), dtype=bool)}


def load_symbol(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d["dt"] = pd.to_datetime(d["ts"], unit="ms", utc=True)
    return d.set_index("dt").sort_index()


def payoff_from_cell(cell: Dict, profile: CostProfile) -> Tuple[Payoff, float]:
    """Hücrenin NET getiri profili. Maliyet HER DALDA ödenir."""
    maliyet = profile.total_bps(cell["horizon_hours"], cell["direction"]) / 100.0
    net_win = TARGET_NET_PCT                       # hedef zaten net %1'e kuruldu
    net_loss = cell["stop_pct_median"] + maliyet
    to = cell.get("timeout_return_pct")
    net_to = (to if to is not None else 0.0) - maliyet
    return Payoff(net_win, net_loss, net_to), maliyet


def cell_robust(cell: Dict, profile: CostProfile,
                psi: Optional[float] = None) -> Dict:
    pay, maliyet = payoff_from_cell(cell, profile)
    saat = expected_holding_hours(
        cell["p_target_first"], cell["p_stop_first"], cell["p_timeout"],
        cell.get("median_hours_to_tp"), cell.get("median_hours_to_sl"),
        cell["horizon_hours"])
    r = robust_expected_value(
        cell["p_target_first"], cell["p_stop_first"], cell["p_timeout"], pay,
        p_tp_lower=cell["p_target_lower95"],
        tail_excess_pct=cell.get("stop_gap_excess_pct", 0.0),
        psi=psi, expected_holding_hours=saat)
    d = r.to_dict()
    d["cost_pct"] = round(maliyet, 5)
    d["payoff"] = {"net_win": pay.net_win,
                   "net_loss": round(pay.net_loss, 5),
                   "net_timeout": round(pay.net_timeout, 5)}
    return d


# ── 1. taban oranları ──────────────────────────────────────────────────────

def run_baserates(data_dir: Path, out_dir: Path, feats: Optional[pd.DataFrame],
                  symbols: Sequence[str], horizons: Sequence[str]) -> pd.DataFrame:
    tum: List[Dict] = []
    profiller: Dict[str, Dict] = {}
    for sym in symbols:
        p = data_dir / f"{sym}_5m.parquet"
        if not p.exists():
            print(f"  {sym}: veri yok, atlandı", flush=True)
            continue
        d = load_symbol(p)
        prof = profile_from_recorder(sym, feats)
        profiller[sym] = prof.to_dict()
        t0 = time.time()
        m = measure_symbol(sym, d, prof, horizons=list(horizons),
                           periods=_periods(d.index))
        tum.extend(m.cells)
        print(f"  {sym}: {len(m.cells)} hücre · {time.time()-t0:.0f} sn "
              f"· maliyet modeli {prof.model}", flush=True)
    df = pd.DataFrame(tum)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "cells.parquet", index=False)
    (out_dir / "cost_profiles.json").write_text(
        json.dumps(profiller, ensure_ascii=False, indent=2), encoding="utf-8")
    return df


HOURS_PER_YEAR = 8760.0


def trades_per_year(holding_hours: float) -> float:
    """Bir hücrenin yılda kaç işlem ürettiği — Sharpe yıllıklaştırması için."""
    return HOURS_PER_YEAR / max(1e-6, float(holding_hours))


def _per_trade_sharpe(cell: Dict, profile: CostProfile) -> float:
    """Hücrenin İŞLEM-BAŞI Sharpe'ı — DSR bu ölçekte hesaplanır.

    μ = Σ p·getiri ,  σ = √(Σ p·getiri² − μ²)

    ⚠️ NEDEN YILLIK DEĞİL — ÖLÇÜLEREK BULUNDU
    Yıllıklaştırma √(işlem/yıl) ile çarpar: 5 dakikalık ufukta 17.520 işlem/yıl
    varsayımı Sharpe'ı 132 kat büyütür. Deneme dağılımı 5m'de **std = 100**,
    24h'de **std = 1,17** çıkıyordu; `expected_max_sharpe(N, 100)` astronomik
    bir eşik üretiyor ve DSR **540 hücrenin hepsinde tam 0** oluyordu — kapı
    çalışıyor görünüp hiçbir şey ölçmüyordu.

    Ayrıca 17.520 BAĞIMSIZ işlem varsayımı gerçekçi değil: pozisyonlar üst üste
    binmez, sermaye sınırlıdır, işlemler koreledir.

    İşlem-başı ölçekte kalmak hem tutarlı hem dürüst. Yıllık karşılık bağlam
    olarak AYRICA kaydedilir (`sharpe_annual`), ama DSR onu kullanmaz."""
    pay, _ = payoff_from_cell(cell, profile)
    p = np.array([cell["p_target_first"], cell["p_stop_first"], cell["p_timeout"]])
    x = np.array([pay.net_win, -abs(pay.net_loss), pay.net_timeout])
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    mu = float((p * x).sum())
    var = float((p * x * x).sum() - mu * mu)
    if var <= 1e-12:
        return 0.0
    return round(mu / math.sqrt(var), 6)


def _annual_sharpe(cell: Dict, profile: CostProfile) -> float:
    """Yıllık karşılık — YALNIZ bağlam için, DSR bunu KULLANMAZ."""
    ps = _per_trade_sharpe(cell, profile)
    saat = expected_holding_hours(
        cell["p_target_first"], cell["p_stop_first"], cell["p_timeout"],
        cell.get("median_hours_to_tp"), cell.get("median_hours_to_sl"),
        cell["horizon_hours"])
    return round(ps * math.sqrt(trades_per_year(saat)), 6)


# ── 2. stop seçimi (yalnız TRAIN) ──────────────────────────────────────────

def select_stops(cells: pd.DataFrame, profiles: Dict[str, CostProfile],
                 log_trials: bool = True) -> Dict[Tuple[str, str, str], float]:
    """Her (parite, ufuk, yön) için stop*, TRAIN RobustEV ile.

    ⚠️ Bu bir ARAMA'dır ve her arama bir denemedir (şartname 40). Beş stop
    adayı × parite × ufuk × yön kadar deneme `runs/trials.jsonl`'a yazılır;
    DSR bu sayıyı kullanır. Kayıtsız DSR anlamsızdır — bu ders bu projede
    zaten bir kez ödendi."""
    secim: Dict[Tuple[str, str, str], float] = {}
    tr = cells[(cells.period == "train") & (cells.regime == "ALL")]
    for (sym, hz, d), g in tr.groupby(["symbol", "horizon", "direction"]):
        en_iyi, en_iyi_ev = None, -1e18
        for _, row in g.iterrows():
            c = row.to_dict()
            r = cell_robust(c, profiles[sym])
            ev = r["robust_ev"] if r["robust_ev"] is not None else -1e18
            if log_trials:
                # `sharpe` alanı ZORUNLU: DSR'ın sr_std'si `trial_sharpes()`
                # ile okunur ve yalnız bu alanı taşıyan kayıtları görür.
                # Yazmazsak deneme SAYISI bu aramadan, dağılım BAŞKA bir
                # aramadan gelir — DSR içsel olarak tutarsız olur.
                V.trial_log(f"qual_stop_{hz}_{d}",
                            {"symbol": sym, "horizon": hz, "direction": d,
                             "stop_sigma_mult": c["stop_sigma_mult"]},
                            {"robust_ev": round(float(ev), 6),
                             # `sharpe` DSR'ın okuduğu alandır → İŞLEM-BAŞI.
                             "sharpe": _per_trade_sharpe(c, profiles[sym]),
                             "sharpe_annual": _annual_sharpe(c, profiles[sym]),
                             "p_target_first": c["p_target_first"],
                             "rr": c["rr_median"]})
            if ev > en_iyi_ev:
                en_iyi_ev, en_iyi = ev, c["stop_sigma_mult"]
        secim[(sym, hz, d)] = float(en_iyi if en_iyi is not None else 1.0)
    return secim


# ── 3. model veri kümesi ───────────────────────────────────────────────────

_FEATURE_CACHE: Dict[str, Dict] = {}
_CLOSE_CACHE: Dict[str, "pd.Series"] = {}


def _all_closes(data_dir: Path, symbols: Sequence[str]) -> Dict[str, "pd.Series"]:
    """Piyasa faktörü için bütün kapanışlar — bir kez okunur.

    Önbelleksiz hâlde her parite için diğer 7 parite yeniden okunuyordu:
    8 yerine 64 parquet okuması."""
    for s in symbols:
        if s in _CLOSE_CACHE:
            continue
        p = data_dir / f"{s}_5m.parquet"
        if p.exists():
            _CLOSE_CACHE[s] = load_symbol(p)["close"]
    return {s: v for s, v in _CLOSE_CACHE.items() if s in symbols}


def _symbol_base(data_dir: Path, symbols: Sequence[str], sym: str,
                 stride: int) -> Optional[Dict]:
    """Ufuktan BAĞIMSIZ taban özellikleri — parite başına BİR KEZ.

    NEDEN ÖNBELLEK: özellik matrisi 20 (ufuk, yön) kombinasyonunun her biri
    için 8 paritede yeniden hesaplanıyordu; ölçüldü, koşunun süresini saatlere
    çıkaran tek kalem buydu. Özelliklerin yalnız iki sütunu (geometri) ufka
    bağlıdır; gerisi ortaktır.

    ÖRNEKLEM IZGARASI SABİT: `np.arange(0, n, stride)`. Önceki hâl "geçerli
    barların her stride'inci"siydi; ufka göre farklı zaman noktaları seçiyordu
    ve önbelleğe alınamazdı. Sabit ızgara hem tekrarlanabilir hem de ufuklar
    arasında AYNI zaman noktalarını kullanır — ufuk karşılaştırması böylece
    örneklem farkından değil, gerçek farktan gelir.
    """
    anahtar = f"{sym}|{stride}"
    if anahtar in _FEATURE_CACHE:
        return _FEATURE_CACHE[anahtar]
    p = data_dir / f"{sym}_5m.parquet"
    if not p.exists():
        return None
    d = load_symbol(p)
    kapali = _all_closes(data_dir, symbols)
    sb = sigma_bar_pct(d)
    mkt = FT.market_factor(kapali, sym, index=d.index)
    X, ad, ail, tf = FT.build(d, sb, market_ret_4h=mkt)
    n = len(d)
    grid = np.arange(0, n, stride)
    out = {
        "index": d.index, "grid": grid,
        "Xg": X[grid], "names": ad, "families": ail, "tfs": tf,
        "sigma_bar": sb, "sigma_bar_g": sb[grid],
        "close": d["close"].to_numpy(dtype=float),
        "high": d["high"].to_numpy(dtype=float),
        "low": d["low"].to_numpy(dtype=float),
        "ts_g": d.index.to_numpy()[grid],
        "n": n,
    }
    _FEATURE_CACHE[anahtar] = out
    return out


def build_model_dataset(data_dir: Path, symbols: Sequence[str],
                        horizon: str, direction: str,
                        stops: Dict[Tuple[str, str, str], float],
                        profiles: Dict[str, CostProfile],
                        stride: int = MODEL_STRIDE) -> Optional[Dict]:
    """Bir (ufuk, yön) için bütün paritelerin özellik+etiket kümesi."""
    H = horizon_bars(horizon, BASE_TF_MIN)

    Xs, ys, ts, syms, nets, sps = [], [], [], [], [], []
    isimler = ailelerr = tflerr = None
    for sym in symbols:
        taban = _symbol_base(data_dir, symbols, sym, stride)
        if taban is None:
            continue
        sb = taban["sigma_bar"]
        tgt = gross_target_pct(profiles[sym], HORIZON_MIN[horizon] / 60.0,
                               direction, TARGET_NET_PCT)
        X, ad, ail, tf = FT.add_geometry(
            taban["Xg"], taban["names"], taban["families"], taban["tfs"],
            taban["sigma_bar_g"], tgt, H)

        c, h, l = taban["close"], taban["high"], taban["low"]
        grid = taban["grid"]
        m = stops.get((sym, horizon, direction), 1.0)
        sp = stop_grid(sb, H, (m,))[m]
        if direction == "LONG":
            up = (c * (1.0 + tgt / 100.0))[None, :]
            dn = barrier_levels(c, sp, "LONG", "stop")[None, :]
            tT = first_passage_times(h, up, H, "up")[0]
            tS = first_passage_times(l, dn, H, "dn")[0]
        else:
            dn = (c * (1.0 - tgt / 100.0))[None, :]
            up = barrier_levels(c, sp, "SHORT", "stop")[None, :]
            tT = first_passage_times(l, dn, H, "dn")[0]
            tS = first_passage_times(h, up, H, "up")[0]
        ls = label_cell(tT, tS, H, n_total=len(c))

        # SABİT IZGARA üzerinde geçerlilik: örnekler her ufukta AYNI zaman
        # noktalarından seçilir; yalnız geçerlilik maskesi ufka göre değişir.
        ok_g = (ls.valid[grid] & (ls.label[grid] != LABEL_AMBIGUOUS)
                & np.isfinite(sb[grid]) & np.isfinite(X).all(axis=1))
        idx = grid[ok_g]
        X = X[ok_g]
        if len(idx) < 500:
            continue
        lab = ls.label[idx]
        y = np.where(lab == LABEL_TARGET, CLASS_INDEX["TP"],
                     np.where(lab == LABEL_STOP, CLASS_INDEX["SL"],
                              CLASS_INDEX["TIMEOUT"]))

        # GERÇEKLEŞEN NET GETİRİ — örnek örnek, varsayımsız.
        # Olasılıklardan EV geri-kurmak yerine bunu ölçmek daha dürüsttür:
        # yüksek oynaklıkta hem hedef hem stop daha olası ve stop daha
        # GENİŞ olur; sadece P(hedef)'e bakan bir hesap bunu kaçırır.
        maliyet = profiles[sym].total_bps(HORIZON_MIN[horizon] / 60.0,
                                          direction) / 100.0
        son = np.minimum(idx + H, len(c) - 1)
        to_ret = (c[son] / c[idx] - 1.0) * 100.0
        if direction == "SHORT":
            to_ret = -to_ret
        net = np.where(y == CLASS_INDEX["TP"], TARGET_NET_PCT,
                       np.where(y == CLASS_INDEX["SL"],
                                -(sp[idx] + maliyet), to_ret - maliyet))

        Xs.append(X); ys.append(y)
        nets.append(net.astype(float)); sps.append(sp[idx].astype(float))
        ts.append(taban["index"].to_numpy()[idx])
        syms.append(np.full(len(idx), sym))
        isimler, ailelerr, tflerr = ad, ail, tf

    if not Xs:
        return None
    X = np.vstack(Xs); y = np.concatenate(ys)
    t = np.concatenate(ts); s = np.concatenate(syms)
    net = np.concatenate(nets); sp_all = np.concatenate(sps)
    sira = np.argsort(t, kind="stable")
    return {"X": X[sira], "y": y[sira], "t": t[sira], "symbol": s[sira],
            "net_ret": net[sira], "stop_pct": sp_all[sira],
            "names": isimler, "families": ailelerr, "tfs": tflerr,
            "horizon_bars": H}


# ── 4. eğitim + değerlendirme ──────────────────────────────────────────────

def _wf_oos(X: np.ndarray, y: np.ndarray, kat: List, names: List[str],
            l2: float) -> Tuple[np.ndarray, List[float]]:
    """Purged walk-forward ile OOS olasılık üret."""
    p_oos = np.full(len(X), np.nan)
    perf: List[float] = []
    for f in kat:
        mdl = fit_softmax(X[f.train_start:f.train_end],
                          y[f.train_start:f.train_end], l2=l2, names=names)
        P = mdl.predict_proba(X[f.test_start:f.test_end])
        if P is None:
            continue
        p_oos[f.test_start:f.test_end] = P[:, CLASS_INDEX["TP"]]
        yy = (y[f.test_start:f.test_end] == CLASS_INDEX["TP"]).astype(float)
        perf.append(-brier_score(P[:, CLASS_INDEX["TP"]], yy))
    return p_oos, perf


def ablation(ds: Dict, kat: List, dev_mask: np.ndarray, l2: float = 1.0) -> Dict:
    """Şartname 34/35 + yeni şartname 4 — İKİ EKSENDE ablasyon.

    A) FEATURE TIMEFRAME: {5m, +1h, +4h, +1d, hepsi} kümülatif kümeler.
       "Hangi çözünürlük gerçekten katkı veriyor?" sorusu ÖLÇÜLÜR.
    B) ÖZELLİK AİLESİ: her aile TEK TEK çıkarılır, ΔBrier ölçülür.

    SHAP/önem sıralaması burada KULLANILMAZ: bir özelliğin yüksek önem skoru
    olması OOS katkı verdiği anlamına gelmez (şartname 35). Ölçüt, aileyi
    çıkarınca OOS Brier'in BOZULUP bozulmadığıdır.
    """
    X, y = ds["X"][dev_mask], ds["y"][dev_mask]
    tfs, ail, ad = ds["tfs"], ds["families"], ds["names"]
    out: Dict = {"timeframe_sets": [], "families": []}

    for tf_set in FT.TF_SETS:
        msk = FT.tf_mask(tfs, tf_set)
        if msk.sum() < 3:
            continue
        p, _ = _wf_oos(X[:, msk], y, kat, [a for a, k in zip(ad, msk) if k], l2)
        f = np.isfinite(p)
        if f.sum() < 500:
            continue
        yy = (y == CLASS_INDEX["TP"]).astype(float)
        out["timeframe_sets"].append({
            "tf_set": tf_set, "n_features": int(msk.sum()),
            "oos_brier": brier_score(p[f], yy[f]),
            "oos_ece": ece(p[f], yy[f]),
            "top_decile": top_decile_rate(decile_table(p[f], yy[f]))[0]})

    tam = None
    for r in out["timeframe_sets"]:
        if r["tf_set"] == "all":
            tam = r["oos_brier"]
    for aile in sorted(set(ail)):
        msk = np.array([a != aile for a in ail], dtype=bool)
        if msk.sum() < 3:
            continue
        p, _ = _wf_oos(X[:, msk], y, kat, [a for a, k in zip(ad, msk) if k], l2)
        f = np.isfinite(p)
        if f.sum() < 500:
            continue
        yy = (y == CLASS_INDEX["TP"]).astype(float)
        b = brier_score(p[f], yy[f])
        out["families"].append({
            "removed": aile, "n_features_left": int(msk.sum()),
            "oos_brier_without": b,
            "delta_brier": (None if tam is None else float(b - tam)),
            "contributes": (None if tam is None else bool(b > tam + 1e-6))})
    return out


def train_and_validate(ds: Dict, horizon: str, direction: str,
                       l2: float = 1.0, run_ablation: bool = False) -> Dict:
    """Purged walk-forward (train+validation) → OOS olasılıklar → kalibrasyon."""
    t = pd.DatetimeIndex(ds["t"])
    dev = np.asarray(t < pd.Timestamp(VALID_END, tz="UTC"))
    X, y = ds["X"][dev], ds["y"][dev]
    sym = ds["symbol"][dev]
    stride_bars = max(1, ds["horizon_bars"] // MODEL_STRIDE)
    kat = purged_walk_forward(len(X), stride_bars, n_folds=5)
    if not kat:
        return {"ok": False, "reason": "purged_walk_forward yeterli veri bulamadı"}

    p_oos, fold_perf = _wf_oos(X, y, kat, ds["names"], l2)
    abl = ablation(ds, kat, dev, l2) if run_ablation else None

    # ŞARTNAME 35 — OOS katkısı ölçülemeyen aile karar motorunda KALMAZ.
    # Önem skoru (SHAP vb.) yeterli değildir; ölçüt aileyi çıkarınca OOS
    # Brier'in bozulup bozulmadığıdır. Budanmış model daha iyiyse o kullanılır.
    budanan: List[str] = []
    kolon_maskesi = np.ones(X.shape[1], dtype=bool)
    if abl and abl.get("families"):
        budanan = [r["removed"] for r in abl["families"]
                   if r.get("contributes") is False]
        if budanan:
            kolon_maskesi = np.array([a not in budanan for a in ds["families"]],
                                     dtype=bool)
            if kolon_maskesi.sum() >= 3:
                adlar = [a for a, k in zip(ds["names"], kolon_maskesi) if k]
                p2, perf2 = _wf_oos(X[:, kolon_maskesi], y, kat, adlar, l2)
                f2 = np.isfinite(p2)
                yy2 = (y == CLASS_INDEX["TP"]).astype(float)
                if f2.sum() >= 500 and brier_score(p2[f2], yy2[f2]) <= \
                        brier_score(p_oos[np.isfinite(p_oos)],
                                    yy2[np.isfinite(p_oos)]):
                    p_oos, fold_perf = p2, perf2
                else:
                    kolon_maskesi = np.ones(X.shape[1], dtype=bool)
                    budanan = []
            else:
                kolon_maskesi = np.ones(X.shape[1], dtype=bool)
                budanan = []
    X = X[:, kolon_maskesi]
    ds = dict(ds)
    ds["names"] = [a for a, k in zip(ds["names"], kolon_maskesi) if k]
    ds["families"] = [a for a, k in zip(ds["families"], kolon_maskesi) if k]
    ds["tfs"] = [a for a, k in zip(ds["tfs"], kolon_maskesi) if k]
    ds["column_mask"] = kolon_maskesi

    m = np.isfinite(p_oos)
    if m.sum() < 500:
        return {"ok": False, "reason": "OOS örneklem yetersiz"}
    y_tp = (y == CLASS_INDEX["TP"]).astype(float)
    dec = decile_table(p_oos[m], y_tp[m])
    ust, ust_n = top_decile_rate(dec)

    # SEÇİM EŞİĞİ SADECE DEV'DEN GELİR. Kilitli teste eşik aratmak, testi
    # validasyona çevirir (SPLIT.md kural 3).
    esik = float(np.quantile(p_oos[m], 0.90))
    net_dev = ds["net_ret"][dev]
    sec = m & (p_oos >= esik)
    dev_sel = _selection_stats(y[sec], net_dev[sec])

    kal = calibration_slope_intercept(p_oos[m], y_tp[m])
    basliklar: Dict[str, Dict] = {}
    for s in np.unique(sym):
        sm = m & (sym == s)
        basliklar[str(s)] = fit_platt(p_oos[sm], y_tp[sm]).to_dict()

    son = fit_softmax(X, y, l2=l2, names=ds["names"])
    return {
        "ok": True, "horizon": horizon, "direction": direction,
        "n_dev": int(len(X)), "n_oos": int(m.sum()),
        "folds": len(kat),
        "baseline_dev": float(y_tp.mean()),
        "oos_baseline": float(y_tp[m].mean()),
        "oos_brier": brier_score(p_oos[m], y_tp[m]),
        "oos_brier_base": brier_score(np.full(int(m.sum()), y_tp[m].mean()), y_tp[m]),
        "oos_logloss": log_loss(p_oos[m], y_tp[m]),
        "oos_ece": ece(p_oos[m], y_tp[m]),
        "calibration": kal,
        "reliability": reliability_curve(p_oos[m], y_tp[m]),
        "deciles": dec,
        "top_decile_rate": ust, "top_decile_n": ust_n,
        "selection_threshold": esik,
        "dev_selected": dev_sel,
        "p_oos": p_oos[m],
        "blind_net_mean": float(np.mean(net_dev[m])),
        "fold_perf": fold_perf,
        "ablation": abl,
        "pruned_families": budanan,
        "column_mask": kolon_maskesi,
        "pair_heads": basliklar,
        "model": son.to_dict(),
    }


def _selection_stats(y: np.ndarray, net: np.ndarray,
                     stop_pct: Optional[np.ndarray] = None,
                     min_n: int = 20) -> Dict:
    """Seçilen sinyallerin GERÇEKLEŞEN sonuç dağılımı ve net getirisi.

    Bu, olasılıklardan geri-kurulmuş bir EV değil; her örneğin fiilen ne
    getirdiğinin ortalamasıdır. Yüksek oynaklıkta stop'un genişlemesi,
    zaman aşımı getirisinin negatifliği, hepsi içinde.

    `stop_pct` de döndürülür çünkü BAŞABAŞ NOKTASI seçilen altkümenin kendi
    kayıp büyüklüğünden hesaplanmalıdır. Bütün hücrenin medyan stop'unu
    kullanmak, modelin tam da geniş-stop bölgesini seçtiği durumda gereken
    lift'i sistematik olarak AZ gösterir."""
    n = int(len(y))
    if n < min_n:
        return {"n": n, "ok": False}
    tp = float((y == CLASS_INDEX["TP"]).mean())
    sl = float((y == CLASS_INDEX["SL"]).mean())
    to = float((y == CLASS_INDEX["TIMEOUT"]).mean())
    ort = float(np.mean(net))
    sd = float(np.std(net, ddof=1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 and math.isfinite(sd) else float("nan")
    t = ort / se if se and math.isfinite(se) and se > 0 else float("nan")
    d = {"n": n, "ok": True, "tp_rate": tp, "sl_rate": sl, "timeout_rate": to,
         "net_mean_pct": ort, "net_std_pct": sd,
         "net_se_pct": (None if not math.isfinite(se) else se),
         "t_stat": (None if not math.isfinite(t) else float(t)),
         "net_lower95": (None if not math.isfinite(se) else ort - 1.96 * se)}
    if stop_pct is not None and len(stop_pct) == n:
        d["stop_pct_mean"] = float(np.mean(stop_pct))
    m_to = y == CLASS_INDEX["TIMEOUT"]
    d["timeout_net_mean"] = float(np.mean(net[m_to])) if m_to.any() else None
    return d


def locked_test(ds: Dict, mdl: SoftmaxModel, heads: Dict[str, Dict],
                threshold: Optional[float] = None,
                column_mask: Optional[np.ndarray] = None) -> Dict:
    """KİLİTLİ TEST — bir kez açılır, sonuç ne olursa olsun konfig değişmez.

    Seçim eşiği DEV'den gelir (parametre); test setinde eşik ARANMAZ."""
    t = pd.DatetimeIndex(ds["t"])
    te = np.asarray(t >= pd.Timestamp(VALID_END, tz="UTC"))
    if te.sum() < 300:
        return {"ok": False, "reason": "kilitli test örneklemi yetersiz"}
    Xt = ds["X"] if column_mask is None else ds["X"][:, column_mask]
    X, y, sym = Xt[te], ds["y"][te], ds["symbol"][te]
    net = ds["net_ret"][te]
    P = mdl.predict_proba(X)
    if P is None:
        return {"ok": False, "reason": "model çıkarımı başarısız"}
    p = P[:, CLASS_INDEX["TP"]].copy()
    for s, hd in heads.items():
        m = sym == s
        if m.any() and hd.get("fitted"):
            p[m] = PlattHead(**hd).apply(p[m])
    y_tp = (y == CLASS_INDEX["TP"]).astype(float)
    dec = decile_table(p, y_tp)
    ust, ust_n = top_decile_rate(dec)
    esik = threshold if threshold is not None else float(np.quantile(p, 0.90))
    sec = p >= esik
    sp = ds["stop_pct"][te]
    # ALT DÖNEM TUTARLILIĞI (şartname 42) — kazanç tek aya sıkışmışsa
    # "kenar" değil talih eseridir.
    aylar = pd.DatetimeIndex(t[te]).to_period("M")
    alt: List[Dict] = []
    for ay in aylar.unique():
        ms = sec & np.asarray(aylar == ay)
        if ms.sum() >= 15:
            alt.append({"period": str(ay), "n": int(ms.sum()),
                        "net_mean": float(np.mean(net[ms]))})
    poz = (sum(1 for a in alt if a["net_mean"] > 0) / len(alt)) if alt else None
    # Parite başına ayrı ölçüm — hiyerarşik: örneklem yetersizse havuzlanmış
    # sonuç kullanılır, uydurma bir parite-özel sayı üretilmez (şartname 36).
    per: Dict[str, Dict] = {}
    for s in np.unique(sym):
        ms = sec & (sym == s)
        per[str(s)] = _selection_stats(y[ms], net[ms], sp[ms], min_n=60)
    return {"ok": True, "n": int(te.sum()), "baseline": float(y_tp.mean()),
            "brier": brier_score(p, y_tp),
            "brier_base": brier_score(np.full(len(p), y_tp.mean()), y_tp),
            "ece": ece(p, y_tp),
            "calibration": calibration_slope_intercept(p, y_tp),
            "reliability": reliability_curve(p, y_tp),
            "deciles": dec, "top_decile_rate": ust, "top_decile_n": ust_n,
            "threshold": esik, "selected_frac": float(sec.mean()),
            "selected": _selection_stats(y[sec], net[sec], sp[sec]),
            "selected_by_symbol": per,
            "subperiods": alt, "positive_subperiod_frac": poz,
            "blind_net_mean": float(np.mean(net)),
            "p_dist_test": p,
            "p_test": p, "y_test": y, "symbol_test": sym, "net_test": net,
            "selected_mask": sec}


# ── 5. matris birleştirme ──────────────────────────────────────────────────

def _model_robust(cell: Dict, sel: Dict, blind: Dict,
                  psi: Optional[float]) -> Dict:
    """Model seçtiğinde GERÇEKLEŞEN net getiriden robust EV.

    Olasılıklardan EV geri-kurulmaz; kilitli testte seçilen sinyallerin
    fiilen ne kazandırdığı ölçülür. Belirsizlik cezası bu ortalamanın kendi
    %95 alt sınırıdır — ayrı bir varsayım gerekmez."""
    from .robust import UNMEASURED_DRIFT_HAIRCUT_PCT
    ev = sel.get("net_mean_pct")
    ev_alt = sel.get("net_lower95")
    if ev is None or ev_alt is None:
        return blind
    kuyruk = float(sel.get("sl_rate", 0.0) * cell.get("stop_gap_excess_pct", 0.0))
    if psi is None:
        surukleme, olculdu = UNMEASURED_DRIFT_HAIRCUT_PCT, False
    else:
        surukleme = float(min(1.0, max(0.0, psi / 0.25))
                          * UNMEASURED_DRIFT_HAIRCUT_PCT)
        olculdu = True
    rev = float(ev_alt) - kuyruk - surukleme
    saat = expected_holding_hours(
        sel.get("tp_rate", 0.0), sel.get("sl_rate", 0.0),
        sel.get("timeout_rate", 0.0),
        cell.get("median_hours_to_tp"), cell.get("median_hours_to_sl"),
        cell["horizon_hours"])
    return {"ev": float(ev), "ev_lower": float(ev_alt),
            "uncertainty_penalty": float(ev - ev_alt),
            "tail_penalty": kuyruk, "drift_penalty": surukleme,
            "robust_ev": rev, "expected_holding_hours": saat,
            "robust_utility": (rev / saat if saat > 0 else None),
            "drift_measured": olculdu,
            "cost_pct": blind.get("cost_pct"),
            "payoff": blind.get("payoff"),
            "notes": ["EV, olasılıklardan türetilmedi — kilitli testte seçilen "
                      "sinyallerin gerçekleşen net getirisidir"]}


def _convergence_for(cells: pd.DataFrame, sym: str, hz: str, d: str,
                     m: float, cell: Dict) -> Dict:
    """Bir hücrenin yakınsama değerlendirmesi.

    Dönem ve rejim kırılımları AYRI hücreler olarak zaten ölçülmüştür
    (period ∈ {train, validation, test}, regime ∈ {LOW_VOL, …}); burada
    yalnız toplanır. Yeniden tarama YAPILMAZ.

    Bu, "sayı kesinleşiyor mu?" sorusunun ölçülebilir cevabıdır: örneklem
    azlığı ile GERÇEK kayma birbirinden ayrılır. İkincisi veri biriktirerek
    düzelmez."""
    # Karar ve gerekçe TEK yerden gelir (convergence.decide_verdict);
    # mantığı burada tekrarlamak bir kez 472 hücreyi yanlış etiketledi.
    from .convergence import build_checks, decide_verdict, explain, group_spread

    alt = cells[(cells.symbol == sym) & (cells.horizon == hz)
                & (cells.direction == d) & (cells.stop_sigma_mult == m)]

    donem = {}
    for pname in ("train", "validation", "test"):
        rr = alt[(alt.period == pname) & (alt.regime == "ALL")]
        if len(rr):
            x = rr.iloc[0]
            donem[pname] = (int(x["tp_events"]), int(x["n_raw"]))
    rejim = {}
    for rname in ("LOW_VOL", "NORMAL_VOL", "HIGH_VOL", "PANIC"):
        rr = alt[(alt.period == "full") & (alt.regime == rname)]
        if len(rr):
            x = rr.iloc[0]
            rejim[rname] = (int(x["tp_events"]), int(x["n_raw"]))

    p_spread, p_list = group_spread(donem)
    r_spread, r_list = group_spread(rejim)
    ci_w = float(cell["p_target_upper95"] - cell["p_target_lower95"])
    ne = float(cell["n_eff_used"])
    daralma = cell.get("shrink_ratio")

    kontrol = build_checks(ci_w, ne, p_spread, r_spread, daralma)
    neden = explain(kontrol, ci_w, ne, p_spread, r_spread, daralma)
    verdikt = decide_verdict(kontrol)

    return {"verdict": verdikt, "ci_width": round(ci_w, 5),
            "n_effective": round(ne, 1),
            "period_spread": (None if p_spread is None else round(p_spread, 5)),
            "period_estimates": p_list,
            "regime_spread": (None if r_spread is None else round(r_spread, 5)),
            "regime_estimates": r_list,
            "shrink_ratio": daralma, "shrink_curve": cell.get("shrink_curve"),
            "checks": kontrol, "reasons": neden}


def assemble(cells: pd.DataFrame, profiles: Dict[str, CostProfile],
             stops: Dict[Tuple[str, str, str], float],
             models: Dict[Tuple[str, str], Dict],
             gates: EvidenceGates) -> Tuple[List[Dict], Dict]:
    """Ölçüm + model → durum makinesi → kullanıcıya gösterilecek hücreler."""
    n_trials = V.trial_count()
    out: List[Dict] = []
    ozet: Dict = {"n_trials_registry": n_trials, "cells": 0, "by_status": {},
                  "by_convergence": {}}

    tam = cells[(cells.period == "full") & (cells.regime == "ALL")]
    for (sym, hz, d), g in tam.groupby(["symbol", "horizon", "direction"]):
        m = stops.get((sym, hz, d))
        row = g[g.stop_sigma_mult == m]
        if not len(row):
            continue
        c = row.iloc[0].to_dict()
        yakinsama = _convergence_for(cells, sym, hz, d, m, c)
        prof = profiles[sym]
        pay, maliyet = payoff_from_cell(c, prof)
        mdl = models.get((hz, d), {})
        dev = mdl.get("dev") or {}
        test = mdl.get("test") or {}

        sel: Dict = {}
        sel_kaynak = "yok"
        if test.get("ok"):
            ozel = (test.get("selected_by_symbol") or {}).get(sym) or {}
            if ozel.get("ok"):
                sel, sel_kaynak = ozel, "parite"
            elif (test.get("selected") or {}).get("ok"):
                sel, sel_kaynak = test["selected"], "havuzlanmis"
        model_rate = sel.get("tp_rate") if sel.get("ok") else None
        n_model = float(sel.get("n") or 0)

        # BAŞABAŞ, SEÇİLEN ALTKÜMENİN KENDİ KAYBINDAN hesaplanır. Model tam da
        # geniş-stop (yüksek oynaklık) bölgesini seçiyorsa, hücrenin medyan
        # stop'uyla hesaplanan başabaş gereken lift'i AZ gösterir.
        pay_sel = pay
        if sel.get("ok") and sel.get("stop_pct_mean") is not None:
            pay_sel = Payoff(TARGET_NET_PCT,
                             float(sel["stop_pct_mean"]) + maliyet,
                             float(sel.get("timeout_net_mean") or pay.net_timeout))
        lift = evaluate_lift(c["p_target_first"], pay_sel,
                             (sel.get("timeout_rate") if sel.get("ok")
                              else c["p_timeout"]),
                             model_rate=model_rate,
                             n_model_eff=n_model,
                             n_base_eff=c["n_eff_used"])

        psi_val = mdl.get("psi")
        kor = cell_robust(c, prof, psi=psi_val)          # KÖR taban EV'si
        r = _model_robust(c, sel, kor, psi_val) if sel.get("ok") else kor

        dsr = dsr_program = pbo_v = None
        if test.get("ok") and sel.get("ok"):
            rets = np.asarray(test["net_test"])[np.asarray(test["selected_mask"])]
            if len(rets) >= 30:
                # ⚠️ DSR'ın İKİ SORUSU VAR VE İKİSİ AYNI DEĞİL — ölçüldü:
                # n_trials olarak bütün programı (3.005) ve sr_std olarak o
                # ufkun dağılımını (5m'de 92,7) birlikte vermek
                # `expected_max_sharpe`'ı astronomik yapıyor ve DSR **540
                # hücrenin hepsinde tam 0** çıkıyordu. Kapı çalışıyor görünüp
                # hiçbir şey ölçmüyordu.
                #
                # Doğrusu iki ayrı sayı:
                #   dsr          — AYNI ölçekteki denemeler arasından seçim
                #                  (n = o ufuk+yönün deneme sayısı)
                #   dsr_program  — bütün program genişliği (muhafazakâr üst kapı)
                # İkisi de raporlanır; kapı `dsr`'ı kullanır, `dsr_program`
                # bağlam olarak görünür.
                ad = f"qual_stop_{hz}_{d}"
                grup_n = max(1, V.trial_count(ad))
                yayilim = _trial_dispersion(ad)
                # periods_per_year=1 → İŞLEM-BAŞI ölçek. `rets` zaten
                # işlem-başı getiri; deneme dağılımı da işlem-başı. Üç
                # büyüklüğün aynı ölçekte olması ZORUNLU.
                dsr = float(V.deflated_sharpe(
                    np.asarray(rets, dtype=float), grup_n,
                    sr_std=yayilim, periods_per_year=1.0)["dsr"])
                dsr_program = float(V.deflated_sharpe(
                    np.asarray(rets, dtype=float), max(1, n_trials),
                    sr_std=yayilim, periods_per_year=1.0)["dsr"])

        # ── PBO: GERÇEK aday matrisi ──────────────────────────────────────
        # Eskiden tek sütunlu bir diziyi ikiye kopyalıyordu (`np.repeat(pm,2)`)
        # ve PBO **540 hücrenin hepsinde 1,0** çıkıyordu — yani "her seçim
        # aşırı uyum" diyordu, hiçbir şey ayırt etmiyordu.
        # Gerçek matris zaten elde: stop adayları × dönemler.
        pbo_v = _pbo_from_stop_grid(cells, sym, hz, d, profiles[sym])

        from .stats import proportion_with_ci
        if sel.get("ok"):
            mci = proportion_with_ci(int(round(sel["tp_rate"] * sel["n"])),
                                     int(sel["n"]), float(sel["n"]))
            p_lo, p_hi = mci["lower95"], mci["upper95"]
            n_eff_karar = float(sel["n"])
            n_tp_karar = int(round(sel["tp_rate"] * sel["n"]))
        else:
            p_lo, p_hi = c["p_target_lower95"], c["p_target_upper95"]
            n_eff_karar = float(c["n_eff_used"])
            n_tp_karar = int(c["tp_events"])

        kanit = CellEvidence(
            n_effective=n_eff_karar,
            n_tp_events=n_tp_karar,
            baseline=c["p_target_first"],
            p_target_first=(model_rate if model_rate is not None
                            else c["p_target_first"]),
            p_lower95=p_lo, p_upper95=p_hi,
            robust_ev=r["robust_ev"],
            edge_proven=lift.edge, lift_reason=lift.reason,
            has_validation_report=bool(dev.get("ok")),
            has_model=bool(dev.get("ok")),
            calibration_ece=(test.get("ece") if test.get("ok") else dev.get("oos_ece")),
            calibration_slope=((test.get("calibration") or {}).get("slope")
                               if test.get("ok")
                               else (dev.get("calibration") or {}).get("slope")),
            psi=psi_val,
            data_quality=1.0, liquidity_score=1.0,
            cost_model_valid=(prof.model in ("MEASURED_L2_VWAP", "ESTIMATED")),
            cost_model_measured=(prof.model == "MEASURED_L2_VWAP"),
            data_stale=False, regime_supported=True,
            dsr=dsr, pbo=pbo_v,
            positive_subperiod_frac=test.get("positive_subperiod_frac"),
            regime_concentrated=bool(
                test.get("positive_subperiod_frac") is not None
                and test["positive_subperiod_frac"] < 0.5))
        karar = decide_state(kanit, gates)

        hucre = dict(c)
        hucre.update(r)
        hucre.update(karar.to_dict())
        if sel.get("ok"):
            # Panelde gösterilen olasılık MODELİN SEÇTİĞİ altkümeye aittir;
            # kör taban ayrı alanda korunur ki karşılaştırma kaybolmasın.
            hucre.update({"p_target_first": sel["tp_rate"],
                          "p_stop_first": sel["sl_rate"],
                          "p_timeout": sel["timeout_rate"],
                          "p_target_lower95": p_lo,
                          "p_target_upper95": p_hi,
                          "n_eff_used": float(sel["n"]),
                          "selected_frac": test.get("selected_frac"),
                          "net_t_stat": sel.get("t_stat")})
        hucre.update({
            "baseline": c["p_target_first"],
            "blind_p_target_first": c["p_target_first"],
            "blind_robust_ev": kor["robust_ev"],
            "blind_net_mean": test.get("blind_net_mean"),
            "required_lift": lift.required_lift,
            "actual_lift": lift.actual_lift,
            "actual_lift_lower95": lift.actual_lift_lower95,
            "breakeven_probability": lift.breakeven,
            "lift_reason": lift.reason,
            "model_rate_oos": model_rate,
            "model_rate_n": n_model,
            "brier": test.get("brier") if test.get("ok") else dev.get("oos_brier"),
            "brier_base": (test.get("brier_base") if test.get("ok")
                           else dev.get("oos_brier_base")),
            "ece": kanit.calibration_ece,
            "calibration_slope": kanit.calibration_slope,
            "dsr": dsr, "dsr_program": dsr_program, "pbo": pbo_v,
            "expected_net_return": r["ev"],
            "net_mean": sel.get("net_mean_pct"),
            "net_lower95": sel.get("net_lower95"),
            "net_t_stat": sel.get("t_stat"),
            "selected_stop_pct": sel.get("stop_pct_mean"),
            "selection_source": sel_kaynak,
            "subperiods": test.get("subperiods"),
            "positive_subperiod_frac": test.get("positive_subperiod_frac"),
            "cost_model": prof.model,
            "convergence": yakinsama,
        })
        out.append(hucre)
        ozet["by_status"][karar.state] = ozet["by_status"].get(karar.state, 0) + 1
        v = yakinsama["verdict"]
        ozet["by_convergence"][v] = ozet["by_convergence"].get(v, 0) + 1
    ozet["cells"] = len(out)
    return out, ozet


PBO_MIN_ROWS = 8          # CSCV'nin blok bölmesi için asgari dilim sayısı


def _pbo_from_stop_grid(cells: pd.DataFrame, sym: str, hz: str, d: str,
                        profile: CostProfile) -> Optional[float]:
    """Aşırı uyum olasılığı — GERÇEK aday matrisiyle.

    PBO'nun sorusu: "örneklem-İÇİNDE en iyi çıkan aday, örneklem-DIŞINDA da üst
    yarıda mı?" Bunun için (dilim × aday) performans matrisi gerekir ve CSCV
    dilimleri kombinatoryal olarak ikiye böler.

    ⚠️ İKİ KEZ YANLIŞ HESAPLANDI — ölçülerek bulundu:
      1. Tek sütunlu dizi ikiye kopyalanıyordu → PBO 594/594 hücrede tam 1,0.
      2. Düzeltildi ama yalnız 3 dilim (train/validation/test) verildi; CSCV
         3 dilimi anlamlı bölemez ve YİNE 1,0 döndü. Oysa gerçek matris
         gösteriyordu ki en iyi aday (k=0,50) HER dönemde en iyi — yani
         aşırı uyum YOK, PBO düşük olmalıydı.

    Şimdi dilimler **dönem × rejim** çaprazından geliyor (3 × 4 = 12'ye kadar).
    Yeterli dilim yoksa PBO **tanımsızdır ve `None` döner** — 1,0 döndürmek
    "her seçim aşırı uyum" demek olur ve hiçbir şey ayırt etmez."""
    alt = cells[(cells.symbol == sym) & (cells.horizon == hz)
                & (cells.direction == d)]
    if not len(alt):
        return None
    adaylar = sorted(alt.stop_sigma_mult.unique())
    if len(adaylar) < 2:
        return None                      # tek aday → PBO tanımsız

    satirlar: List[List[float]] = []
    for donem in ("train", "validation", "test"):
        for rejim in ("ALL", "LOW_VOL", "NORMAL_VOL", "HIGH_VOL", "PANIC"):
            g = alt[(alt.period == donem) & (alt.regime == rejim)]
            if len(g) != len(adaylar):
                continue                 # eksik dilim ATLANIR, doldurulmaz
            g = g.sort_values("stop_sigma_mult")
            satir = []
            for _, x in g.iterrows():
                ev = cell_robust(x.to_dict(), profile).get("robust_ev")
                if ev is None or not math.isfinite(ev):
                    satir = []
                    break
                satir.append(float(ev))
            if satir:
                satirlar.append(satir)

    if len(satirlar) < PBO_MIN_ROWS:
        return None                      # CSCV için yetersiz — UYDURMA YOK
    M = np.array(satirlar, dtype=float)
    # CSCV her dilim çiftini ikiye böler → n_splits ≤ T/2 ve ÇİFT olmalı.
    n = len(satirlar) // 2
    n_splits = max(4, n - (n % 2))
    try:
        v = V.pbo(M, n_splits=n_splits)["pbo"]
        return None if v is None else float(v)
    except Exception:
        return None


def _trial_dispersion(name: Optional[str] = None) -> float:
    """DSR için denenen stratejilerin Sharpe dağılımı.

    Kayıt yoksa 1,0 varsayımı DSR'ı ANLAMSIZ yapar — bu proje bunu bir kez
    yaşadı (0,963 'geçti' → gerçek kayıtla 0,344 'kaldı').

    ⚠️ ÖLÇEK TUZAĞI — ÖLÇÜLDÜ VE DÜZELTİLDİ
    Yıllık Sharpe ufuk uzunluğuna √(yıllık işlem sayısı) ile bağlıdır: aynı
    işlem-başı kenar 5 dakikalık ufukta 24 saatlikten ~130 kat büyük bir
    yıllık Sharpe üretir. Ölçülen dağılımlar:

        5m_LONG   std  92,7      4h_LONG   std 1,69
        15m_LONG  std  27,6     24h_SHORT  std 0,69

    Hepsini tek havuza atmak std'yi 105'e çıkarıyordu; o değerle
    `expected_max_sharpe` devasa çıkar ve DSR HER hücrede 0 olur — kapı
    çalışıyormuş gibi görünür ama aslında hiçbir şey ölçmez.

    Doğrusu: dağılım AYNI ÖLÇEKTEKİ denemelerden (aynı ufuk+yön) alınır.
    Çoklu-test genişliği ise `n_trials` ile korunur: deneme SAYISI bütün
    programdan gelir, YAYILIM aynı ölçekten. İkisi ayrı sorulardır."""
    s = V.trial_sharpes(name)
    if len(s) >= 3:
        return float(max(0.05, np.std(np.asarray(s, dtype=float), ddof=1)))
    if name is not None:                       # gruba düşmediyse havuza düşme
        return 1.0
    s = V.trial_sharpes()
    if len(s) >= 3:
        return float(max(0.05, np.std(np.asarray(s, dtype=float), ddof=1)))
    return 1.0


# ── 6. ana koşu ────────────────────────────────────────────────────────────

def _default_symbols(kok: Path) -> List[str]:
    """Evren `select_universe_5m.py` çıktısından; yoksa indirilmiş veriden."""
    u = kok / "runs" / "qualification" / "universe_5m.json"
    if u.exists():
        try:
            d = json.loads(u.read_text(encoding="utf-8"))
            s = [x["symbol"] for x in d.get("selected", [])]
            if s:
                return s
        except Exception:
            pass
    d5 = kok / "runs" / "data_5m"
    if d5.exists():
        return sorted(p.name.split("_")[0] for p in d5.glob("*_5m.parquet"))
    return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Parite × Ufuk nitelendirme koşusu")
    kok = Path(__file__).resolve().parents[2]
    ap.add_argument("--data-dir", default=str(kok / "runs" / "data_5m"))
    ap.add_argument("--features", default=str(kok / "runs" / "features"))
    ap.add_argument("--out", default=str(kok / "runs" / "qualification"))
    ap.add_argument("--symbols", default=",".join(_default_symbols(kok)))
    ap.add_argument("--horizons", default=",".join(primary_horizons() + ["48h"]))
    ap.add_argument("--reuse-cells", action="store_true",
                    help="cells.parquet varsa taban oranlarını yeniden ölçme")
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--ablation-horizons", default="1h,4h,24h",
                    help="tf/aile ablasyonunun koşulacağı ufuklar (pahalı)")
    a = ap.parse_args(list(argv) if argv is not None else None)
    a.ablation_horizons = set(x for x in a.ablation_horizons.split(",") if x)

    data_dir, out_dir = Path(a.data_dir), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = [s for s in a.symbols.split(",") if s]
    horizons = [h for h in a.horizons.split(",") if h]

    # KOŞU KİMLİĞİ — yeniden üretilebilirlik iddiasının kanıtı (§XCI)
    from .provenance import build as _prov_build
    t_bas = time.time()
    prov = _prov_build(
        seed=SEED,
        data_paths=[data_dir / f"{s}_5m.parquet" for s in symbols],
        pkg_dir=Path(__file__).resolve().parents[1],
        config={"symbols": symbols, "horizons": horizons,
                "train_end": TRAIN_END, "validation_end": VALID_END,
                "model_stride": MODEL_STRIDE,
                "stop_mults": list(STOP_SIGMA_MULTS),
                "target_net_pct": TARGET_NET_PCT,
                "ablation_horizons": sorted(a.ablation_horizons),
                "reuse_cells": bool(a.reuse_cells),
                "skip_model": bool(a.skip_model)})
    print(f"koşu kimliği: {prov.run_id}", flush=True)

    fp = Path(a.features)
    feats = None
    if fp.exists():
        parcalar = sorted(fp.glob("*.parquet"))
        if parcalar:
            feats = pd.concat([pd.read_parquet(x) for x in parcalar],
                              ignore_index=True)
    print(f"kaydedici gözlemi: {0 if feats is None else len(feats)}", flush=True)

    # 1 — taban oranları
    cell_p = out_dir / "cells.parquet"
    if a.reuse_cells and cell_p.exists():
        cells = pd.read_parquet(cell_p)
        print(f"cells.parquet yeniden kullanıldı: {len(cells)} hücre", flush=True)
    else:
        print("── taban oranları ölçülüyor ──", flush=True)
        cells = run_baserates(data_dir, out_dir, feats, symbols, horizons)
    profiles = {s: profile_from_recorder(s, feats) for s in symbols}

    # 2 — stop seçimi (yalnız TRAIN)
    print("── stop seçimi (TRAIN, RobustEV) ──", flush=True)
    stops = select_stops(cells, profiles)
    print(f"  {len(stops)} (parite,ufuk,yön) için stop seçildi", flush=True)

    # 3 — model
    models: Dict[Tuple[str, str], Dict] = {}
    if not a.skip_model:
        for hz in [h for h in horizons if h in primary_horizons()]:
            for d in ("LONG", "SHORT"):
                t0 = time.time()
                ds = build_model_dataset(data_dir, symbols, hz, d, stops, profiles)
                if ds is None:
                    print(f"  {hz}/{d}: veri kümesi kurulamadı", flush=True)
                    continue
                dev = train_and_validate(ds, hz, d,
                                         run_ablation=(hz in a.ablation_horizons))
                if not dev.get("ok"):
                    print(f"  {hz}/{d}: {dev.get('reason')}", flush=True)
                    models[(hz, d)] = {"dev": dev, "test": {"ok": False}}
                    continue
                mdl = SoftmaxModel.from_dict(dev["model"])
                test = locked_test(ds, mdl, dev["pair_heads"],
                                   threshold=dev.get("selection_threshold"),
                                   column_mask=dev.get("column_mask"))
                # SÜRÜKLENME: dev-OOS ile kilitli testteki TAHMİN dağılımları
                # karşılaştırılır. (İlk yazımda bir özellik sütunu ile
                # olasılıklar karşılaştırılıyordu — anlamsızdı ve her hücreye
                # sahte MODEL_DRIFT bayrağı takıyordu.)
                psi_v = None
                if test.get("ok"):
                    from .stats import psi as _psi
                    try:
                        psi_v = float(_psi(np.asarray(dev["p_oos"], dtype=float),
                                           np.asarray(test["p_dist_test"],
                                                      dtype=float)))
                    except Exception:
                        psi_v = None
                V.trial_log(f"qual_model_{hz}_{d}",
                            {"horizon": hz, "direction": d, "model": "softmax_l2"},
                            {"oos_brier": dev["oos_brier"],
                             "oos_brier_base": dev["oos_brier_base"],
                             "top_decile_oos": dev["top_decile_rate"],
                             "test_top_decile": test.get("top_decile_rate")})
                models[(hz, d)] = {"dev": dev, "test": test, "psi": psi_v}
                print(f"  {hz}/{d}: dev n={dev['n_oos']:,} Brier "
                      f"{dev['oos_brier']:.5f} (taban {dev['oos_brier_base']:.5f}) "
                      f"· üst desil {dev['top_decile_rate']} "
                      f"· test {test.get('top_decile_rate')} "
                      f"· {time.time()-t0:.0f} sn", flush=True)

    # 4 — birleştirme
    print("── durum makinesi ──", flush=True)
    gates = EvidenceGates()
    hucreler, ozet = assemble(cells, profiles, stops, models, gates)

    from .matrix import pair_card, rank_pairs, scanner_summary
    kartlar = []
    for sym in sorted({h["symbol"] for h in hucreler}):
        kartlar.append(pair_card(
            sym, [h for h in hucreler if h["symbol"] == sym],
            cost_model=profiles[sym].model,
            model_version=("softmax_l2/1" if models else "yok")))
    sirali = rank_pairs(kartlar)
    tarayici = scanner_summary(kartlar, eligible=len(kartlar), excluded=0)

    temiz = {k: _json_safe(v) for k, v in
             {"scanner": tarayici, "cards": kartlar, "ranked":
              [k["symbol"] for k in sirali], "summary": ozet,
              "regime_note": regime_note(),
              "split": {"train_end": TRAIN_END, "validation_end": VALID_END},
              }.items()}
    (out_dir / "matrix.json").write_text(
        json.dumps(temiz, ensure_ascii=False, indent=1), encoding="utf-8")

    mdl_out = {f"{h}|{d}": {"model": v["dev"].get("model"),
                            "pair_heads": v["dev"].get("pair_heads"),
                            "names": (v["dev"].get("model") or {}).get("names")}
               for (h, d), v in models.items() if v.get("dev", {}).get("ok")}
    (out_dir / "models.json").write_text(
        json.dumps(_json_safe(mdl_out), ensure_ascii=False), encoding="utf-8")

    # Evren taramasının "yeterli tarihsel veri" kapısı için envanter.
    # Ham 5m veri sunucuda tutulmadığından bu dosya ARTEFAKTIN parçasıdır.
    from .universe import bar_inventory
    (out_dir / "data_inventory.json").write_text(json.dumps({
        "bars": bar_inventory(data_dir),
        "source": f"{BASE_TF_MIN}m Binance USD-M vadeli kline",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    prov.finish(t_bas)
    (out_dir / "provenance.json").write_text(
        json.dumps(prov.to_dict(), ensure_ascii=False, indent=1),
        encoding="utf-8")

    # MODEL KAYIT DEFTERİ — her model için kart (§XVI, CII)
    from .model_registry import Registry, card_for_softmax
    reg = Registry(out_dir / "model_registry.json")
    for (hz, d), v in models.items():
        dev, test = v.get("dev", {}), v.get("test", {})
        if not dev.get("ok"):
            continue
        sel = (test.get("selected") or {}) if test.get("ok") else {}
        reg.register(card_for_softmax(
            hz, d, symbols,
            metrics={"oos_n": dev.get("n_oos"),
                     "oos_brier": dev.get("oos_brier"),
                     "oos_brier_base": dev.get("oos_brier_base"),
                     "oos_ece": dev.get("oos_ece"),
                     "calibration_slope": (dev.get("calibration") or {}).get("slope"),
                     "locked_test_n": test.get("n"),
                     "locked_test_tp_rate": sel.get("tp_rate"),
                     "locked_test_net_mean": sel.get("net_mean_pct"),
                     "locked_test_t": sel.get("t_stat"),
                     "psi": v.get("psi")},
            provenance={"run_id": prov.run_id,
                        "dataset_hash": prov.dataset["hash"],
                        "code_hash": prov.code["source_hash"],
                        "config_hash": prov.config_hash, "seed": prov.seed},
            train_end=TRAIN_END, valid_end=VALID_END))
    print(f"model kartı: {len(reg.all())} kayıt "
          f"(hiçbiri APPROVED değil — bağımsız doğrulayıcı yok)", flush=True)

    _write_validation_report(out_dir, models, cells, stops, ozet, prov)
    print(f"\nyazıldı: {out_dir}")
    print("durum dağılımı:", json.dumps(ozet["by_status"], ensure_ascii=False))
    return 0


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()
                if not isinstance(v, np.ndarray)}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if not math.isfinite(f) else f
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, np.ndarray):
        return None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _write_validation_report(out_dir: Path, models: Dict, cells: pd.DataFrame,
                             stops: Dict, ozet: Dict, prov=None) -> None:
    """Şartname 85/86 — rapor yoksa model UNVERIFIED sayılır."""
    rapor = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": {"train": f"< {TRAIN_END}",
                  "validation": f"{TRAIN_END} → {VALID_END}",
                  "locked_test": f"≥ {VALID_END}"},
        "base_resolution": f"{BASE_TF_MIN}m Binance USD-M vadeli kline",
        "n_cells": int(len(cells)),
        "n_stop_selections": len(stops),
        "n_trials_registry": V.trial_count(),
        "trial_dispersion_sharpe_pooled": round(_trial_dispersion(), 4),
        "trial_dispersion_by_horizon": {
            k: round(_trial_dispersion(f"qual_stop_{k}"), 4)
            for k in sorted({f"{h}_{d}" for h in primary_horizons()
                             for d in ("LONG", "SHORT")})},
        "dispersion_note": ("DSR'ın sr_std'si AYNI ÖLÇEKTEKİ denemelerden "
                            "alınır; havuzlanmış 105'lik yayılım ufuklar arası "
                            "√(işlem/yıl) farkından gelir ve DSR'ı anlamsız "
                            "yapardı."),
        "seed": SEED,
        "provenance": (prov.to_dict() if prov is not None else None),
        "models": {},
        "status_distribution": ozet.get("by_status", {}),
        "acceptance": {},
    }
    for (h, d), v in models.items():
        dev, test = v.get("dev", {}), v.get("test", {})
        rapor["models"][f"{h}|{d}"] = _json_safe({
            "ok": dev.get("ok", False),
            "reason": dev.get("reason"),
            "n_dev": dev.get("n_dev"), "n_oos": dev.get("n_oos"),
            "folds": dev.get("folds"),
            "oos_baseline": dev.get("oos_baseline"),
            "oos_brier": dev.get("oos_brier"),
            "oos_brier_base": dev.get("oos_brier_base"),
            "oos_ece": dev.get("oos_ece"),
            "oos_calibration": dev.get("calibration"),
            "oos_deciles": dev.get("deciles"),
            "oos_reliability": dev.get("reliability"),
            "ablation": dev.get("ablation"),
            "pruned_families": dev.get("pruned_families"),
            "dev_selected": dev.get("dev_selected"),
            "selection_threshold": dev.get("selection_threshold"),
            "locked_test": {k: val for k, val in test.items()
                            if k not in ("p_test", "y_test", "symbol_test")},
            "psi": v.get("psi"),
        })
    (out_dir / "validation_report.json").write_text(
        json.dumps(rapor, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
