"""Vektörel ilk-geçiş motoru — şartname 7, 10, 38, 80, 81.

NE YAPAR
Her bar i için, giriş fiyatı close[i] alındığında, verilen bariyer dizilerine
İLK KAÇ BAR SONRA ulaşıldığını hesaplar. Bariyerler bar-başına değişebilir
(ATR'ye bağlı dinamik stop bu sayede bedava gelir).

NEDEN BÖYLE
Naif yöntem her (parite, ufuk, yön, stop) için ayrı tarama yapar → binlerce
tarama. Burada tek gözlem kullanılıyor: **ilk-geçiş süresi ufka bağlı değildir**;
ufuk yalnız KESER. Bu yüzden en uzun ufka kadar bir kez taranır, sonra bütün
ufuklar bu iki sayıdan (t_up, t_dn) türetilir. Şartname 38'in monotonluk
gerekliliği de böylece matematiksel olarak garanti altına alınır: aynı t ile
hesaplanan kümülatif insidans ufuk büyüdükçe azalamaz.

AYNI-BAR BELİRSİZLİĞİ (şartname 7 ve 81 — bu programın en kritik tuzağı)
Bir barın `high`'ı hedefi, `low`'u stop'u aynı anda vurmuşsa hangisinin ÖNCE
geldiği OHLC'den bilinemez. Bu örnek `AMBIGUOUS` sayılır ve ölçümden DÜŞER.
"yukarı saydım" varsayımı backtest'i sistematik olarak iyimser gösterir; bu
modül bu varsayımı yapmaz ve `test_ayni_bar_belirsizligi_basari_sayilmaz`
testi bunu kilitler.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

NEVER = 0                  # t == 0 → ufuk içinde hiç vurulmadı
LABEL_TARGET = 1
LABEL_STOP = -1
LABEL_TIMEOUT = 0
LABEL_AMBIGUOUS = -9       # ölçümden düşer


def first_passage_times(extreme: np.ndarray,
                        barriers: np.ndarray,
                        kmax: int,
                        side: str) -> np.ndarray:
    """Bariyer demetine ilk ulaşma süreleri.

    extreme  : (n,)   'up' için high, 'dn' için low
    barriers : (B, n) her senaryo için bar-başına MUTLAK bariyer fiyatı
    kmax     : taranacak en fazla ileri bar sayısı
    side     : "up" (extreme >= barrier) | "dn" (extreme <= barrier)

    Dönüş    : (B, n) int32, 1..kmax = ilk vuruş barı, 0 = ufuk içinde yok.

    ⚠️ i barındaki giriş için tarama i+1'den başlar — giriş barının kendi
    high/low'u KULLANILMAZ. Bu, look-ahead sızıntısının en yaygın biçimidir
    ve `test_giris_barinin_kendisi_taranmaz` ile kilitlenmiştir.
    """
    if barriers.ndim != 2:
        raise ValueError("barriers (B, n) olmalı")
    B, n = barriers.shape
    if extreme.shape[0] != n:
        raise ValueError("extreme ile barriers aynı uzunlukta olmalı")
    T = np.zeros((B, n), dtype=np.int32)
    ge = (side == "up")
    for k in range(1, kmax + 1):
        m = n - k
        if m <= 0:
            break
        ileri = extreme[k:]                       # (m,) = extreme[i+k]
        bar = barriers[:, :m]                     # (B, m)
        vurdu = (ileri >= bar) if ge else (ileri <= bar)
        hedef = T[:, :m]                          # temel dilim → GÖRÜNÜM
        np.putmask(hedef, vurdu & (hedef == NEVER), k)
    return T


def running_extremes(high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, horizons_bars: List[int]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Her ufuk için MFE/MAE ham girdisi: pencere içi en yüksek/en düşük oran.

    Dönüş: {H: (max_ratio, min_ratio)} — close[i]'ye göre yüzde (i+1..i+H).
    Ufuk sonuna kadar veri yoksa NaN.
    """
    n = len(close)
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    hs = sorted(set(horizons_bars))
    runmax = np.full(n, -np.inf)
    runmin = np.full(n, np.inf)
    k = 0
    for H in hs:
        while k < H:
            k += 1
            m = n - k
            if m <= 0:
                break
            np.maximum(runmax[:m], high[k:], out=runmax[:m])
            np.minimum(runmin[:m], low[k:], out=runmin[:m])
        mx = np.full(n, np.nan)
        mn = np.full(n, np.nan)
        m = max(0, n - H)
        if m:
            mx[:m] = runmax[:m] / close[:m] - 1.0
            mn[:m] = runmin[:m] / close[:m] - 1.0
        out[H] = (mx * 100.0, mn * 100.0)
    return out


@dataclass
class LabelSet:
    """Bir (ufuk, yön, stop) hücresinin etiketleri ve yardımcı büyüklükleri."""
    label: np.ndarray          # +1 hedef, -1 stop, 0 zaman aşımı, -9 belirsiz
    t_hit: np.ndarray          # hedefe/stopa ulaşma barı (yoksa 0)
    t_target: np.ndarray
    t_stop: np.ndarray
    valid: np.ndarray          # ufuk sonuna kadar veri var mı


def label_cell(t_target: np.ndarray, t_stop: np.ndarray,
               horizon_bars: int, n_total: Optional[int] = None) -> LabelSet:
    """İki ilk-geçiş süresinden üçlü-bariyer etiketi.

    ÖNCELİK KURALI
      t_target < t_stop        → +1  (hedef önce)
      t_stop   < t_target      → -1  (stop önce)
      t_target == t_stop       → -9  BELİRSİZ (aynı bar, sıra bilinmiyor)
      ikisi de ufuk dışı       →  0  (zaman aşımı)
    """
    H = horizon_bars
    tt = np.where((t_target > 0) & (t_target <= H), t_target, 0)
    ts = np.where((t_stop > 0) & (t_stop <= H), t_stop, 0)
    hit_t, hit_s = tt > 0, ts > 0

    lab = np.zeros(len(tt), dtype=np.int8)              # varsayılan: zaman aşımı
    onc_t = hit_t & (~hit_s | (tt < ts))
    onc_s = hit_s & (~hit_t | (ts < tt))
    ayni = hit_t & hit_s & (tt == ts)
    lab[onc_t] = LABEL_TARGET
    lab[onc_s] = LABEL_STOP
    lab[ayni] = LABEL_AMBIGUOUS

    t_hit = np.where(onc_t, tt, np.where(onc_s, ts, 0)).astype(np.int32)

    n = n_total if n_total is not None else len(tt)
    gecerli = np.zeros(len(tt), dtype=bool)
    if n - H > 0:
        gecerli[: n - H] = True                          # ufuk kadar ileri veri şart
    return LabelSet(lab, t_hit, tt.astype(np.int32), ts.astype(np.int32), gecerli)


def cif_curve(t_target: np.ndarray, t_stop: np.ndarray,
              horizons_bars: List[int], valid_mask: Optional[np.ndarray] = None
              ) -> List[Dict]:
    """Rekabet eden risk — kümülatif insidans eğrisi (şartname 10, 11, 38).

    CIF_TP(H) = P(hedef, stop'tan ÖNCE ve H içinde)
    CIF_SL(H) = P(stop, hedeften ÖNCE ve H içinde)

    İkisi rakiptir: toplamları 1'i aşamaz, kalan kısım sansürlü (timeout).
    Ham kümülatif insidans H büyüdükçe AZALAMAZ — `monotonic_ok` bunu ölçer.
    """
    out: List[Dict] = []
    n = len(t_target)
    onceki_tp = onceki_sl = -1.0
    mono = True
    for H in sorted(horizons_bars):
        m = np.ones(n, dtype=bool) if valid_mask is None else valid_mask.copy()
        if n - H > 0:
            m[n - H:] = False
        else:
            m[:] = False
        if not m.any():
            out.append({"bars": H, "n": 0, "cif_tp": None, "cif_sl": None,
                        "ambiguous": None})
            continue
        ls = label_cell(t_target[m], t_stop[m], H, n_total=int(m.sum()) + H)
        lab = ls.label
        kesin = lab != LABEL_AMBIGUOUS
        nk = int(kesin.sum())
        tp = float((lab[kesin] == LABEL_TARGET).mean()) if nk else float("nan")
        sl = float((lab[kesin] == LABEL_STOP).mean()) if nk else float("nan")
        if onceki_tp >= 0 and (tp < onceki_tp - 1e-9 or sl < onceki_sl - 1e-9):
            mono = False
        onceki_tp, onceki_sl = tp, sl
        out.append({"bars": H, "n": nk,
                    "n_ambiguous": int((~kesin).sum()),
                    "cif_tp": tp, "cif_sl": sl,
                    "cif_timeout": float(1.0 - tp - sl)})
    for r in out:
        r["monotonic_ok"] = mono
    return out


def monotonic_violation(curve: List[Dict]) -> Optional[str]:
    """Şartname 38 sanity testi — ihlal varsa açıklaması, yoksa None."""
    ontp = onsl = -1.0
    for r in curve:
        if r.get("cif_tp") is None:
            continue
        if ontp >= 0 and r["cif_tp"] < ontp - 1e-9:
            return (f"CIF_TP {r['bars']} barda düştü ({ontp:.4f} → "
                    f"{r['cif_tp']:.4f}) — kümülatif insidans azalamaz")
        if onsl >= 0 and r["cif_sl"] < onsl - 1e-9:
            return (f"CIF_SL {r['bars']} barda düştü ({onsl:.4f} → "
                    f"{r['cif_sl']:.4f})")
        ontp, onsl = r["cif_tp"], r["cif_sl"]
    return None
