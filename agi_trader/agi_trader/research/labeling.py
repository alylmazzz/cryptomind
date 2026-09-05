"""
Üçlü bariyer etiketleme ve örnek benzersizlik ağırlıkları (FAZ 4).

Kaynak: López de Prado, "Advances in Financial Machine Learning", bölüm 3-4.

NEDEN SABİT-UFUK ETİKETLEME YETMEZ: "5 bar sonraki getiri pozitif mi?" etiketi,
gerçek işlemin nasıl kapandığını yansıtmaz — gerçekte pozisyon ya hedefe, ya
stopa, ya da süre dolduğunda kapanır. Üçlü bariyer tam olarak bunu modeller ve
etiketi volatiliteye göre ölçekler (sakin piyasada %1 çok, oynak piyasada az).

ÖRNEK BENZERSİZLİĞİ NEDEN GEREKLİ: ardışık barların etiketleri ÖRTÜŞÜR (t ve t+1
etiketleri büyük ölçüde aynı fiyat hareketini kullanır). Bunları bağımsız örnek
saymak, modeli olduğundan emin gösterir ve çapraz doğrulamayı bozar. Ağırlıklar
örtüşmeyi düzeltir; `research.validation.purged_kfold_splits` ise bölmeyi düzeltir.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ===========================================================================
# Volatilite tahmini (bariyer ölçeği)
# ===========================================================================
def daily_volatility(close: pd.Series, span: int = 50) -> pd.Series:
    """Üstel ağırlıklı getiri volatilitesi — bariyer genişliği bununla ölçeklenir."""
    r = close.pct_change()
    return r.ewm(span=span, min_periods=max(5, span // 5)).std()


# ===========================================================================
# Üçlü bariyer
# ===========================================================================
def triple_barrier_labels(close: pd.Series,
                          side: Optional[pd.Series] = None,
                          pt_mult: float = 2.0,
                          sl_mult: float = 2.0,
                          max_hold: int = 10,
                          vol_span: int = 50,
                          min_ret: float = 0.0) -> pd.DataFrame:
    """Her bar için üçlü bariyer sonucu.

    close    : fiyat serisi
    side     : birincil modelin yönü (+1 long / -1 short / 0 işlem yok).
               Verilirse META-ETİKET üretilir: `bin` = 1 ise "birincil model
               haklıydı, pozisyon aç", 0 ise "haklı değildi, açma".
               Verilmezse yön etiketi üretilir (bin ∈ {-1,0,1}).
    pt_mult  : kâr al bariyeri = pt_mult × volatilite
    sl_mult  : zarar kes bariyeri = sl_mult × volatilite
    max_hold : dikey bariyer (bar sayısı)
    min_ret  : bu getirinin altındaki sonuçlar 0 sayılır (gürültü eşiği)

    Döner: DataFrame[t1(int konum), ret, bin, barrier]
           barrier ∈ {"pt","sl","vertical"}
    """
    c = close.astype(float).reset_index(drop=True)
    n = len(c)
    vol = daily_volatility(c, vol_span).fillna(0.0).values
    px = c.values
    sd = (side.astype(float).reset_index(drop=True).values
          if side is not None else np.ones(n))

    t1 = np.full(n, -1, dtype=int)
    ret = np.zeros(n, dtype=float)
    barrier = np.array(["vertical"] * n, dtype=object)

    for i in range(n - 1):
        if sd[i] == 0 or vol[i] <= 0:
            t1[i] = min(i + max_hold, n - 1)
            continue
        up = px[i] * (1 + pt_mult * vol[i])
        dn = px[i] * (1 - sl_mult * vol[i])
        end = min(i + max_hold, n - 1)
        hit = end
        b = "vertical"
        for k in range(i + 1, end + 1):
            if px[k] >= up:
                hit, b = k, "pt"
                break
            if px[k] <= dn:
                hit, b = k, "sl"
                break
        t1[i] = hit
        barrier[i] = b
        ret[i] = (px[hit] / px[i] - 1.0) * sd[i]     # yöne göre işaretli getiri

    if side is None:
        bins = np.sign(ret)
        bins[np.abs(ret) < min_ret] = 0
    else:
        # meta-etiket: birincil modelin yönü KÂRLI mıydı?
        bins = (ret > min_ret).astype(float)
        bins[sd == 0] = 0

    return pd.DataFrame({"t1": t1, "ret": ret, "bin": bins, "barrier": barrier},
                        index=close.index)


# ===========================================================================
# Örnek benzersizliği
# ===========================================================================
def concurrency(t1: np.ndarray, n: int) -> np.ndarray:
    """Her barda kaç etiketin 'canlı' olduğunu sayar."""
    cnt = np.zeros(n, dtype=float)
    for i, end in enumerate(t1):
        if end < i:
            continue
        cnt[i:end + 1] += 1.0
    return cnt


def uniqueness_weights(t1: "np.ndarray | pd.Series") -> np.ndarray:
    """Örnek ağırlığı = etiket süresi boyunca ortalama 1/eşzamanlılık.

    Çok örtüşen örnekler düşük ağırlık alır. Bu yapılmazsa model, aynı fiyat
    hareketini defalarca görüp ezberler ve çapraz doğrulama iyimser çıkar."""
    a = np.asarray(t1, dtype=int)
    n = len(a)
    cnt = concurrency(a, n)
    w = np.zeros(n, dtype=float)
    for i, end in enumerate(a):
        if end < i:
            continue
        seg = cnt[i:end + 1]
        seg = seg[seg > 0]
        w[i] = float((1.0 / seg).mean()) if seg.size else 0.0
    s = w.sum()
    return w * (n / s) if s > 0 else np.ones(n)


def label_summary(lab: pd.DataFrame) -> Dict:
    """Etiket dağılımı — dengesizlik modeli sessizce bozar, önce buna bak."""
    b = lab["bin"].values
    br = pd.Series(lab["barrier"]).value_counts().to_dict()
    return {"n": int(len(b)),
            "pozitif_oran": round(float((b > 0).mean()), 4),
            "sifir_oran": round(float((b == 0).mean()), 4),
            "negatif_oran": round(float((b < 0).mean()), 4),
            "ortalama_getiri": round(float(lab["ret"].mean()), 6),
            "bariyer_dagilimi": {k: int(v) for k, v in br.items()},
            "ortalama_tutma": round(float((lab["t1"].values -
                                           np.arange(len(lab))).mean()), 2)}
