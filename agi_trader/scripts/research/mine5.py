# -*- coding: utf-8 -*-
"""
%5 HAREKET MADENCİLİĞİ — "yükselişin öncesinde ayırt edici sinyal var mı?"

TASARIM (dürüstlük için açıkça yazılır):
  • Karar noktası: her 5 dakikada bir (ileriye bakış YOK, tüm özellikler geçmişten).
  • Etiket: ilk-geçiş (first passage). Girişten sonra 24 saat içinde
        +%5 ÖNCE gelirse  → 1  (KAZANAN)
        −%2 ÖNCE gelirse  → 0  (STOP)
        ikisi de gelmezse → 0  (ZAMAN AŞIMI, ayrı sayılır)
  • Aynı bar içinde ikisi de görülürse KAYBEDEN sayılır (kötümser; bar içi sıra gözlenemez).
  • BAŞABAŞ kazanma oranı = 2/(5+2) = %28,6 (maliyet hariç). Maliyetle ~%30.
    Yani bir sinyalin işe yaraması için isabet oranını %30'un ÜSTÜNE çıkarması gerekir.
"""
import numpy as np, pandas as pd, glob, os, re, json, collections, math

HIST = os.path.expanduser("~/cmbench/runs/history")
TP, SL, H_BARS, STEP = 0.05, 0.02, 1440, 5     # +%5 / −%2 / 24 saat / 5 dk adım

def son_dosyalar():
    d = {}
    for f in glob.glob(f"{HIST}/*_1m_*.csv"):
        m = re.match(r".*/mexc_(.+?)_1m_(\d+)_(\d+)\.csv", f)
        if not m: continue
        sym, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        if sym not in d or b > d[sym][1]: d[sym] = (f, b)
    return {k: v[0] for k, v in d.items()}

def ozellikler(df):
    c = df["close"].values; h = df["high"].values; l = df["low"].values; v = df["volume"].values
    n = len(c)
    def ret(k):
        r = np.full(n, np.nan); r[k:] = c[k:]/c[:-k] - 1.0; return r
    out = {}
    out["ret_15m"] = ret(15); out["ret_1h"] = ret(60); out["ret_4h"] = ret(240); out["ret_24h"] = ret(1440)
    # hacim: son 15 dk / önceki 4 saat medyanı
    v15 = pd.Series(v).rolling(15).sum().values
    v4h = pd.Series(v).rolling(240).sum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        out["hacim_orani"] = np.where(v4h > 0, v15 / (v4h/16.0), np.nan)
    # oynaklık (1 saatlik getiri std, 4 saatlik pencere)
    r1 = pd.Series(c).pct_change().values
    out["oynaklik_1h"] = pd.Series(r1).rolling(60).std().values * math.sqrt(60) * 100
    # ATR%
    tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
    tr = np.concatenate([[np.nan], tr])
    out["atr_pct"] = pd.Series(tr).rolling(60).mean().values / c * 100
    # RSI(14) 15dk eşdeğeri → 210 dk
    dc = pd.Series(c).diff()
    up = dc.clip(lower=0).rolling(210).mean(); dn = (-dc.clip(upper=0)).rolling(210).mean()
    out["rsi"] = (100 - 100/(1 + up/dn.replace(0, np.nan))).values
    # 24 saatlik aralıktaki konum
    hi24 = pd.Series(h).rolling(1440).max().values; lo24 = pd.Series(l).rolling(1440).min().values
    with np.errstate(divide="ignore", invalid="ignore"):
        out["aralik_konum"] = np.where(hi24 > lo24, (c - lo24) / (hi24 - lo24), np.nan)
        out["tepe_uzaklik"] = (hi24 / c - 1.0) * 100
    # EMA sapması
    e60 = pd.Series(c).ewm(span=60).mean().values
    out["ema60_sapma"] = (c/e60 - 1.0) * 100
    # sıkışma: son 4 saatin aralığı / ATR
    r4 = (pd.Series(h).rolling(240).max().values - pd.Series(l).rolling(240).min().values)/c*100
    with np.errstate(divide="ignore", invalid="ignore"):
        out["sikisma"] = np.where(out["atr_pct"] > 0, r4/out["atr_pct"], np.nan)
    return out

def etiketle(h, l, c, i):
    """i barından sonra +%5 mi −%2 mi önce gelir?"""
    e = c[i]; up = e*(1+TP); dn = e*(1-SL)
    j0, j1 = i+1, min(len(c), i+1+H_BARS)
    if j1 <= j0: return None
    hh = h[j0:j1]; ll = l[j0:j1]
    iu = np.argmax(hh >= up) if (hh >= up).any() else -1
    idn = np.argmax(ll <= dn) if (ll <= dn).any() else -1
    if iu < 0 and idn < 0: return ("timeout", (c[j1-1]/e-1)*100, j1-j0)
    if iu < 0: return ("stop", -SL*100, idn)
    if idn < 0: return ("hedef", TP*100, iu)
    if iu < idn: return ("hedef", TP*100, iu)
    return ("stop", -SL*100, idn)          # aynı bar → kötümser

rows = []
paritedeki = collections.Counter()
files = son_dosyalar()
print(f"parite: {len(files)}")
for sym, f in sorted(files.items()):
    df = pd.read_csv(f, parse_dates=["ts"])
    if len(df) < 3000: continue
    F = ozellikler(df)
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    ts = df["ts"].dt.tz_convert("UTC") if df["ts"].dt.tz is not None else df["ts"].dt.tz_localize("UTC")
    hours = ts.dt.hour.values
    for i in range(1440, len(c)-H_BARS-1, STEP):
        if any(not np.isfinite(F[k][i]) for k in F): continue
        r = etiketle(h, l, c, i)
        if r is None: continue
        rows.append({"sym": sym, "i": i, "saat": int(hours[i]), "sonuc": r[0],
                     "getiri": r[1], "bar": r[2], **{k: float(F[k][i]) for k in F}})
        if r[0] == "hedef": paritedeki[sym] += 1

D = pd.DataFrame(rows)
D["kazanan"] = (D["sonuc"] == "hedef").astype(int)
print(f"karar noktası: {len(D):,}")
print(f"\n{'='*78}\nTABAN ORAN — 24 saat içinde +%5 (−%2 stop'tan ÖNCE)\n{'='*78}")
print(f"  hedef  : {D['kazanan'].sum():,}  (%{100*D['kazanan'].mean():.2f})")
print(f"  stop   : {(D['sonuc']=='stop').sum():,}  (%{100*(D['sonuc']=='stop').mean():.2f})")
print(f"  zaman  : {(D['sonuc']=='timeout').sum():,}  (%{100*(D['sonuc']=='timeout').mean():.2f})")
bek = D["kazanan"].mean()*5 - (D["sonuc"]=="stop").mean()*2 + D.loc[D["sonuc"]=="timeout","getiri"].mean()*(D["sonuc"]=="timeout").mean()
print(f"  BAŞABAŞ için gereken isabet: %{100*SL/(TP+SL):.1f}  ·  ŞU ANKİ: %{100*D['kazanan'].mean():.2f}")
print(f"  ham beklenti (maliyetsiz): {bek:+.4f} %/işlem")
D.to_parquet(os.path.expanduser("~/cmwatch/mine5.parquet")) if False else D.to_csv(os.path.expanduser("~/cmwatch/mine5.csv"), index=False)
print(f"\nkayıt: ~/cmwatch/mine5.csv")
print(f"\n{'='*78}\nPARİTE BAŞINA +%5 EPİZODU (14 gün)\n{'='*78}")
g = D.groupby("sym").agg(n=("kazanan","size"), hedef=("kazanan","sum"))
g["oran%"] = 100*g["hedef"]/g["n"]
g = g.sort_values("oran%", ascending=False)
print(g.head(18).to_string())
print("...")
print(g.tail(8).to_string())
