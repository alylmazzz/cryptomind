"""
VİDEO KAYNAKLI KURULUMLAR — YouTube'da "günlük %10", "%90 kazanma" iddiasıyla anlatılan
stratejilerin MEKANİK çekirdekleri, kaynak künyesiyle ve iddia/kanıt kaydıyla birlikte.

NEDEN BÖYLE BİR MODÜL: 21 videonun transkripti çıkarıldı (2026-09-04). Hiçbirinde üçüncü
parti denetimli (myfxbook/broker onaylı) günlük %10 kanıtı YOK; en iyi kanıtlar 303 işlemlik
bir geri test (kazanma %36, R/R 1:4), 31 işlemlik tek haftalık manuel test ve 6-10 işlemlik
mini testler. İDDİALARI ALMIYORUZ; anlattıkları KURULUMLARI alıyoruz.

**SONUÇ (ölçüldü, bkz. MEASURED): 10 kurulumun HİÇBİRİ pozitif kenar göstermedi.** 33 parite ×
7 gün gerçek 1 dk veride 7.343 ham aday, maliyet %0,14 düşülmüş: toplam ortalama net −0,129%,
t −20,4. Tek günlük ilk ölçümde New York seansı pozitif görünmüştü (t +3,38); 7 günde bu
DOĞRULANMADI (t −0,90) — örneklem gürültüsüydü. Bu yüzden kurulumlar `lifecycle.SHADOW_SLEEVES`
ile **GÖLGEDE** doğar: sinyal üretir, emir VERMEZ, kaçırılan-fırsat motorunda ölçülmeye devam
eder. Kanıt pozitife dönerse (`scripts/cm_replay.py --evidence` kapıları) PAPER'a terfi eder ve
o zaman `learn/allocator.py` kanıt tavanı (25 $) devreye girer. Kanıtsız kenarla emir gönderilmez.

ALINMAYANLAR (dürüstlük): kapalı kaynak göstergeler (MT4 Glass, "Bull Trading", "AI trading",
"Manipulation X") — formülü yayımlanmamış, yeniden üretilemez. "Gözle çizilen trend çizgisi",
"bariz likidite seviyesi", "sağlam görünen bölge" gibi eşiksiz ifadeler proxy ile (swing/pivot
tespiti, ATR eşiği) yaklaşık kodlanır ve bu NOTLARDA yazılır.

Her sleeve `{kind, direction, size, exit_mode, note, stop_hint?, target_hint?}` döndürür;
`committee.evaluate` EV yarışmasına sokar, veto rolleri ve kanıt kapıları aynen uygulanır.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

EXIT_FIXED, EXIT_PARTIAL_RUN, EXIT_DYNAMIC_PEAK = "FIXED_TARGET", "PARTIAL_AND_RUN", "DYNAMIC_PEAK"

# ---------------------------------------------------------------------------
# KAYNAK KÜNYESİ — panelde "bu kurulum nereden geldi, iddia neydi, kanıtı neydi"
# ---------------------------------------------------------------------------
SOURCES: Dict[str, Dict] = {
    "fvg_fill": {
        "sleeve_tr": "adil değer boşluğu dolumu (FVG)",
        "channels": ["Chart Fanatics / TG Capital", "PB Blake", "Craig Percoco", "Data Trader"],
        "claim": "%90 kazanma · 1:20+ R/R · 303 işlemlik geri testte kazanma %36 / R 1:4",
        "evidence": "BACKTEST_SMALL",       # tek doğrulanabilir sayı: Percoco n=303, kazanma %36, 1:4 R
        "note": "4 videoda ortak. Giriş: 3 mumlu boşluğun %50'sine geri dönüş. Boşluk tespiti tam mekanik.",
    },
    "ifvg_reclaim": {
        "sleeve_tr": "ters çevrilen boşluk (inverse FVG)",
        "channels": ["Data Trader", "LuxAlgo (ICT Silver Bullet)"],
        "claim": "sayı verilmedi — 'tutarlı kâr'",
        "evidence": "NONE",
        "note": "Ayı boşluğunun üstüne gövde kapanışı = boşluk ters çevrildi. Likidite süpürmesiyle birlikte aranır.",
    },
    "range_reclaim": {
        "sleeve_tr": "aralık dışına taşıp geri dönüş (4h range reclaim)",
        "channels": ["Data Trader", "Broken Lollipop"],
        "claim": "BTC 7 işlem 5G/2K +8R · EURUSD 6 işlem 5G/1K +9R · XAU 10 işlem 6G/4K +8R",
        "evidence": "BACKTEST_TINY",        # sunucunun kendi itirafı: n çok küçük
        "note": "En mekanik iki kurulumdan biri: kapanmış 4h aralığı, mum KAPANIŞIYLA dışarı, sonra içeri; fitil sayılmaz.",
    },
    "manipulation_candle": {
        "sleeve_tr": "manipülasyon mumu (önceki dibi al, tepesinin üstünde kapat)",
        "channels": ["Funded Brothers"],
        "claim": "'%100 mekanik, tutarlı kâr' — sayı yok, TradingView replay modunda gösterildi",
        "evidence": "REPLAY_DEMO",
        "note": "Tam tanımlı: low[t] < low[t-1] VE close[t] > high[t-1]. Hacim POC/değer alanı kenarında aranır.",
    },
    "opening_range": {
        "sleeve_tr": "açılış aralığı kırılımı + yeniden test",
        "channels": ["Jdub Trades"],
        "claim": "bir ayda 65.000 $ (yalnız kâr/zarar grafiği ekran görüntüsü)",
        "evidence": "SCREENSHOT",
        "note": "Seans açılışının ilk penceresinin yüksek/düşüğü; kapanışla kırılım, sonra seviyeye geri test.",
    },
    "ema_engulf": {
        "sleeve_tr": "EMA dizilimi + yutan mum (1s bağlam)",
        "channels": ["TrippaTrading Türkçe"],
        "claim": "1000 $ ile günde 100 $ (%10/gün) · 7 günlük elle geri test: 31 işlem, 16 kazanan (%51,6), R/R 1:2",
        "evidence": "MANUAL_BACKTEST_31",
        "note": "Türkçe videolar içinde en kodlanabiliri. İşlem başına %10 risk öneriyor — biz %1 kullanırız (risk of ruin).",
    },
    "poc_reversion": {
        "sleeve_tr": "hacim yoğunluk noktası (POC/değer alanı) dönüşü",
        "channels": ["Funded Brothers"],
        "claim": "sayı yok",
        "evidence": "REPLAY_DEMO",
        "note": "POC/VAH/VAL bar-içi hacim dağılımından YAKLAŞIK (tik verisi yok) — kova bazlı, nedensel hesap.",
    },
    "order_block": {
        "sleeve_tr": "emir bloğu (yapı kırılımı öncesi son zıt mum) yeniden testi",
        "channels": ["Jdub Trades", "PB Blake", "LuxAlgo"],
        "claim": "500.000 $ / '%70 kazanma, doğru uygulanırsa'",
        "evidence": "NONE",
        "note": "MSS = gövde kapanışıyla son onaylı swing tepesinin üstü; blok = ondan önceki son kırmızı mum.",
    },
    "stoch_cross_back": {
        "sleeve_tr": "EMA'ya geri kapanış + stokastik kesişimi",
        "channels": ["Trader DNA"],
        "claim": "'%90 kazanma potansiyeli' — ölçüm değil, varsayımsal olasılık tablosu",
        "evidence": "NONE",
        "note": "Kapalı kaynak 'non-repaint' gösterge ALINMADI; yalnız EMA20 geri kapanışı + Stoch(8,5,3) kesişimi.",
    },
    "bb_lower_band": {
        "sleeve_tr": "Bollinger alt bandı → orta bant (yatay parite)",
        "channels": ["Expert Para"],
        "claim": "günlük 50-100 $ (kişisel beyan, ekran görüntüsü yok)",
        "evidence": "ANECDOTE",
        "note": "Yalnız DAR bantlı (yatay) paritede; hedef orta bant. Bant genişliği eşiği bizim eklediğimiz (videoda yok).",
    },
}

# İddia edilen ama ALINMAYAN unsurlar — panelde dürüstlük notu olarak gösterilir
NOT_IMPLEMENTED = [
    {"item": "Kapalı kaynak göstergeler (MT4 Glass · 'Bull Trading' · 'AI trading' · 'Manipulation X' · QQE Signal)",
     "why": "formülü yayımlanmadı, bağımsız yeniden üretilemez → kod kör kopyalanmaz"},
    {"item": "Elle çizilen trend çizgisi / 'bariz likidite seviyesi' / 'sağlam bölge'",
     "why": "eşiksiz, iki kişi farklı çizer; swing-pivot + ATR eşiğiyle YAKLAŞIK proxy kullanıldı"},
    {"item": "50x-100x kaldıraç ve işlem başına %10 risk",
     "why": "iflas olasılığı: %10 riskle 7 ardışık kayıp sermayenin yarısını siler; sistem %1 risk + kanıt tavanı kullanır"},
    {"item": "'Stop kullanmadan tut' / 'düşerse ortalama düşür'",
     "why": "kill-switch ve stop zorunlu; martingale yasak (drawdown'da boyut YARIYA iner, artmaz)"},
    {"item": "Günlük %10 hedefi",
     "why": "bileşik olarak yılda ~1.400.000× eder; hiçbir videoda üçüncü parti denetimli kanıt yok. Hedef DEĞİL, ölçüm var."},
]

SLEEVE_TR_VIDEO = {k: v["sleeve_tr"] for k, v in SOURCES.items()}
SLEEVE_EXIT_MODE_VIDEO = {
    "fvg_fill": EXIT_PARTIAL_RUN, "ifvg_reclaim": EXIT_PARTIAL_RUN, "range_reclaim": EXIT_FIXED,
    "manipulation_candle": EXIT_PARTIAL_RUN, "opening_range": EXIT_DYNAMIC_PEAK, "ema_engulf": EXIT_FIXED,
    "poc_reversion": EXIT_FIXED, "order_block": EXIT_PARTIAL_RUN, "stoch_cross_back": EXIT_FIXED,
    "bb_lower_band": EXIT_FIXED,
}
SLEEVE_TIME_STOP_MIN_VIDEO = {
    "fvg_fill": 120, "ifvg_reclaim": 90, "range_reclaim": 120, "manipulation_candle": 90,
    "opening_range": 180, "ema_engulf": 120, "poc_reversion": 90, "order_block": 120,
    "stoch_cross_back": 45, "bb_lower_band": 240,
}
# maker kaç bar bekleyebilir (0 = acil). Video kurulumlarının hepsi mum KAPANIŞI bekler → sabırlı.
SLEEVE_URGENCY_VIDEO = {
    "fvg_fill": 2, "ifvg_reclaim": 1, "range_reclaim": 1, "manipulation_candle": 1,
    "opening_range": 0, "ema_engulf": 1, "poc_reversion": 2, "order_block": 2,
    "stoch_cross_back": 1, "bb_lower_band": 2,
}
# rejim → hangi video sleeve'leri açık (zıt sistemler aynı anda oy kullanmasın)
REGIME_SLEEVES_VIDEO = {
    "TREND YUKARI": ["fvg_fill", "order_block", "opening_range", "ema_engulf", "range_reclaim", "ifvg_reclaim"],
    "RANGE / YATAY": ["range_reclaim", "manipulation_candle", "poc_reversion", "bb_lower_band",
                      "stoch_cross_back", "ifvg_reclaim", "fvg_fill"],
    "VOLATİL": ["manipulation_candle", "ifvg_reclaim", "range_reclaim"],
    "TREND AŞAĞI": ["manipulation_candle", "ifvg_reclaim"],
}
ALL_VIDEO_SLEEVES = sorted(SOURCES)

# ---------------------------------------------------------------------------
# SEANS PENCERELERİ (UTC) — 4 videoda "kill zone" filtresi vardı (Londra/NY açılışı).
# Kripto 7/24 ama hacim bu pencerelerde yoğunlaşır. ÖNSEL bir yasak DEĞİL: yalnız etiket +
# `session_gate` (ölçülmüş beklenti) boyutu belirler. Buradaki pencereler bilgi amaçlıdır.
# ---------------------------------------------------------------------------
KILLZONES = [("ASYA", 0, 3), ("LONDRA", 7, 10), ("NY_AM", 13, 16), ("LONDRA_KAPANIS", 15, 17)]

# ÖLÇÜLMÜŞ SEANS ÖNCÜLÜ — İKİ AŞAMALI, ve ikincisi birinciyi ÇÜRÜTTÜ (kayıt dürüstlük için duruyor):
#
#  1) 1 GÜNLÜK (33 parite, 868 aday): NY_AM +0,141% t +3,38 · ASYA −0,241% t −5,72 · seans dışı −0,145% t −6,32.
#     Videolardaki "kill zone" iddiası doğrulanmış göründü.
#  2) 7 GÜNLÜK (aynı 33 parite, 7.343 aday, aynı yöntem): NY_AM **−0,017% t −0,90** — POZİTİF DEĞİL.
#     Bütün seanslar negatif (LONDRA −0,244 t −14,5 en kötü). Tek günlük bulgu ÖRNEKLEM GÜRÜLTÜSÜYMÜŞ.
#
# Sonuç: seansın GÖRECELİ sıralaması korunuyor (NY_AM en iyi pencere) ama MUTLAK kenar hiçbir pencerede
# pozitif değil. Bu yüzden çarpanlar duruyor fakat kurulumların kendisi `lifecycle.SHADOW_SLEEVES` ile
# GÖLGEYE alındı: sinyal üretirler, emir VERMEZLER. Çarpan ancak terfi hâlinde etkili olur.
SESSION_SIZE_MULT = {"NY_AM": 1.0, "LONDRA": 0.5, "LONDRA_KAPANIS": 0.6, "ASYA": 0.5, None: 0.6}
MEASURED = {
    "primary": {
        "window": "2026-08-28 → 2026-09-04 UTC · 33 parite · 1 dk · 7 gün",
        "n_candidates": 7343, "cost_pct_roundtrip": 0.14,
        "all": {"n": 7343, "mean_net_pct": -0.129, "t": -20.37, "win_rate": 37.0},
        "by_session": {"NY_AM": {"n": 1129, "mean_net_pct": -0.017, "t": -0.90},
                       "SEANS_DIŞI": {"n": 4340, "mean_net_pct": -0.125, "t": -15.89},
                       "ASYA": {"n": 883, "mean_net_pct": -0.171, "t": -8.76},
                       "LONDRA_KAPANIS": {"n": 232, "mean_net_pct": -0.234, "t": -6.31},
                       "LONDRA": {"n": 759, "mean_net_pct": -0.244, "t": -14.45}},
        "by_sleeve": {"range_reclaim": {"n": 947, "mean_net_pct": -0.074, "t": -2.73},
                      "bb_lower_band": {"n": 181, "mean_net_pct": -0.093, "t": -2.68},
                      "manipulation_candle": {"n": 326, "mean_net_pct": -0.121, "t": -4.64},
                      "opening_range": {"n": 417, "mean_net_pct": -0.128, "t": -2.88},
                      "ifvg_reclaim": {"n": 1264, "mean_net_pct": -0.133, "t": -10.33},
                      "order_block": {"n": 1623, "mean_net_pct": -0.144, "t": -13.42},
                      "fvg_fill": {"n": 1056, "mean_net_pct": -0.146, "t": -10.45},
                      "stoch_cross_back": {"n": 982, "mean_net_pct": -0.149, "t": -11.67},
                      "ema_engulf": {"n": 135, "mean_net_pct": -0.154, "t": -3.98},
                      "poc_reversion": {"n": 412, "mean_net_pct": -0.115, "t": -4.98}},
        "best_cells_ny_am": {"manipulation_candle": {"n": 60, "mean_net_pct": 0.118, "t": 1.95},
                             "opening_range": {"n": 59, "mean_net_pct": 0.112, "t": 1.04},
                             "range_reclaim": {"n": 182, "mean_net_pct": 0.090, "t": 1.13},
                             "bb_lower_band": {"n": 38, "mean_net_pct": 0.052, "t": 0.67},
                             "poc_reversion": {"n": 71, "mean_net_pct": 0.033, "t": 0.55}},
        "verdict": "SHADOW",
        "note": ("Ham sinyal (komite oylaması/vetoları OLMADAN) HİÇBİR pencerede pozitif değil. NY_AM'de 5 kurulum "
                 "pozitif hücre veriyor ama en iyisi t +1,95 — 50 hücre denendiği için çoklu-test eşiğinin (t≈3) "
                 "altında. Kanıtsız kenarla emir gönderilmez: kurulumlar GÖLGEDE ölçülmeye devam eder."),
    },
    "superseded_1day": {
        "window": "2026-09-03 06:47 → 2026-09-04 06:55 UTC · 33 parite · 1 gün",
        "n_candidates": 868,
        "by_session": {"NY_AM": {"n": 190, "mean_net_pct": 0.141, "t": 3.38},
                       "ASYA": {"n": 156, "mean_net_pct": -0.241, "t": -5.72},
                       "SEANS_DIŞI": {"n": 485, "mean_net_pct": -0.145, "t": -6.32}},
        "note": "7 günlük ölçüm bunu çürüttü (NY_AM t +3,38 → −0,90). Örneklem gürültüsü kaydı olarak duruyor.",
    },
}
MEASURED_2026_09_04 = MEASURED["primary"]        # geriye dönük ad


def killzone_of(ts: float) -> Optional[str]:
    import time as _t
    h = _t.gmtime(float(ts)).tm_hour
    for name, a, b in KILLZONES:
        if a <= h < b:
            return name
    return None


def session_size_mult(ts: Optional[float]) -> float:
    """Seans boyut çarpanı (başlangıç öncülü). Kapatmaz — ölçüm sürsün diye küçültür."""
    if ts is None:
        return 1.0
    return float(SESSION_SIZE_MULT.get(killzone_of(ts), 0.5))


# ---------------------------------------------------------------------------
# ÖZELLİKLER — hepsi aynı 1 dk DataFrame'inden, tek geçişte, ileriye bakış YOK
# ---------------------------------------------------------------------------
def _pivots(h: np.ndarray, l: np.ndarray, n: int = 2):
    """Fraktal pivot: high[i] son n ve sonraki n barın üstündeyse swing tepe (i ≤ len-n-1 → onaylı)."""
    hi_idx, lo_idx = [], []
    m = len(h)
    for i in range(n, m - n):
        if h[i] == max(h[i - n:i + n + 1]) and h[i] > max(h[i - n:i]) :
            hi_idx.append(i)
        if l[i] == min(l[i - n:i + n + 1]) and l[i] < min(l[i - n:i]):
            lo_idx.append(i)
    return hi_idx, lo_idx


def _fvg_zones(h: np.ndarray, l: np.ndarray, price: float, atr_abs: float, lookback: int = 60):
    """3 mumlu adil değer boşlukları. Boğa: low[i] > high[i-2] → bölge [high[i-2], low[i]].
    Ayı: high[i] < low[i-2] → bölge [high[i], low[i-2]].

    Döndürür (bull, bear, bear_raw):
      bull      — GEÇERLİ (henüz aşağı doldurulmamış) boğa boşluğu; `fvg_fill` bunun İÇİNE dönüşü arar
      bear      — geçerli (henüz yukarı aşılmamış) ayı boşluğu; direnç olarak taşınır
      bear_raw  — aşılma filtresi UYGULANMAMIŞ en son ayı boşluğu; `ifvg_reclaim` bunun ÜSTÜNE
                  kapanışı arar. (Filtreli bölgeyle aranırsa mantıksal olarak asla ateşlemez:
                  "hiç aşılmamış" ile "şimdi aşıldı" aynı anda doğru olamaz — canlı veride 0 ateşleme
                  bu hatayı gösterdi.)"""
    m = len(h)
    lo_i = max(2, m - lookback)
    bull = bear = bear_raw = None
    for i in range(m - 1, lo_i - 1, -1):
        if i - 2 < 0:
            break
        if bull is None and l[i] > h[i - 2]:
            z = (float(h[i - 2]), float(l[i]))
            # sonradan tamamen doldurulduysa (fiyat bölgenin altına indi) geçersiz
            if float(np.min(l[i:])) > z[0] and z[1] > z[0]:
                bull = {"lo": z[0], "hi": z[1], "mid": (z[0] + z[1]) / 2.0, "age": m - 1 - i}
        if h[i] < l[i - 2]:
            z = (float(h[i]), float(l[i - 2]))
            if z[1] > z[0]:
                if bear_raw is None:
                    bear_raw = {"lo": z[0], "hi": z[1], "mid": (z[0] + z[1]) / 2.0, "age": m - 1 - i}
                if bear is None and float(np.max(h[i:])) < z[1]:
                    bear = {"lo": z[0], "hi": z[1], "mid": (z[0] + z[1]) / 2.0, "age": m - 1 - i}
        if bull and bear and bear_raw:
            break
    return bull, bear, bear_raw


def _volume_profile(c: np.ndarray, v: np.ndarray, bars: int = 240, bins: int = 48):
    """Kova bazlı hacim profili (tik verisi yok → bar kapanışına hacim atanır; YAKLAŞIKTIR).
    POC = en çok hacimli kova; değer alanı = POC'tan başlayıp komşu kovaları hacme göre ekleyerek
    toplam hacmin %70'ini kapsayan en dar aralık."""
    n = min(len(c), bars)
    if n < 40:
        return None
    cc, vv = c[-n:], v[-n:]
    lo, hi = float(np.min(cc)), float(np.max(cc))
    if not (hi > lo):
        return None
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(cc, edges) - 1, 0, bins - 1)
    vol = np.zeros(bins)
    np.add.at(vol, idx, vv)
    total = float(vol.sum())
    if total <= 0:
        return None
    p = int(np.argmax(vol))
    a = b = p
    acc = vol[p]
    while acc < 0.70 * total and (a > 0 or b < bins - 1):
        left = vol[a - 1] if a > 0 else -1.0
        right = vol[b + 1] if b < bins - 1 else -1.0
        if right >= left:
            b += 1; acc += vol[b]
        else:
            a -= 1; acc += vol[a]
    mid = (edges[:-1] + edges[1:]) / 2.0
    return {"poc": float(mid[p]), "val": float(edges[a]), "vah": float(edges[b + 1]),
            "width_pct": float((edges[b + 1] - edges[a]) / max(1e-12, mid[p]) * 100.0)}


def _stoch(h: pd.Series, l: pd.Series, c: pd.Series, k: int = 8, d: int = 5, smooth: int = 3):
    """Stokastik (8,5,3) — Trader DNA'nın verdiği parametreler."""
    if len(c) < k + d + smooth + 2:
        return None, None
    ll = l.rolling(k).min(); hh = h.rolling(k).max()
    raw = 100.0 * (c - ll) / (hh - ll).replace(0, np.nan)
    kk = raw.rolling(smooth).mean()
    dd = kk.rolling(d).mean()
    a, b = float(kk.iloc[-1]), float(dd.iloc[-1])
    return (a if math.isfinite(a) else None), (b if math.isfinite(b) else None)


def _htf_range(df: pd.DataFrame, block_hours: int = 4):
    """Son KAPANMIŞ UTC blok aralığı (00-04, 04-08, ...) — Broken Lollipop / Data Trader kurulumu
    'son kapanmış 4 saatlik mum' diyor. İleriye bakış yok: yalnız kapanmış blok kullanılır."""
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 30:
        return None
    idx = df.index
    try:
        hours = idx.hour.to_numpy()
        days = idx.normalize().astype("int64").to_numpy()
    except Exception:
        return None
    block = days // 10 ** 9 * 100 + (hours // block_hours)      # gün + blok no
    cur = block[-1]
    prev_mask = block == np.roll(block, 0)
    uniq = np.unique(block)
    if len(uniq) < 2:
        return None
    prev_id = uniq[np.searchsorted(uniq, cur) - 1]
    m = block == prev_id
    if m.sum() < 10:
        return None
    hh = float(df["high"].to_numpy()[m].max())
    ll = float(df["low"].to_numpy()[m].min())
    cur_m = block == cur
    return {"hi": hh, "lo": ll, "bars_in_current": int(cur_m.sum()),
            "cur_hi": float(df["high"].to_numpy()[cur_m].max()) if cur_m.any() else hh,
            "cur_lo": float(df["low"].to_numpy()[cur_m].min()) if cur_m.any() else ll}


def _opening_range(df: pd.DataFrame, minutes: int = 15, block_hours: int = 4):
    """Mevcut UTC bloğunun ilk `minutes` dakikasının yüksek/düşüğü (Jdub'un 'seans ilk mumu')."""
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < minutes + 5:
        return None
    idx = df.index
    try:
        hours = idx.hour.to_numpy(); mins = idx.minute.to_numpy()
        days = idx.normalize().astype("int64").to_numpy()
    except Exception:
        return None
    block = days // 10 ** 9 * 100 + (hours // block_hours)
    cur = block[-1]
    m = block == cur
    if m.sum() <= minutes:
        return None
    pos = np.where(m)[0]
    first = pos[:minutes]
    or_hi = float(df["high"].to_numpy()[first].max())
    or_lo = float(df["low"].to_numpy()[first].min())
    return {"hi": or_hi, "lo": or_lo, "bars_since": int(len(pos) - minutes)}


def video_features(df: pd.DataFrame, f: Dict) -> Dict:
    """`committee.fast_features` + `sleeves_fast.extra_features` çıktısını video kurulumlarının
    ihtiyaç duyduğu alanlarla genişletir. AYNI DataFrame, ek veri çekimi YOK."""
    try:
        c = df["close"].astype(float)
        h = df["high"].astype(float) if "high" in df else c
        l = df["low"].astype(float) if "low" in df else c
        o = df["open"].astype(float) if "open" in df else c.shift(1).fillna(c)
        v = df["volume"].astype(float) if "volume" in df else pd.Series(np.ones(len(c)), index=c.index)
        price = float(c.iloc[-1])
        n = len(c)
        atr_pct = float(f.get("atr_pct") or 0.3)
        atr_abs = max(1e-12, atr_pct / 100.0 * price)
        ha, la, ca, oa, va = h.to_numpy(), l.to_numpy(), c.to_numpy(), o.to_numpy(), v.to_numpy()

        # 1) adil değer boşlukları
        bull_fvg, bear_fvg, bear_raw = _fvg_zones(ha, la, price, atr_abs)
        in_bull_fvg = bool(bull_fvg and bull_fvg["lo"] <= price <= bull_fvg["hi"])
        # inverse FVG: ayı boşluğunun ÜSTÜNE gövde kapanışı (boşluk ters çevrildi → LONG).
        # Filtresiz (`bear_raw`) bölge kullanılır; filtreli bölgeyle koşul mantıksal olarak imkânsızdı.
        # SIKILAŞTIRMA (ölçüldü: gevşek hâli pencerelerin %9,5'inde ateşliyordu = gürültü): boşluk
        # anlamlı genişlikte (≥ 0,15 ATR) ve en az 3 bar yaşında olmalı. Kaynak video (Data Trader)
        # zaten "likidite süpürmesi → boşluk → ters çevirme" diyor; süpürme/trend teyidi sleeve'de aranır.
        ifvg_up = bool(bear_raw and price > bear_raw["hi"] and float(c.iloc[-2]) <= bear_raw["hi"]
                       and (bear_raw["hi"] - bear_raw["lo"]) >= 0.15 * atr_abs and bear_raw["age"] >= 3)

        # 2) manipülasyon mumu (Funded Brothers, tam tanımlı)
        manip_bull = bool(n > 2 and la[-1] < la[-2] and ca[-1] > ha[-2])
        manip_bear = bool(n > 2 and ha[-1] > ha[-2] and ca[-1] < la[-2])

        # 3) yutan mum (Trippa)
        body_prev = abs(ca[-2] - oa[-2]) if n > 2 else 0.0
        engulf_bull = bool(n > 2 and ca[-1] > oa[-1] and ca[-1] > ha[-2] and la[-1] < la[-2]
                           and abs(ca[-1] - oa[-1]) > max(body_prev, 0.15 * atr_abs))

        # 4) üst blok aralığı + açılış aralığı
        rng = _htf_range(df)
        orr = _opening_range(df)
        range_reclaim_up = False
        range_break_lo = None
        if rng and n > 6:
            lo_, hi_ = rng["lo"], rng["hi"]
            # son 5 barda: aralığın ALTINA kapanış (fitil değil) VE şimdi aralığın İÇİNE kapanış
            closes = ca[-6:]
            below = np.where(closes[:-1] < lo_)[0]
            if below.size and closes[-1] > lo_:
                range_reclaim_up = True
                range_break_lo = float(np.min(la[-6:]))

        # 5) hacim profili (yaklaşık)
        vp = _volume_profile(ca, va)
        near_val = bool(vp and abs(price - vp["val"]) <= 0.35 * atr_abs)
        near_poc = bool(vp and abs(price - vp["poc"]) <= 0.35 * atr_abs)

        # 6) yapı kırılımı (MSS) + emir bloğu
        # MSS anında fiyat bloğun İÇİNDE olmaz (blok geride kalır) — kurulum "MSS oldu, sonra bloğa
        # GERİ DÖNÜLDÜ" der. Bu yüzden MSS son `mss_lookback` barda ARANIR, bloğu hatırlanır.
        hi_idx, lo_idx = _pivots(ha, la, n=2)
        last_swing_hi = float(ha[hi_idx[-1]]) if hi_idx else None
        last_swing_lo = float(la[lo_idx[-1]]) if lo_idx else None
        mss_up = bool(last_swing_hi is not None and price > last_swing_hi and float(c.iloc[-2]) <= last_swing_hi)
        ob_lo = ob_hi = None
        mss_bar = None
        mss_lookback = min(40, n - 3)
        for j in range(n - 1, n - 1 - mss_lookback, -1):
            prior_hi = [ha[k] for k in hi_idx if k < j - 1]
            if not prior_hi:
                continue
            lvl = float(prior_hi[-1])
            if ca[j] > lvl and ca[j - 1] <= lvl:
                mss_bar = j
                break
        if mss_bar is not None:
            # kırılımı üreten atağın öncesindeki SON kırmızı (düşüş) mum = emir bloğu
            for i in range(mss_bar - 1, max(0, mss_bar - 30), -1):
                if ca[i] < oa[i]:
                    ob_lo, ob_hi = float(la[i]), float(ha[i])
                    break
        mss_recent = mss_bar is not None
        in_ob = bool(ob_lo is not None and ob_lo <= price <= ob_hi)

        # 7) stokastik + EMA20 geri kapanışı (Trader DNA)
        k_, d_ = _stoch(h, l, c)
        stoch_cross_up = bool(k_ is not None and d_ is not None and k_ > d_ and k_ < 45.0)
        ema20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
        crossback_up = bool(n > 3 and float(l.iloc[-2]) < ema20 and price > ema20 and float(c.iloc[-2]) < ema20)

        # 8) Bollinger (Expert Para: yalnız DAR bantlı yatay paritede alt bant → orta bant)
        sma20 = c.rolling(20).mean()
        sd20 = c.rolling(20).std(ddof=0)
        bb_lo = float(sma20.iloc[-1] - 2 * sd20.iloc[-1]) if n >= 20 else None
        bb_mid = float(sma20.iloc[-1]) if n >= 20 else None
        bb_w = float(4 * sd20.iloc[-1] / sma20.iloc[-1] * 100.0) if n >= 20 and float(sma20.iloc[-1]) > 0 else None
        at_bb_lower = bool(bb_lo is not None and float(l.iloc[-1]) <= bb_lo and price > bb_lo)

        out = {
            "low_last": float(la[-1]), "low_2bar": float(min(la[-1], la[-2])) if n > 2 else float(la[-1]),
            "fvg_bull": bull_fvg, "fvg_bear": bear_fvg, "in_bull_fvg": in_bull_fvg, "ifvg_up": ifvg_up,
            "manip_bull": manip_bull, "manip_bear": manip_bear, "engulf_bull": engulf_bull,
            "htf_range": rng, "range_reclaim_up": range_reclaim_up, "range_break_lo": range_break_lo,
            "opening_range": orr, "vprofile": vp, "near_val": near_val, "near_poc": near_poc,
            "mss_up": mss_up, "mss_recent": mss_recent, "mss_bar_ago": (None if mss_bar is None else n - 1 - mss_bar),
            "ob_lo": ob_lo, "ob_hi": ob_hi, "in_ob": in_ob,
            "fvg_bear_raw": bear_raw,
            "swing_hi_piv": last_swing_hi, "swing_lo_piv": last_swing_lo,
            "stoch_k": k_, "stoch_d": d_, "stoch_cross_up": stoch_cross_up, "crossback_up": crossback_up,
            "bb_lower": bb_lo, "bb_mid": bb_mid, "bb_width_pct_v": bb_w, "at_bb_lower": at_bb_lower,
        }
        f.update(out)
    except Exception as e:
        # SESSİZ YUTMA YASAK (projenin kendi dersi, 2026-09-04): burada `pass` demek,
        # video kurulumlarının TAMAMININ sessizce kaybolması demekti — dışarıdan
        # "kurulum yok" ile "özellik hesabı patladı" ayırt edilemiyordu. Hata artık
        # özellik sözlüğüne yazılır; kurulumlar yine ateşlenmez ama SEBEBİ görünür.
        f["video_features_error"] = f"{type(e).__name__}: {e}"
    return f


# ---------------------------------------------------------------------------
# TETİKLEYİCİLER
# ---------------------------------------------------------------------------
def fire_video_sleeves(f: Dict, allowed: List[str], p, allow_short: bool = False,
                       now_ts: Optional[float] = None) -> List[Dict]:
    """Video kaynaklı kurulumlar. Yalnız `allowed` listesindekiler; spot'ta SHORT yok."""
    out: List[Dict] = []
    if not f.get("ok"):
        return out
    price = float(f.get("price") or f.get("close") or 0.0)
    if price <= 0:
        return out
    atr = float(f.get("atr_pct") or 0.3)
    atr_abs = max(1e-12, atr / 100.0 * price)
    up = bool(f.get("bar_up"))
    rsi = float(f.get("rsi") or 50.0)
    vr = f.get("vol_ratio")
    kz = killzone_of(now_ts) if now_ts else None
    kz_note = f" · {kz} seansı" if kz else ""
    smult = session_size_mult(now_ts)
    if smult < 1.0:
        kz_note += f" · seans çarpanı ×{smult:g}"

    # 1) FVG dolumu — fiyat boğa boşluğunun içinde, dönüş barı var (Chart Fanatics / Percoco: %50 midpoint)
    z = f.get("fvg_bull")
    if "fvg_fill" in allowed and z and f.get("in_bull_fvg") and up and rsi < 65 and f.get("trend_up"):
        out.append({"kind": "fvg_fill", "direction": "LONG", "size": 0.8, "exit_mode": EXIT_PARTIAL_RUN,
                    "stop_hint": z["lo"] - 0.3 * atr_abs,
                    "note": f"boğa FVG {z['lo']:.6g}-{z['hi']:.6g} içine geri dönüş (yaş {z['age']} bar) + dönüş barı{kz_note}"})

    # 2) inverse FVG — ayı boşluğunun üstüne gövde kapanışı; süpürme ile birlikte daha güçlü
    # Kaynak kurulum "süpürme → boşluk → ters çevirme" zinciri; süpürme YA DA trend teyidi şart
    # (teyitsiz hâli ölçümde gürültüydü). Süpürmeli olan tam boyut, trend teyitlisi yarım.
    if "ifvg_reclaim" in allowed and f.get("ifvg_up") and up:
        zb = f.get("fvg_bear_raw") or {}
        sweep = bool(f.get("swept_low"))
        if sweep or f.get("trend_up"):
            out.append({"kind": "ifvg_reclaim", "direction": "LONG", "size": 0.8 if sweep else 0.5,
                        "exit_mode": EXIT_PARTIAL_RUN,
                        "stop_hint": float(zb.get("lo", price)) - 0.3 * atr_abs,
                        "note": f"ayı boşluğu ters çevrildi (gövde {zb.get('hi', 0):.6g} üstünde kapandı)"
                                + (" · likidite süpürmesiyle birlikte" if sweep else " · trend teyitli") + kz_note})

    # 3) aralık dışına taşıp geri dönüş (yalnız ALT taraf — spot'ta short yok)
    if "range_reclaim" in allowed and f.get("range_reclaim_up") and up:
        rng = f.get("htf_range") or {}
        brk = f.get("range_break_lo")
        out.append({"kind": "range_reclaim", "direction": "LONG", "size": 1.0, "exit_mode": EXIT_FIXED,
                    "stop_hint": (float(brk) - 0.2 * atr_abs) if brk else None,
                    "target_hint": float(rng.get("hi")) if rng.get("hi") else None,
                    "note": f"4h aralığın ({rng.get('lo', 0):.6g}-{rng.get('hi', 0):.6g}) altına kapanış sonrası "
                            f"aralığa geri kapanış — fitil sayılmaz{kz_note}"})

    # 4) manipülasyon mumu — hacim değer alanı kenarında daha güçlü
    if "manipulation_candle" in allowed and f.get("manip_bull") and (vr is None or vr >= 1.0):
        vp = f.get("vprofile") or {}
        at_level = bool(f.get("near_val") or f.get("near_poc"))
        out.append({"kind": "manipulation_candle", "direction": "LONG", "size": 0.9 if at_level else 0.6,
                    "exit_mode": EXIT_PARTIAL_RUN,
                    "stop_hint": float(f.get("low_last") or price) - 0.2 * atr_abs,
                    "note": "önceki dip alındı ve önceki tepenin üstünde kapanış"
                            + (f" · hacim {'değer alanı alt kenarı' if f.get('near_val') else 'POC'} "
                               f"({vp.get('val' if f.get('near_val') else 'poc', 0):.6g})" if at_level else "") + kz_note})

    # 5) açılış aralığı kırılımı + geri test
    orr = f.get("opening_range")
    if "opening_range" in allowed and orr and up and price > orr["hi"] and orr["bars_since"] >= 1 \
            and abs(price - orr["hi"]) <= 0.6 * atr_abs and f.get("trend_up"):
        out.append({"kind": "opening_range", "direction": "LONG", "size": 0.7, "exit_mode": EXIT_DYNAMIC_PEAK,
                    "stop_hint": orr["lo"] if (price - orr["lo"]) / price * 100.0 <= p.max_stop_pct else orr["hi"] - 1.0 * atr_abs,
                    "note": f"açılış aralığı ({orr['lo']:.6g}-{orr['hi']:.6g}) yukarı kırıldı, seviyeye geri test{kz_note}"})

    # 6) EMA dizilimi + yutan mum (Trippa) — EMA20 altına sarkma sonrası yutan boğa mumu
    if "ema_engulf" in allowed and f.get("engulf_bull") and f.get("trend_up") and rsi < 70 \
            and (f.get("dist_ema_pct") is None or float(f.get("dist_ema_pct") or 0) < 0.6 * atr):
        out.append({"kind": "ema_engulf", "direction": "LONG", "size": 0.8, "exit_mode": EXIT_FIXED,
                    "stop_hint": float(f.get("low_2bar") or price) - 0.15 * atr_abs,
                    "note": f"EMA dizilimi yukarı + EMA'ya sarkma sonrası yutan boğa mumu (R/R 1:2 kurulumu){kz_note}"})

    # 7) POC / değer alanı dönüşü
    if "poc_reversion" in allowed and f.get("near_val") and up and rsi < 50:
        vp = f.get("vprofile") or {}
        out.append({"kind": "poc_reversion", "direction": "LONG", "size": 0.7, "exit_mode": EXIT_FIXED,
                    "target_hint": float(vp.get("poc")) if vp.get("poc") else None,
                    "stop_hint": float(vp.get("val", price)) - 0.5 * atr_abs,
                    "note": f"hacim değer alanı alt kenarı ({vp.get('val', 0):.6g}) → POC ({vp.get('poc', 0):.6g}) hedefi{kz_note}"})

    # 8) emir bloğu yeniden testi (MSS sonrası geri dönüş)
    if "order_block" in allowed and f.get("in_ob") and f.get("mss_recent") and up and f.get("trend_up"):
        out.append({"kind": "order_block", "direction": "LONG", "size": 0.7, "exit_mode": EXIT_PARTIAL_RUN,
                    "stop_hint": float(f.get("ob_lo", price)) - 0.3 * atr_abs,
                    "note": f"yapı kırılımı sonrası emir bloğu ({f.get('ob_lo', 0):.6g}-{f.get('ob_hi', 0):.6g}) yeniden test ediliyor{kz_note}"})

    # 9) EMA20 geri kapanışı + stokastik kesişimi
    if "stoch_cross_back" in allowed and f.get("crossback_up") and f.get("stoch_cross_up") and up:
        out.append({"kind": "stoch_cross_back", "direction": "LONG", "size": 0.6, "exit_mode": EXIT_FIXED,
                    "note": f"EMA20'nin altına sarkıp üstüne kapanış + Stoch(8,5,3) yukarı kesişim "
                            f"(K {f.get('stoch_k'):.0f} > D {f.get('stoch_d'):.0f}){kz_note}"})

    # 10) Bollinger alt bandı → orta bant (yalnız DAR bant = yatay parite)
    bw = f.get("bb_width_pct_v")
    if "bb_lower_band" in allowed and f.get("at_bb_lower") and up and bw is not None and bw <= 3.0 * max(atr, 0.05) \
            and not f.get("breakdown"):
        out.append({"kind": "bb_lower_band", "direction": "LONG", "size": 0.6, "exit_mode": EXIT_FIXED,
                    "target_hint": float(f.get("bb_mid")) if f.get("bb_mid") else None,
                    "stop_hint": float(f.get("bb_lower", price)) - 0.6 * atr_abs,
                    "note": f"dar Bollinger bandı (%{bw:.2f}) alt kenarından dönüş → orta bant hedefi{kz_note}"})

    # SEANS ÖNCÜLÜ — ölçülmüş: seans dışında ham kenar anlamlı negatif (t −6,3), NY_AM'de pozitif (t +3,4).
    # Kapatmıyoruz (ölçüm sürsün); boyutu küçültüyoruz. `_session_gate` kendi verisiyle bunu devralır.
    if smult != 1.0:
        for g in out:
            g["size"] = round(float(g.get("size", 1.0)) * smult, 3)
    return out


def describe() -> Dict:
    """Panel: hangi kurulum hangi kanaldan, iddiası ne, kanıtı neydi, ne alınmadı."""
    return {
        "sources": [{"sleeve": k, **v} for k, v in SOURCES.items()],
        "not_implemented": NOT_IMPLEMENTED,
        "killzones": [{"name": n, "utc": f"{a:02d}:00-{b:02d}:00", "size_mult": SESSION_SIZE_MULT.get(n)}
                      for n, a, b in KILLZONES],
        "session_size_mult": {(k or "SEANS_DIŞI"): v for k, v in SESSION_SIZE_MULT.items()},
        "measured": MEASURED["primary"],
        "measured_superseded": MEASURED["superseded_1day"],
        "n_videos": 21, "n_setups": len(SOURCES),
        "verdict": "SHADOW",
        "verdict_tr": ("Ölçüldü: hiçbir kurulum pozitif kenar göstermedi (7.343 aday, t −20,4) → GÖLGE modu: "
                       "sinyal üretilir, emir gönderilmez, ölçüm sürer."),
        "evidence_legend": {
            "BACKTEST_SMALL": "geri test var ama küçük/tek dönem (n≈300)",
            "BACKTEST_TINY": "6-10 işlemlik geri test — sunucunun kendisi 'yetersiz' diyor",
            "MANUAL_BACKTEST_31": "7 günlük elle geriye dönük çizim, 31 işlem, komisyon yok",
            "REPLAY_DEMO": "TradingView replay (simülasyon) gösterimi",
            "SCREENSHOT": "yalnız kâr/zarar ekran görüntüsü",
            "ANECDOTE": "sözlü kişisel beyan",
            "NONE": "kanıt sunulmadı",
        },
        "note": ("Bu kurulumlar İDDİA edilen kazançları değil, anlatılan MEKANİĞİ taşır. Gerçek veride ölçüldüler ve "
                 "hiçbiri pozitif kenar göstermedi; bu yüzden GÖLGEDE çalışırlar (emir yok). Kanıt pozitife dönerse "
                 "kanıt tavanıyla (25 $) canlıya alınırlar — kanıtsız kenar tam boyutla oynanmaz."),
    }
