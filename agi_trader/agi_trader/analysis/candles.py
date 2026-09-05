"""
Mum (candlestick) formasyonları — FAZ 5'in yazılmamış parçası.

FAZ 5 planında `analysis/candles.py` yeni dosya olarak listelenmişti ama hiç
yazılmadı; denetim raporunda faz yine de "tamam" görünüyordu. Bu dosya o boşluğu
kapatır ve aynı ölçüm disiplinine tabidir.

TASARIM KARARLARI (ölçümden önce alınan, gerekçeli):

1. **Her şey ATR cinsinden.** "Uzun gövde" mutlak fiyatla tanımlanamaz; BTC'de
   500 $ küçük, DOGE'de devasadır. Gövde/fitil eşikleri ATR'ye bölünür.

2. **BAĞLAM ZORUNLU.** Çekiç, düşüş trendinin SONUNDA çekiçtir; yükselişin
   ortasında sadece alt fitilli bir mumdur. Dönüş formasyonları öncül trend
   şartı olmadan raporlanmaz. Bu, çift tepe denetiminde öğrenilen dersin
   doğrudan uygulanmasıdır — kapı sonradan eklenince tespit çöküyor, baştan
   tasarıma konması gerekiyor.

3. **Yön "NÖTR" olabilir.** Doji ve spinning top kararsızlıktır; bunları
   LONG/SHORT'a zorlamak uydurma bilgi üretir.

⚠️ Mum formasyonlarının örneklem dışı kâr ürettiğine dair güçlü akademik kanıt
YOKTUR (Marshall-Young-Rose 2006 ve devamı: ABD hisselerinde anlamlı getiri
bulunamadı). Bu modül KARAR DESTEĞİ üretir. Ölçülmüş kanıt `CANDLE_EVIDENCE`
sabitinde tutulur ve panelde formasyonun yanında gösterilir.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Eşikler — ATR katı cinsinden, ölçekten bağımsız
LONG_BODY = 0.60        # gövde ≥ 0,60 ATR → "uzun"
SMALL_BODY = 0.25       # gövde ≤ 0,25 ATR → "küçük"
DOJI_BODY = 0.08        # gövde ≤ 0,08 ATR → doji
LONG_WICK = 1.8         # fitil ≥ gövdenin 1,8 katı → "uzun fitil"
TREND_BARS = 10         # bağlam penceresi
TREND_MIN_ATR = 1.5     # öncül hareket ≥ 1,5 ATR olmalı


@dataclass
class CandlePattern:
    key: str
    name: str
    direction: str          # "LONG" | "SHORT" | "NÖTR"
    family: str             # "dönüş" | "devam" | "kararsızlık"
    i: int                  # bitiş bar indeksi
    bars: int               # kaç mumdan oluşuyor
    strength: float         # 0..1 — şeklin ne kadar belirgin olduğu
    context: str            # öncül trend açıklaması
    note: str
    evidence: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# ===========================================================================
def _atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    s = pd.Series(tr).ewm(alpha=1.0 / n, adjust=False).mean().values
    return np.where(s > 0, s, np.nan)


def _trend(c: np.ndarray, i: int, atr: float, bars: int = 1) -> float:
    """Formasyonun KENDİSİNDEN önceki eğilim, ATR cinsinden. + yükseliş, − düşüş.

    `bars` = formasyonun kaç mumdan oluştuğu. Bu parametre olmadan pencere
    formasyonun kendi mumlarını içine alıyordu ve bağlam ŞARTI DÖNGÜSEL
    oluyordu: "düşüş sonrası boğa yutan" derken kastedilen düşüş, yutan
    formasyonun kendi ilk kırmızı mumuydu. Ölçüldü — bu hatayla trendsiz
    ortamda 16 dönüş formasyonunun 12'si yine de raporlanıyordu."""
    son = i - bars                      # formasyon başlamadan ÖNCEKİ bar
    j = max(0, son - TREND_BARS)
    if son < 0 or son - j < 3 or not math.isfinite(atr) or atr <= 0:
        return 0.0
    return float((c[son] - c[j]) / atr)


def detect_candles(df: pd.DataFrame, lookback: int = 3) -> List[Dict]:
    """Son `lookback` bar içinde biten mum formasyonlarını döndürür.

    lookback=1 → yalnız son mum. Formasyon 2-3 mumdan oluşuyorsa `i` SON
    mumunun indeksidir."""
    need = {"open", "high", "low", "close"}
    if df is None or len(df) < TREND_BARS + 6 or not need.issubset(df.columns):
        return []

    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    A = _atr(df)
    n = len(df)
    out: List[CandlePattern] = []

    for i in range(max(TREND_BARS + 3, n - lookback), n):
        a = A[i]
        if not math.isfinite(a) or a <= 0:
            continue
        body = (c[i] - o[i]) / a
        rng = (h[i] - l[i]) / a
        up_w = (h[i] - max(c[i], o[i])) / a
        dn_w = (min(c[i], o[i]) - l[i]) / a
        ab = abs(body)
        # Bağlam formasyon UZUNLUĞUNA göre ayrı ayrı ölçülür (bkz. `_trend`).
        T = {k: _trend(c, i, a, k) for k in (1, 2, 3)}
        yuk = {k: T[k] >= TREND_MIN_ATR for k in T}
        dus = {k: T[k] <= -TREND_MIN_ATR for k in T}
        # tek-mumluk kısayollar (aşağıdaki tek mum bloğu için)
        yukselis, dusus, tr = yuk[1], dus[1], T[1]

        def ekle(key, name, yon, fam, bars, guc, note):
            t = T.get(bars, T[1])
            ctx = (f"öncesinde {abs(t):.1f} ATR "
                   f"{'yükseliş' if t > 0 else 'düşüş' if t < 0 else 'yatay'}")
            out.append(CandlePattern(key=key, name=name, direction=yon, family=fam,
                                     i=int(i), bars=bars,
                                     strength=round(float(min(1.0, max(0.0, guc))), 3),
                                     context=ctx, note=note))

        # ---------------- TEK MUM ----------------
        # Çekiç / Asılı Adam — küçük gövde, uzun ALT fitil, kısa üst fitil
        if ab <= SMALL_BODY * 2 and dn_w >= LONG_WICK * max(ab, 0.05) and up_w <= ab * 1.2:
            guc = min(1.0, dn_w / 1.5)
            if dusus:
                ekle("hammer", "Çekiç", "LONG", "dönüş", 1, guc,
                     "Düşüş sonunda uzun alt fitil: satıcılar dibi test etti ama "
                     "kapanış yukarı toparlandı.")
            elif yukselis:
                ekle("hanging_man", "Asılı Adam", "SHORT", "dönüş", 1, guc,
                     "Yükseliş sonunda uzun alt fitil: gün içi satış baskısı belirdi.")

        # Ters Çekiç / Kayan Yıldız — uzun ÜST fitil
        if ab <= SMALL_BODY * 2 and up_w >= LONG_WICK * max(ab, 0.05) and dn_w <= ab * 1.2:
            guc = min(1.0, up_w / 1.5)
            if dusus:
                ekle("inverted_hammer", "Ters Çekiç", "LONG", "dönüş", 1, guc,
                     "Düşüş sonunda uzun üst fitil: alıcılar ilk kez yukarı denedi.")
            elif yukselis:
                ekle("shooting_star", "Kayan Yıldız", "SHORT", "dönüş", 1, guc,
                     "Yükseliş sonunda uzun üst fitil: zirve reddedildi.")

        # Doji ailesi — kararsızlık; YÖN VERİLMEZ
        if ab <= DOJI_BODY and rng >= 0.4:
            if dn_w >= 2 * up_w and dn_w >= 0.5:
                ekle("dragonfly_doji", "Yusufçuk Doji", "NÖTR", "kararsızlık", 1,
                     min(1.0, dn_w), "Açılış=kapanış, uzun alt fitil — dip testi.")
            elif up_w >= 2 * dn_w and up_w >= 0.5:
                ekle("gravestone_doji", "Mezar Taşı Doji", "NÖTR", "kararsızlık", 1,
                     min(1.0, up_w), "Açılış=kapanış, uzun üst fitil — zirve testi.")
            elif rng >= 1.2:
                ekle("long_legged_doji", "Uzun Bacaklı Doji", "NÖTR", "kararsızlık", 1,
                     min(1.0, rng / 2), "İki yönde de geniş salınım, kapanış başa döndü.")
            else:
                ekle("doji", "Doji", "NÖTR", "kararsızlık", 1, 0.5,
                     "Açılış ≈ kapanış — kararsızlık.")

        # Marubozu — neredeyse fitilsiz uzun gövde (DEVAM)
        if ab >= LONG_BODY and up_w <= 0.08 * rng and dn_w <= 0.08 * rng:
            ekle("marubozu", "Marubozu", "LONG" if body > 0 else "SHORT", "devam", 1,
                 min(1.0, ab / 1.5),
                 "Fitilsiz uzun gövde: tek taraf bütün seansa hâkim oldu.")

        # Spinning top — küçük gövde, iki yanda fitil
        if ab <= SMALL_BODY and up_w >= ab and dn_w >= ab and rng >= 0.5:
            ekle("spinning_top", "Topaç", "NÖTR", "kararsızlık", 1, 0.4,
                 "Küçük gövde, iki yanda fitil — denge.")

        # ---------------- İKİ MUM ----------------
        if i >= 1:
            pb = (c[i - 1] - o[i - 1]) / a
            pab = abs(pb)
            # Yutan
            if ab >= LONG_BODY and pab >= 0.05:
                if body > 0 and pb < 0 and c[i] >= o[i - 1] and o[i] <= c[i - 1] and dus[2]:
                    ekle("bullish_engulfing", "Boğa Yutan", "LONG", "dönüş", 2,
                         min(1.0, ab / max(pab, 0.1) / 2),
                         "Yeşil gövde bir önceki kırmızı gövdeyi tamamen içine aldı.")
                if body < 0 and pb > 0 and c[i] <= o[i - 1] and o[i] >= c[i - 1] and yuk[2]:
                    ekle("bearish_engulfing", "Ayı Yutan", "SHORT", "dönüş", 2,
                         min(1.0, ab / max(pab, 0.1) / 2),
                         "Kırmızı gövde bir önceki yeşil gövdeyi tamamen içine aldı.")
            # Harami — küçük gövde öncekinin İÇİNDE
            if pab >= LONG_BODY and ab <= pab * 0.5:
                ic = (max(c[i], o[i]) <= max(c[i - 1], o[i - 1]) and
                      min(c[i], o[i]) >= min(c[i - 1], o[i - 1]))
                if ic and pb < 0 and dus[2]:
                    ekle("bullish_harami", "Boğa Harami", "LONG", "dönüş", 2,
                         min(1.0, pab / 1.5), "Uzun kırmızı mumun içinde küçük gövde: satış hızı kesildi.")
                if ic and pb > 0 and yuk[2]:
                    ekle("bearish_harami", "Ayı Harami", "SHORT", "dönüş", 2,
                         min(1.0, pab / 1.5), "Uzun yeşil mumun içinde küçük gövde: alış hızı kesildi.")
            # Delen Çizgi / Kara Bulut
            if pab >= LONG_BODY and ab >= LONG_BODY * 0.7:
                orta = (o[i - 1] + c[i - 1]) / 2
                if body > 0 and pb < 0 and o[i] < l[i - 1] and c[i] > orta and c[i] < o[i - 1] and dus[2]:
                    ekle("piercing", "Delen Çizgi", "LONG", "dönüş", 2, 0.7,
                         "Boşlukla açıp önceki kırmızı gövdenin yarısını geri aldı.")
                if body < 0 and pb > 0 and o[i] > h[i - 1] and c[i] < orta and c[i] > o[i - 1] and yuk[2]:
                    ekle("dark_cloud", "Kara Bulut", "SHORT", "dönüş", 2, 0.7,
                         "Boşlukla açıp önceki yeşil gövdenin yarısını geri verdi.")
            # Cımbız
            if abs(l[i] - l[i - 1]) <= 0.08 * a and dus[2] and body > 0:
                ekle("tweezer_bottom", "Cımbız Dip", "LONG", "dönüş", 2, 0.55,
                     "İki mum aynı dibi test etti ve tutundu.")
            if abs(h[i] - h[i - 1]) <= 0.08 * a and yuk[2] and body < 0:
                ekle("tweezer_top", "Cımbız Tepe", "SHORT", "dönüş", 2, 0.55,
                     "İki mum aynı zirveyi test etti ve reddedildi.")

        # ---------------- ÜÇ MUM ----------------
        if i >= 2:
            b1 = (c[i - 2] - o[i - 2]) / a
            b2 = (c[i - 1] - o[i - 1]) / a
            # Sabah / Akşam Yıldızı
            if abs(b1) >= LONG_BODY and abs(b2) <= SMALL_BODY and ab >= LONG_BODY:
                if b1 < 0 and body > 0 and c[i] > (o[i - 2] + c[i - 2]) / 2 and dus[3]:
                    ekle("morning_star", "Sabah Yıldızı", "LONG", "dönüş", 3,
                         min(1.0, ab / 1.2),
                         "Uzun düşüş → kararsız mum → uzun yükseliş: dönüş üçlüsü.")
                if b1 > 0 and body < 0 and c[i] < (o[i - 2] + c[i - 2]) / 2 and yuk[3]:
                    ekle("evening_star", "Akşam Yıldızı", "SHORT", "dönüş", 3,
                         min(1.0, ab / 1.2),
                         "Uzun yükseliş → kararsız mum → uzun düşüş: dönüş üçlüsü.")
            # Üç Beyaz Asker / Üç Siyah Karga
            if b1 >= SMALL_BODY and b2 >= SMALL_BODY and body >= SMALL_BODY and \
               c[i] > c[i - 1] > c[i - 2] and o[i] > o[i - 1] > o[i - 2]:
                ekle("three_white_soldiers", "Üç Beyaz Asker", "LONG", "devam", 3,
                     min(1.0, (b1 + b2 + body) / 2), "Üst üste üç güçlü yeşil mum.")
            if b1 <= -SMALL_BODY and b2 <= -SMALL_BODY and body <= -SMALL_BODY and \
               c[i] < c[i - 1] < c[i - 2] and o[i] < o[i - 1] < o[i - 2]:
                ekle("three_black_crows", "Üç Siyah Karga", "SHORT", "devam", 3,
                     min(1.0, abs(b1 + b2 + body) / 2), "Üst üste üç güçlü kırmızı mum.")
            # Üç İçeride Yukarı/Aşağı (harami + onay)
            if abs(b1) >= LONG_BODY and abs(b2) <= abs(b1) * 0.5:
                if b1 < 0 and body > 0 and c[i] > max(o[i - 1], c[i - 1]) and dus[3]:
                    ekle("three_inside_up", "Üç İçeride Yukarı", "LONG", "dönüş", 3, 0.65,
                         "Harami sonrası ONAY mumu geldi.")
                if b1 > 0 and body < 0 and c[i] < min(o[i - 1], c[i - 1]) and yuk[3]:
                    ekle("three_inside_down", "Üç İçeride Aşağı", "SHORT", "dönüş", 3, 0.65,
                         "Harami sonrası ONAY mumu geldi.")

    # aynı bar + aynı anahtar tekrarını at, güce göre sırala
    best: Dict[str, CandlePattern] = {}
    for p in out:
        k = f"{p.key}|{p.i}"
        if k not in best or p.strength > best[k].strength:
            best[k] = p
    ranked = sorted(best.values(), key=lambda x: (-x.i, -x.strength))
    for p in ranked:
        p.evidence = CANDLE_EVIDENCE.get(p.key)
    return [p.to_dict() for p in ranked]


def candle_summary(pats: List[Dict]) -> Dict:
    """Panel rozeti — yönlü sayım. Kararsızlık aileleri sayılmaz."""
    yon = [p for p in pats if p["direction"] in ("LONG", "SHORT")]
    lo = sum(1 for p in yon if p["direction"] == "LONG")
    sh = sum(1 for p in yon if p["direction"] == "SHORT")
    notr = sum(1 for p in pats if p["direction"] == "NÖTR")
    bias = "YUKARI" if lo > sh else "AŞAĞI" if sh > lo else "NÖTR"
    return {"n": len(pats), "long": lo, "short": sh, "notr": notr, "bias": bias}


# ---------------------------------------------------------------------------
# ÖLÇÜLMÜŞ KANIT
# ---------------------------------------------------------------------------
# Olay çalışması: 10 bar ileri getiri, 10 parite × {4h, 1d}, 3.300 kontrol
# gözlemi. Kontrol eşdeğeri = aynı yön dağılımıyla kontrolden beklenen getiri
# (yani "SHORT formasyon" kontrolün NEGATİFİYLE karşılaştırılır).
#
# SONUÇ: 9 ölçülebilir formasyonun 3'ü kontrolden iyi, 6'sı kötü. |t| ≥ 2 olan
# İKİ formasyon var ve İKİSİ DE TERS yönde anlamlı:
#   Üç Beyaz Asker (yükseliş bekler) → yön-düzeltilmiş −%3,45 (kontrol −%1,42), t=−2,12
#   Üç Siyah Karga (düşüş bekler)    → yön-düzeltilmiş −%0,31 (kontrol  +%1,42), t=−2,26
#
# Not: 9 formasyon sınandı; çoklu karşılaştırmada şansla ~0,5 anlamlı sonuç
# beklenir. İkisinin de AYNI (ters) yönde çıkması şans açıklamasını zayıflatır
# ama örneklem küçük (formasyon başına 25-80 olay) ve pencereler örtüşüyor.
#
# Literatürle uyumlu: Marshall-Young-Rose (2006) ABD hisselerinde mum
# formasyonlarında anlamlı getiri bulamamıştı.
CANDLE_MEASURED = {
    "control_ret_pct": -1.42, "fwd_bars": 10, "n_control": 3300,
    "n_measurable": 9, "better_than_control": 3, "significant": 2,
    "significant_wrong_way": 2,
    "verdict": ("Yön üstünlüğü ÖLÇÜLDÜ ve bulunamadı. Anlamlı çıkan iki "
                "formasyonun ikisi de TERS yönde. Mum formasyonları grafik "
                "okumaya yardımcı bir ANNOTASYONDUR; işlem gerekçesi değildir."),
}

CANDLE_EVIDENCE: Dict[str, Dict] = {
    "three_white_soldiers": {"tested": True, "edge": "ters", "n": 42,
        "ret_pct": -3.45, "control_pct": -1.42, "t": -2.12,
        "note": "Yükseliş bekler; ölçümde fiyat kontrolden DAHA ÇOK düştü (t=−2,12)."},
    "three_black_crows": {"tested": True, "edge": "ters", "n": 80,
        "ret_pct": -0.31, "control_pct": 1.42, "t": -2.26,
        "note": "Düşüş bekler; ölçümde düşüş kontrolün ALTINDA kaldı (t=−2,26)."},
    "hammer": {"tested": True, "edge": "yok", "n": 64,
        "ret_pct": -0.75, "control_pct": -1.42, "t": 0.68,
        "note": "Kontrolden hafif iyi ama anlamsız (t=0,68)."},
    "tweezer_bottom": {"tested": True, "edge": "yok", "n": 52,
        "ret_pct": -1.01, "control_pct": -1.42, "t": 0.38,
        "note": "Kontrolden hafif iyi ama anlamsız (t=0,38)."},
    "tweezer_top": {"tested": True, "edge": "yok", "n": 32,
        "ret_pct": 1.76, "control_pct": 1.42, "t": 0.51,
        "note": "Kontrolden hafif iyi ama anlamsız (t=0,51)."},
    "marubozu": {"tested": True, "edge": "yok", "n": 42,
        "ret_pct": -1.11, "control_pct": 0.20, "t": -1.07,
        "note": "Devam beklenir; ölçümde kontrolün altında, anlamsız."},
    "shooting_star": {"tested": True, "edge": "yok", "n": 35,
        "ret_pct": 0.89, "control_pct": 1.42, "t": -0.51,
        "note": "Kontrolün altında, anlamsız."},
    "bullish_engulfing": {"tested": True, "edge": "yok", "n": 28,
        "ret_pct": -2.17, "control_pct": -1.42, "t": -0.55,
        "note": "Kontrolün altında, anlamsız."},
    "inverted_hammer": {"tested": True, "edge": "yok", "n": 28,
        "ret_pct": -1.97, "control_pct": -1.42, "t": -0.42,
        "note": "Kontrolün altında, anlamsız."},
}
