"""
Olay çalışması — hangi X hesabı GERÇEKTEN fiyat hareket ettiriyor? (FAZ 6)

SORUN: `accounts.py` yüzlerce hesabı ELLE YAZILMIŞ `weight` (0-10) değerleriyle
taşıyor. Bu ağırlıklar bir varsayımdır; hiçbiri ölçülmedi. Ölçülmemiş bir ağırlık,
modeli o hesaba göre eğer ve sahte güven üretir.

Bu modül ağırlıkları VARSAYMAZ, ÖLÇER:

  Bir hesap t anında gönderi attıysa, t+5dk / t+1s / t+4s / t+24s ufuklarında
  ANORMAL getiri nedir? ("anormal" = varlığın kendi normal hareketinden fazlası)

İKİ AYRI SORU — mover modülündeki dersin aynısı:
  1. BÜYÜKLÜK : gönderi sonrası oynaklık artıyor mu?   (genelde ölçülebilir)
  2. YÖN      : gönderinin duygusu yönü öngörüyor mu?  (genelde ölçülemez)
İkisi ayrı raporlanır; birini diğerinin kanıtı saymak yanıltıcıdır.

İSTATİSTİKSEL KAPI: bir hesabın skoru ancak `n ≥ MIN_EVENTS` gözlem VE `|t| ≥ 2`
ise "ölçüldü" sayılır. Aksi hâlde `measured=False` döner ve ağırlık ALMAZ —
"veri yok" ile "etkisi yok" ayrı şeylerdir ve panelde ayrı gösterilir.

VERİ GERÇEĞİ: X API Basic (200 $/ay) GEÇMİŞE dönük arama VERMEZ; yalnız ileriye
dönük toplama mümkündür (bkz. `collector.py`). Bu yüzden skorlar ancak veri
biriktikçe anlamlı olur. Sistem bunu gizlemez: yeterli gözlem yoksa "ölçülmedi"
yazar.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ölçüm ufukları (dakika)
HORIZONS = {"5m": 5, "1h": 60, "4h": 240, "24h": 1440}
MIN_EVENTS = 20             # bu sayının altında istatistik yapılmaz
T_THRESHOLD = 2.0           # |t| ≥ 2 → %95 anlamlılık (kabaca)


@dataclass
class HorizonResult:
    horizon: str
    n: int
    mean_abnormal_pct: float        # ortalama anormal getiri (%)
    mean_abs_abnormal_pct: float    # ortalama MUTLAK anormal getiri (oynaklık etkisi)
    t_stat: float                   # yön anlamlılığı
    t_stat_abs: float               # büyüklük anlamlılığı
    hit_rate: Optional[float]       # duygu yönü tuttu mu (duygu varsa)
    significant_direction: bool
    significant_magnitude: bool


@dataclass
class AccountScore:
    handle: str
    n_events: int
    measured: bool                  # istatistiksel kapıyı geçti mi
    impact_score: float             # 0..10 — ölçülmüş etki (ölçülmediyse 0)
    prior_weight: Optional[float]   # accounts.py'deki ELLE YAZILMIŞ değer
    best_horizon: Optional[str]
    horizons: List[Dict] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


# ===========================================================================
# Anormal getiri
# ===========================================================================
def abnormal_returns(prices: pd.Series, event_ts: pd.Timestamp,
                     horizon_min: int, baseline_min: int = 1440
                     ) -> Optional[Tuple[float, float]]:
    """(anormal getiri %, beklenen oynaklık %) — olaydan `horizon_min` sonra.

    "Anormal" = gerçekleşen getiri eksi, olaydan ÖNCEKİ `baseline_min` dakikada
    ölçülen normal sürüklenme. Piyasa geneli hareketini çıkarmak için `market`
    serisi verilebilir (bkz. `run_event_study`)."""
    if prices is None or len(prices) < 10:
        return None
    idx = prices.index
    try:
        i0 = idx.searchsorted(event_ts, side="left")
    except Exception:
        return None
    if i0 >= len(idx) - 1 or i0 == 0:
        return None

    t1 = event_ts + pd.Timedelta(minutes=horizon_min)
    i1 = idx.searchsorted(t1, side="left")
    if i1 >= len(idx):
        return None
    p0, p1 = float(prices.iloc[i0]), float(prices.iloc[i1])
    if p0 <= 0:
        return None
    realized = (p1 / p0 - 1.0) * 100.0

    # olaydan önceki pencerede aynı ufkun normal sürüklenmesi
    t_pre = event_ts - pd.Timedelta(minutes=baseline_min)
    i_pre = idx.searchsorted(t_pre, side="left")
    if i_pre >= i0 - 2:
        drift = 0.0
    else:
        pre = prices.iloc[i_pre:i0]
        r = pre.pct_change().dropna()
        if len(r) < 3:
            drift = 0.0
        else:
            # dakika başına ortalama sürüklenme × ufuk
            per_min = float(r.mean()) * (len(r) / max(1.0, baseline_min))
            drift = per_min * horizon_min * 100.0
    return realized - drift, realized


def _t_stat(x: Sequence[float]) -> float:
    """Ortalamanın SIFIRDAN farkı. Yön testi için doğrudur (etki yoksa ortalama 0)."""
    a = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(a) < 3:
        return 0.0
    sd = a.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(a.mean() / (sd / math.sqrt(len(a))))


def _welch_t(x: Sequence[float], y: Sequence[float]) -> float:
    """İki bağımsız örneklem ortalaması farkı (Welch). BÜYÜKLÜK testi için."""
    a = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in y if np.isfinite(v)], dtype=float)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    den = math.sqrt(va + vb)
    if den < 1e-15:
        return 0.0
    return float((a.mean() - b.mean()) / den)


def _control_abs_moves(prices: pd.Series, horizon_min: int, n: int = 200,
                       seed: int = 0) -> List[float]:
    """Rastgele KONTROL zamanlarında |anormal getiri| — karşılaştırma tabanı.

    NEDEN ZORUNLU: büyüklük etkisini `|getiri| > 0` diye test etmek anlamsızdır;
    mutlak değer daima pozitiftir ve her hesap "anlamlı" çıkar (ölçüldü: saf
    rastgele seride skor 6,6). Doğru soru: 'bu hesabın gönderilerinden sonraki
    hareket, TİPİK bir andakinden BÜYÜK MÜ?'"""
    if prices is None or len(prices) < 50:
        return []
    rng = np.random.default_rng(seed)
    idx = prices.index
    lo, hi = int(len(idx) * 0.05), int(len(idx) * 0.95)
    if hi - lo < 10:
        return []
    out: List[float] = []
    for i in rng.integers(lo, hi, size=min(n, hi - lo)):
        r = abnormal_returns(prices, idx[int(i)], horizon_min)
        if r is not None:
            out.append(abs(r[0]))
    return out


# ===========================================================================
# Olay çalışması
# ===========================================================================
def run_event_study(events: pd.DataFrame, price_panel: Dict[str, pd.Series],
                    market: Optional[pd.Series] = None,
                    priors: Optional[Dict[str, float]] = None,
                    min_events: int = MIN_EVENTS) -> Dict:
    """Hesap başına ölçülmüş etki skoru.

    events : DataFrame[ts, handle, asset, sentiment]  (sentiment: -1..+1 veya NaN)
    price_panel : {asset: dakikalık/saatlik fiyat serisi (DatetimeIndex)}
    market : opsiyonel piyasa endeksi (ör. BTC) — sistematik hareket çıkarılır
    priors : accounts.py'deki elle yazılmış ağırlıklar (yalnız KARŞILAŞTIRMA için)
    """
    if events is None or len(events) == 0:
        return {"accounts": [], "n_events": 0,
                "note": "hiç olay kaydı yok — collector henüz veri toplamadı"}

    ev = events.copy()
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True, errors="coerce")
    ev = ev.dropna(subset=["ts", "handle"])
    priors = priors or {}

    out: List[AccountScore] = []
    for handle, grp in ev.groupby("handle"):
        per_h: List[HorizonResult] = []
        for hname, hmin in HORIZONS.items():
            ab, sig, used_assets = [], [], set()
            for _, row in grp.iterrows():
                asset = str(row.get("asset") or "").upper()
                px = price_panel.get(asset)
                if px is None:
                    continue
                r = abnormal_returns(px, row["ts"], hmin)
                if r is None:
                    continue
                a = r[0]
                # piyasa hareketini çıkar (sistematik değil, hesaba özgü etki)
                if market is not None:
                    m = abnormal_returns(market, row["ts"], hmin)
                    if m is not None:
                        a = a - m[0]
                ab.append(a)
                used_assets.add(asset)
                s = row.get("sentiment")
                sig.append(float(s) if s is not None and np.isfinite(s) else np.nan)

            if len(ab) < 3:
                continue
            arr = np.asarray(ab, dtype=float)
            t_dir = _t_stat(arr)

            # BÜYÜKLÜK: olay penceresindeki |hareket| ile RASTGELE anlardaki
            # |hareket| karşılaştırılır (bkz. _control_abs_moves).
            ctrl: List[float] = []
            for asset in used_assets:
                ctrl += _control_abs_moves(price_panel.get(asset), hmin)
            t_mag = _welch_t(np.abs(arr), ctrl) if ctrl else 0.0

            sg = np.asarray(sig, dtype=float)
            mask = np.isfinite(sg) & (np.abs(sg) > 0.05)
            hit = (float(np.mean(np.sign(sg[mask]) == np.sign(arr[mask])))
                   if mask.sum() >= 5 else None)

            per_h.append(HorizonResult(
                horizon=hname, n=len(arr),
                mean_abnormal_pct=round(float(arr.mean()), 4),
                mean_abs_abnormal_pct=round(float(np.abs(arr).mean()), 4),
                t_stat=round(t_dir, 3), t_stat_abs=round(t_mag, 3),
                hit_rate=(round(hit, 3) if hit is not None else None),
                significant_direction=bool(abs(t_dir) >= T_THRESHOLD and len(arr) >= min_events),
                significant_magnitude=bool(t_mag >= T_THRESHOLD and len(arr) >= min_events)))

        n_ev = int(len(grp))
        measured = any(h.significant_magnitude or h.significant_direction for h in per_h)
        best = max(per_h, key=lambda h: h.mean_abs_abnormal_pct) if per_h else None
        # skor: yalnız ANLAMLI çıkan ufuklardan hesaplanır; en güçlü anlamlı
        # ufuktaki mutlak anormal getiri 0-10 ölçeğine taşınır.
        sig_h = [h for h in per_h if h.significant_magnitude or h.significant_direction]
        best_sig = max(sig_h, key=lambda h: h.mean_abs_abnormal_pct) if sig_h else None
        score = 0.0
        if measured and best_sig:
            score = float(np.clip(best_sig.mean_abs_abnormal_pct * 4.0, 0, 10))
            best = best_sig

        if n_ev < min_events:
            note = (f"ÖLÇÜLMEDİ — {n_ev}/{min_events} olay. "
                    f"Veri biriktikçe otomatik hesaplanacak.")
        elif not measured:
            note = ("Ölçüldü ama ANLAMLI ETKİ YOK (|t| < 2). Bu hesap ağırlık almaz — "
                    "elle yazılmış öncül değeri kullanılmaz.")
        else:
            note = (f"Ölçülmüş etki: {best.horizon} ufkunda ortalama mutlak anormal "
                    f"getiri %{best.mean_abs_abnormal_pct:.3f} (t={best.t_stat_abs:.1f}).")

        out.append(AccountScore(
            handle=str(handle), n_events=n_ev, measured=bool(measured),
            impact_score=round(score, 3),
            prior_weight=priors.get(str(handle)),
            best_horizon=(best.horizon if (measured and best) else None),
            horizons=[asdict(h) for h in per_h], note=note))

    out.sort(key=lambda a: (-a.impact_score, -a.n_events))
    n_measured = sum(1 for a in out if a.measured)
    return {
        "accounts": [a.to_dict() for a in out],
        "n_events": int(len(ev)),
        "n_accounts": len(out),
        "n_measured": n_measured,
        "min_events": min_events,
        "note": (f"{n_measured}/{len(out)} hesap istatistiksel kapıyı geçti. "
                 f"Geçemeyenler ağırlık ALMAZ."),
        "direction_warning": (
            "Yön (duygu → fiyat yönü) ve büyüklük (gönderi → oynaklık) AYRI "
            "ölçülür. Büyüklük anlamlı çıkan bir hesap, yön öngördüğü anlamına "
            "GELMEZ."),
    }


# ===========================================================================
# accounts.py ağırlıklarını ölçümle değiştirme
# ===========================================================================
def effective_weights(study: Dict, fallback_to_prior: bool = False) -> Dict[str, float]:
    """Motorun kullanacağı NİHAİ ağırlıklar.

    fallback_to_prior=False (VARSAYILAN): ölçülmemiş hesap 0 ağırlık alır.
    Bu kasıtlıdır — ölçülmemiş bir varsayımı modele sokmak, tam olarak bu
    projede daha önce sahte sonuç üreten şeydir.

    fallback_to_prior=True: veri birikene kadar elle yazılmış öncüller kullanılır;
    yalnız açıkça istenirse ve panelde 'ÖLÇÜLMEMİŞ' etiketiyle."""
    w: Dict[str, float] = {}
    for a in study.get("accounts", []):
        if a["measured"]:
            w[a["handle"]] = float(a["impact_score"])
        elif fallback_to_prior and a.get("prior_weight") is not None:
            w[a["handle"]] = float(a["prior_weight"])
    return w


def save_scores(study: Dict, output_dir: str = "runs") -> Path:
    p = Path(output_dir)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    p.mkdir(parents=True, exist_ok=True)
    f = p / "account_scores.json"
    f.write_text(json.dumps(study, indent=2, ensure_ascii=False, default=str),
                 encoding="utf-8")
    return f


def load_scores(output_dir: str = "runs") -> Dict:
    p = Path(output_dir)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    f = p / "account_scores.json"
    if not f.exists():
        return {"accounts": [], "n_events": 0, "note": "henüz ölçüm yapılmadı"}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {"accounts": [], "n_events": 0, "note": "skor dosyası okunamadı"}
