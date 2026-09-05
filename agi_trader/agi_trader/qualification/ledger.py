"""Değişmez tahmin defteri ve gerçekleşme değerlendirici — şartname 44, 45,
46, 103, 104, 105, 106, 107 (+ 2. mesaj 41, 42, 43).

DEĞİŞMEZLİK NEDEN ŞART
Bir tahmin yayımlandıktan sonra değiştirilebiliyorsa, sistemin geçmiş
performansı ölçülemez — yalnız anlatılabilir. Bu dosya EKLEMELİ (append-only)
bir JSONL tutar: kayıtlar güncellenmez, sonuç AYRI bir satır olarak yazılır ve
`prediction_id` ile eşleşir.

BAŞARISIZ SİNYALLER SİLİNMEZ (şartname 106)
Yalnız kazananları göstermek yasaktır. `scorecard` paydayı YAYIMLANAN sinyal
sayısı alır; başarısızları paydadan düşürmek Net1PercentPrecision'ı sahte
yükseltir.

SONUÇ KÜMESİ (2. mesaj 42) — altı sonuçtan biri
  TP_FIRST · SL_FIRST · TIMEOUT · EXPIRED_BEFORE_ENTRY · NOT_FILLED · INVALIDATED
Son üçü "işlem olmadı" demektir ve fiyat performansına DEĞİL, yürütme
performansına yazılır. İkisini karıştırmak sinyal kalitesini olduğundan iyi
gösterir.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

OUTCOMES = ("TP_FIRST", "SL_FIRST", "TIMEOUT",
            "EXPIRED_BEFORE_ENTRY", "NOT_FILLED", "INVALIDATED")
PRICE_OUTCOMES = ("TP_FIRST", "SL_FIRST", "TIMEOUT")

# ── KAYIT KAYNAKLARI ──────────────────────────────────────────────────────
# YAYIMLANAN ile GÖLGE aynı deftere yazılır ama ASLA aynı paydada toplanmaz.
#
# Neden gölge kayıt var: bu sistem şu ana kadar sıfır QUALIFIED fırsat üretti,
# dolayısıyla defter boştu — ve defter boş olduğu için sistemin kendi ilan
# ettiği ana metriği (Gerçekleşen ÷ Tahmin Edilen Net EV) ile kalibrasyon
# tablosu SONSUZA DEK "örneklem yetersiz" diyordu. Ölçmeyen bir ölçüm sistemi
# kendi hatasını göremez.
#
# Gölge kayıt bir sinyal DEĞİLDİR: nitelendirme iddiası taşımaz, işlem
# önermez, karneye girmez. Yalnız "tahmin ettiğim olasılık gerçekleşiyor mu"
# sorusunu canlı veriyle sınar.
SOURCE_PUBLISHED = "scanner"
SOURCE_SHADOW = "shadow"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Prediction:
    """Yayımlanan bir fırsatın DEĞİŞMEZ kaydı."""
    prediction_id: str
    timestamp: str
    symbol: str
    horizon: str
    direction: str
    status: str
    entry: float
    net1_exit: float
    stop: float
    p_target_first: Optional[float]
    p_target_lower95: Optional[float]
    p_stop_first: Optional[float]
    p_timeout: Optional[float]
    baseline: Optional[float]
    required_lift: Optional[float]
    actual_lift: Optional[float]
    robust_ev: Optional[float]
    expected_target_hours: Optional[float]
    cost_model: str
    cost_pct: Optional[float]
    max_capacity_usd: Optional[float]
    data_quality: Optional[float]
    model_version: str
    features_hash: str
    valid_until: str
    source: str = "scanner"
    guaranteed: bool = False          # ŞEMA DEĞİŞMEZİ: her zaman False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Outcome:
    prediction_id: str
    resolved_at: str
    outcome: str
    bars_to_resolution: Optional[int] = None
    hours_to_resolution: Optional[float] = None
    realized_mfe_pct: Optional[float] = None
    realized_mae_pct: Optional[float] = None
    entry_vwap: Optional[float] = None
    exit_vwap: Optional[float] = None
    realized_cost_pct: Optional[float] = None
    realized_net_pct: Optional[float] = None
    predicted_p: Optional[float] = None
    prediction_error: Optional[float] = None
    note: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def make_id(symbol: str, horizon: str, direction: str, ts: Optional[str] = None,
            salt: str = "") -> str:
    """Deterministik kimlik. Python `hash()` KULLANILMAZ — süreçler arası
    tuzlanır ve tekrarlanamaz kimlik üretir (şartname 84)."""
    ham = f"{symbol}|{horizon}|{direction}|{ts or _now()}|{salt}"
    return hashlib.sha256(ham.encode()).hexdigest()[:20]


class Ledger:
    """Ekleme-yalnız defter. Satırlar ASLA güncellenmez veya silinmez."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, kind: str, payload: Dict) -> None:
        rec = {"kind": kind, "written_at": _now(), **payload}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def record_prediction(self, p: Prediction) -> str:
        if p.guaranteed:                       # savunma: şema ihlali
            raise ValueError("guaranteed=True yazılamaz — bu sistemde garanti yok")
        self._append("prediction", p.to_dict())
        return p.prediction_id

    def record_outcome(self, o: Outcome) -> None:
        if o.outcome not in OUTCOMES:
            raise ValueError(f"bilinmeyen sonuç: {o.outcome}")
        self._append("outcome", o.to_dict())

    def read(self) -> List[Dict]:
        if not self.path.exists():
            return []
        out: List[Dict] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    def predictions(self, source: Optional[str] = None) -> List[Dict]:
        """`source=None` HEPSİ; filtrelemek çağıranın sorumluluğudur.

        ⚠️ Karne (`scorecard`) yalnız YAYIMLANANLARI saymalıdır — gölge kayıt
        paydaya girerse Net1PercentPrecision sulanır ve §106 ihlal edilir."""
        ps = [r for r in self.read() if r.get("kind") == "prediction"]
        if source is None:
            return ps
        return [p for p in ps
                if (p.get("source") or SOURCE_PUBLISHED) == source]

    def outcomes(self) -> Dict[str, Dict]:
        """prediction_id → İLK yazılan sonuç.

        Sonradan yazılan ikinci bir sonuç kaydı ilkini EZMEZ; ilk karar
        bağlayıcıdır. Aksi hâlde sonuç 'düzeltilerek' iyileştirilebilirdi."""
        out: Dict[str, Dict] = {}
        for r in self.read():
            if r.get("kind") != "outcome":
                continue
            pid = r.get("prediction_id")
            if pid and pid not in out:
                out[pid] = r
        return out

    def open_predictions(self, now: Optional[float] = None) -> List[Dict]:
        cozulen = set(self.outcomes())
        return [p for p in self.predictions()
                if p.get("prediction_id") not in cozulen]


# ── değerlendirme ──────────────────────────────────────────────────────────

def evaluate_open(led: Ledger, price_lookup, now: Optional[float] = None) -> int:
    """Açık tahminleri fiyat geçmişiyle çözer. Çözülen sayısını döndürür.

    `price_lookup(symbol, t0_iso, t1_iso) -> DataFrame[open,high,low,close]`
    çağıranın sağladığı gerçek bar kaynağıdır. Bar yoksa tahmin AÇIK kalır —
    "herhalde zaman aşımıdır" varsayımı YAPILMAZ.
    """
    import pandas as pd
    from .firstpassage import first_passage_times
    from .horizons import HORIZON_MIN

    simdi = now or time.time()
    n = 0
    for p in led.open_predictions():
        t0 = pd.Timestamp(p["timestamp"])
        dk = HORIZON_MIN.get(p["horizon"])
        if dk is None:
            continue
        t1 = t0 + pd.Timedelta(minutes=dk)
        if t1.timestamp() > simdi:
            continue                                  # ufuk henüz dolmadı
        bars = price_lookup(p["symbol"], t0.isoformat(), t1.isoformat())
        if bars is None or not len(bars):
            continue
        giris = float(p["entry"])
        hedef, stop = float(p["net1_exit"]), float(p["stop"])
        uzun = p["direction"] == "LONG"
        # `bars` ZATEN girişten SONRAKİ barlardır. `first_passage_times`
        # i barındaki giriş için i+1'den taradığı için dizinin başına giriş
        # fiyatını yerleştirmek ŞART; aksi hâlde girişten sonraki İLK bar
        # hiç kontrol edilmez (ölçülen ve düzeltilen birim kayması).
        h = np.concatenate([[giris], bars["high"].to_numpy(dtype=float)])
        l = np.concatenate([[giris], bars["low"].to_numpy(dtype=float)])
        bir = np.ones((1, len(h)))
        # LONG: hedef yukarıda, stop aşağıda. SHORT: tam tersi.
        tT = first_passage_times(h if uzun else l, hedef * bir, len(h),
                                 "up" if uzun else "dn")[0]
        tS = first_passage_times(l if uzun else h, stop * bir, len(h),
                                 "dn" if uzun else "up")[0]
        a, b = int(tT[0]), int(tS[0])
        if a and (not b or a < b):
            sonuc, bar = "TP_FIRST", a
        elif b and (not a or b < a):
            sonuc, bar = "SL_FIRST", b
        elif a and b and a == b:
            sonuc, bar = "TIMEOUT", None            # aynı bar → belirsiz, sayılmaz
        else:
            sonuc, bar = "TIMEOUT", None

        gh = bars["high"].to_numpy(dtype=float)
        gl = bars["low"].to_numpy(dtype=float)
        mfe = float((gh.max() / giris - 1.0) * 100.0) if uzun else \
            float((1.0 - gl.min() / giris) * 100.0)
        mae = float((gl.min() / giris - 1.0) * 100.0) if uzun else \
            float((1.0 - gh.max() / giris) * 100.0)
        maliyet = float(p.get("cost_pct") or 0.0)
        if sonuc == "TP_FIRST":
            net = 1.0
        elif sonuc == "SL_FIRST":
            gs = abs(stop / giris - 1.0) * 100.0
            net = -(gs + maliyet)
        else:
            son = float(bars["close"].iloc[-1])
            ham = (son / giris - 1.0) * 100.0
            net = (ham if uzun else -ham) - maliyet

        pp = p.get("p_target_first")
        led.record_outcome(Outcome(
            prediction_id=p["prediction_id"], resolved_at=_now(), outcome=sonuc,
            bars_to_resolution=bar,
            hours_to_resolution=(None if bar is None else bar * 5 / 60.0),
            realized_mfe_pct=round(mfe, 4), realized_mae_pct=round(mae, 4),
            realized_cost_pct=maliyet, realized_net_pct=round(net, 4),
            predicted_p=pp,
            prediction_error=(None if pp is None else
                              round((1.0 if sonuc == "TP_FIRST" else 0.0) - pp, 4))))
        n += 1
    return n


def scorecard(led: Ledger, days: Optional[int] = None) -> Dict:
    """Şartname 46/104 — parite × ufuk karnesi. Payda YAYIMLANAN sinyaldir."""
    import datetime as dt
    kesim = None
    if days:
        kesim = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    son = led.outcomes()
    gruplar: Dict[str, Dict] = {}
    # Payda YAYIMLANAN sinyaldir (§106) — gölge kayıtlar buraya GİRMEZ.
    for p in led.predictions(source=SOURCE_PUBLISHED):
        if kesim is not None:
            try:
                if dt.datetime.fromisoformat(
                        p["timestamp"].replace("Z", "+00:00")) < kesim:
                    continue
            except Exception:
                continue
        k = f"{p['symbol']}|{p['horizon']}|{p['direction']}"
        g = gruplar.setdefault(k, {
            "symbol": p["symbol"], "horizon": p["horizon"],
            "direction": p["direction"], "published": 0, "resolved": 0,
            "tp_first": 0, "sl_first": 0, "timeout": 0,
            "not_traded": 0, "net_sum": 0.0, "pred_sum": 0.0, "errors": []})
        g["published"] += 1
        o = son.get(p["prediction_id"])
        if not o:
            continue
        g["resolved"] += 1
        oc = o["outcome"]
        if oc == "TP_FIRST":
            g["tp_first"] += 1
        elif oc == "SL_FIRST":
            g["sl_first"] += 1
        elif oc == "TIMEOUT":
            g["timeout"] += 1
        else:
            g["not_traded"] += 1
        if o.get("realized_net_pct") is not None:
            g["net_sum"] += float(o["realized_net_pct"])
        if o.get("prediction_error") is not None:
            g["errors"].append(float(o["prediction_error"]))
        if p.get("p_target_first") is not None:
            g["pred_sum"] += float(p["p_target_first"])

    out = []
    for k, g in gruplar.items():
        yay = g["published"]
        fiyat = g["tp_first"] + g["sl_first"] + g["timeout"]
        out.append({
            **{x: g[x] for x in ("symbol", "horizon", "direction", "published",
                                 "resolved", "tp_first", "sl_first", "timeout",
                                 "not_traded")},
            # Şartname 104: payda YAYIMLANAN sinyal — başarısızlar düşürülmez
            "net1_precision": (g["tp_first"] / yay) if yay else None,
            "false_opportunity_rate": ((g["sl_first"] + g["timeout"]) / yay
                                       if yay else None),
            "tp_first_rate_of_traded": (g["tp_first"] / fiyat) if fiyat else None,
            "realized_net_mean_pct": (g["net_sum"] / fiyat) if fiyat else None,
            "mean_predicted_p": (g["pred_sum"] / yay) if yay else None,
            "calibration_error": (
                abs(g["pred_sum"] / yay - g["tp_first"] / fiyat)
                if yay and fiyat else None),
        })
    return {"window_days": days, "cells": sorted(
        out, key=lambda r: (-r["published"], r["symbol"]))}


def calibration_board(led: Ledger,
                      edges: Sequence[float] = (0.6, 0.7, 0.8, 0.9, 1.0),
                      source: Optional[str] = None) -> List[Dict]:
    """Şartname 105 — 'model %80 dediğinde gerçekte ne oldu?' tablosu.

    Kalibrasyon, karnenin tersine, GÖLGE kayıtlardan da beslenir: soru
    "yayımladıklarım tuttu mu" değil, "olasılık tahminim doğru mu". İkincisi
    yalnız yayımlanan bir avuç sinyalle asla ölçülemez."""
    son = led.outcomes()
    kovalar: List[Dict] = []
    alt = [0.0] + list(edges[:-1])
    for a, b in zip(alt, edges):
        n = basari = 0
        toplam_p = 0.0
        for p in led.predictions(source=source):
            pp = p.get("p_target_first")
            o = son.get(p["prediction_id"])
            if pp is None or o is None or o["outcome"] not in PRICE_OUTCOMES:
                continue
            if a <= pp < b or (b >= 1.0 and pp == 1.0):
                n += 1
                toplam_p += pp
                basari += 1 if o["outcome"] == "TP_FIRST" else 0
        kovalar.append({"bucket": f"{a:.0%}-{b:.0%}", "n": n,
                        "predicted": (toplam_p / n) if n else None,
                        "actual": (basari / n) if n else None})
    return kovalar


def daily_panel(led: Ledger, tz: str = "Europe/Istanbul") -> Dict:
    """Şartname 58 — BUGÜN paneli. Başarısızlar da görünür."""
    import datetime as dt
    try:
        from zoneinfo import ZoneInfo
        z = ZoneInfo(tz)
    except Exception:
        z = dt.timezone.utc
    bugun = dt.datetime.now(z).date()
    son = led.outcomes()
    yayin = aktif = suresi_dolan = tp = sl = to = 0
    net = 0.0
    for p in led.predictions():
        try:
            t = dt.datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00"))
        except Exception:
            continue
        if t.astimezone(z).date() != bugun:
            continue
        yayin += 1
        o = son.get(p["prediction_id"])
        if not o:
            try:
                bitis = dt.datetime.fromisoformat(
                    p["valid_until"].replace("Z", "+00:00"))
                if bitis < dt.datetime.now(dt.timezone.utc):
                    suresi_dolan += 1
                else:
                    aktif += 1
            except Exception:
                aktif += 1
            continue
        if o["outcome"] == "TP_FIRST":
            tp += 1
        elif o["outcome"] == "SL_FIRST":
            sl += 1
        elif o["outcome"] == "TIMEOUT":
            to += 1
        if o.get("realized_net_pct") is not None:
            net += float(o["realized_net_pct"])
    return {"date": str(bugun), "timezone": tz,
            "qualified_seen": yayin, "active": aktif, "expired": suresi_dolan,
            "tp_first": tp, "sl_first": sl, "timeout": to,
            "realized_net_pct": round(net, 4),
            "false_opportunity_rate": ((sl + to) / yayin) if yayin else None,
            "note": "Başarısız sinyaller listeden düşürülmez (şartname 106)."}
