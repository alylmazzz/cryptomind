"""Canlı tarayıcı — araştırma artefaktlarını GERÇEK ZAMANLI piyasaya bağlar.

BU MODÜL EMİR GÖNDERMEZ. Yalnız ölçer, nitelendirir ve yayımlar.

FAIL-CLOSED (şartname 113, 114)
Artefakt yoksa, model yüklenemiyorsa, veri bayatsa ya da defter senkron
değilse sistem basit bir tabana düşüp "AL" ÜRETMEZ. Eksik her bileşen bir
RED KODU'na dönüşür ve hücre NO_TRADE kalır.

CANLI OLASILIK NEREDEN GELİR
Araştırma koşusu her (ufuk, yön) için bir softmax modeli ve parite kalibrasyon
başlığı üretir. Burada aynı özellikler CANLI 5 dakikalık barlardan yeniden
hesaplanır ve aynı ağırlıklarla çarpılır. Özellik listesi ve sırası model
kaydındaki `names` ile bire bir eşleşmezse tahmin ÜRETİLMEZ (sessiz kayma
yerine açık ret).
"""
from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..opportunity.costs import (capacity_curve, estimate_costs,
                                 ladder_from_row, required_gross_move_pct)
from . import features as FT
from .horizons import HORIZON_MIN, primary_horizons
from .ledger import SOURCE_SHADOW, Ledger, Prediction, make_id
from .matrix import GUARANTEE_LINE, pair_card, rank_pairs, scanner_summary
from .model import CLASS_INDEX, PlattHead, SoftmaxModel
from .regime import classify
from .robust import build_entry_plan, expected_holding_hours, stress_test
from .state import QualificationState, RejectionCode
from .targets import (STOP_SIGMA_MULTS, TARGET_NET_PCT, CostProfile,
                      gross_target_pct, horizon_sigma_pct,
                      profile_from_recorder, sigma_bar_pct)

FAPI = "https://fapi.binance.com"
# 7 günlük özellikler 2.016 bar, rejim sınıflandırıcısının ısınması 2.304 bar
# ister. 2.200 ile canlı rejim HER ZAMAN "UNKNOWN" kalıyordu — ölçüldü ve
# düzeltildi. Pay bırakılıyor ki eksik bar olan paritede de ısınma dolsun.
NEEDED_BARS = 2600
SIGNAL_VALID_SECONDS = 480         # şartname 29 — giriş penceresi


def signal_half_life_sec(sigma_bar_pct_now: Optional[float],
                         entry_band_pct: Optional[float]) -> Optional[float]:
    """Sinyalin BEKLENEN YARI ÖMRÜ (2. mesaj 41).

    Bir giriş sinyali, fiyat giriş bandının dışına çıkınca geçersizleşir.
    Rastgele yürüyüşte |fiyat| bandın yarısına ulaşma süresi
    t ≈ (band/2 / σ_bar)² bardır. Yarı ömür bunun yarısıdır.

    ⚠️ Bu bir MODEL DEĞİL, geometridir: "bu oynaklıkta bu bant ne kadar
    dayanır?" Ölçülmüş bir sinyal ömrü değildir ve öyle sunulmaz.
    σ ya da bant bilinmiyorsa `None` döner — varsayılan üretilmez."""
    if not sigma_bar_pct_now or not entry_band_pct or sigma_bar_pct_now <= 0:
        return None
    bar = (entry_band_pct / 2.0 / sigma_bar_pct_now) ** 2
    return float(max(30.0, bar * 5 * 60 / 2.0))
CAPACITY_LEVELS = (100, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000)
MAX_DATA_AGE_SEC = 900


def _klines(symbol: str, limit: int = NEEDED_BARS,
            timeout: float = 20.0) -> Optional[pd.DataFrame]:
    """Son N adet 5 dakikalık bar (sayfalayarak)."""
    parcalar: List[List] = []
    end = None
    try:
        while len(parcalar) < limit:
            n = min(1500, limit - len(parcalar))
            u = f"{FAPI}/fapi/v1/klines?symbol={symbol}&interval=5m&limit={n}"
            if end is not None:
                u += f"&endTime={end}"
            with urllib.request.urlopen(u, timeout=timeout) as r:
                b = json.loads(r.read().decode())
            if not b:
                break
            parcalar = b + parcalar
            end = int(b[0][0]) - 1
            if len(b) < n:
                break
    except Exception:
        return None
    if not parcalar:
        return None
    d = pd.DataFrame(parcalar, columns=[
        "ts", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    for c in ("open", "high", "low", "close", "volume", "trades",
              "taker_buy_base", "quote_volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["dt"] = pd.to_datetime(d["ts"].astype("int64"), unit="ms", utc=True)
    return d.drop_duplicates("ts").set_index("dt").sort_index()


class Artifacts:
    """Araştırma çıktıları. Yoksa sistem RESEARCH_ONLY'de kalır."""

    def __init__(self, d: Path):
        self.dir = Path(d)
        self.matrix: Dict = {}
        self.models: Dict = {}
        self.validation: Dict = {}
        self.profiles: Dict = {}
        self.errors: List[str] = []
        self._load()

    def _read(self, ad: str) -> Dict:
        p = self.dir / ad
        if not p.exists():
            self.errors.append(f"{ad} yok")
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            self.errors.append(f"{ad} okunamadı: {type(e).__name__}")
            return {}

    def _load(self) -> None:
        self.matrix = self._read("matrix.json")
        self.models = self._read("models.json")
        self.validation = self._read("validation_report.json")
        self.profiles = self._read("cost_profiles.json")

    @property
    def ok(self) -> bool:
        return bool(self.matrix.get("cards")) and bool(self.validation)

    def cell(self, symbol: str, horizon: str, direction: str) -> Optional[Dict]:
        for k in self.matrix.get("cards", []):
            if k.get("symbol") != symbol:
                continue
            for h in k.get("horizons", []):
                if h.get("horizon") == horizon and h.get("direction") == direction:
                    return h
        return None

    def model(self, horizon: str, direction: str) -> Tuple[Optional[SoftmaxModel],
                                                           Dict]:
        m = self.models.get(f"{horizon}|{direction}")
        if not m or not m.get("model"):
            return None, {}
        try:
            return SoftmaxModel.from_dict(m["model"]), (m.get("pair_heads") or {})
        except Exception:
            return None, {}


def _live_probability(art: Artifacts, symbol: str, horizon: str, direction: str,
                      df: pd.DataFrame, closes: Dict[str, pd.Series],
                      profile: CostProfile) -> Tuple[Optional[float], List[str]]:
    """Canlı özelliklerden model olasılığı. Uyuşmazlıkta None + gerekçe."""
    mdl, heads = art.model(horizon, direction)
    if mdl is None:
        return None, [RejectionCode.MISSING_MODEL]
    H = HORIZON_MIN[horizon] // 5
    sb = sigma_bar_pct(df)
    mkt = FT.market_factor(closes, symbol, index=df.index)
    X, ad, ail, tf = FT.build(df, sb, market_ret_4h=mkt)
    tgt = gross_target_pct(profile, HORIZON_MIN[horizon] / 60.0, direction)
    X, ad, ail, tf = FT.add_geometry(X, ad, ail, tf, sb, tgt, H)

    # Model budanmış olabilir → sütunları İSİMDEN eşle. Sıra/eksik uyuşmazlığı
    # sessizce yanlış tahmin üretmesin diye eksik isim varsa reddedilir.
    idx = {a: i for i, a in enumerate(ad)}
    if any(a not in idx for a in mdl.names):
        return None, [RejectionCode.MISSING_MODEL]
    Xs = X[:, [idx[a] for a in mdl.names]]
    son = Xs[-1:]
    if not np.isfinite(son).all():
        return None, [RejectionCode.DATA_STALE]
    P = mdl.predict_proba(son)
    if P is None:
        return None, [RejectionCode.MISSING_MODEL]
    p = float(P[0, CLASS_INDEX["TP"]])
    hd = heads.get(symbol)
    if hd and hd.get("fitted"):
        p = float(PlattHead(**hd).apply(np.array([p]))[0])
    return p, []


def scan(artifacts_dir: Path, symbols: Sequence[str],
         recorder_feats: Optional[pd.DataFrame] = None,
         ledger: Optional[Ledger] = None,
         notional: float = 1_000.0,
         horizons: Optional[Sequence[str]] = None) -> Dict:
    """Tam canlı tarama → tarayıcı özeti + parite kartları."""
    art = Artifacts(Path(artifacts_dir))
    horizons = list(horizons or primary_horizons())
    veri: Dict[str, pd.DataFrame] = {}
    kapali: Dict[str, pd.Series] = {}
    for s in symbols:
        d = _klines(s)
        if d is not None and len(d) > 300:
            veri[s] = d
            kapali[s] = d["close"]
        # 2 çekirdekli sunucuda 27 paritelik tarama CPU'yu doyurup uvicorn'u
        # aç bırakabiliyor. Bu kısa uyku GIL'i bırakır ve panel istekleri
        # tarama sürerken de yanıtlanır.
        time.sleep(0.05)

    kartlar: List[Dict] = []
    simdi = time.time()
    for s in symbols:
        df = veri.get(s)
        if df is None:
            kartlar.append(_bos_kart(s, [RejectionCode.DATA_STALE],
                                     "canlı 5m bar alınamadı"))
            continue
        prof = profile_from_recorder(s, recorder_feats)
        satir = None
        if recorder_feats is not None and len(recorder_feats):
            g = recorder_feats[recorder_feats["symbol"] == s]
            if len(g):
                satir = g.sort_values("ts").iloc[-1]

        fiyat = float(df["close"].iloc[-1])
        yas = simdi - df.index[-1].timestamp()
        bayat = yas > MAX_DATA_AGE_SEC
        rej = classify(df)
        sb = sigma_bar_pct(df)
        son_sb = float(sb[-1]) if np.isfinite(sb[-1]) else None

        bid_c = ask_c = None
        spread = float(satir["spread_bps"]) if satir is not None and \
            "spread_bps" in satir and pd.notna(satir["spread_bps"]) else None
        if satir is not None:
            bc, ac = ladder_from_row(satir, "bid"), ladder_from_row(satir, "ask")
            if bc and ac:
                bid_c, ask_c = bc, ac
        olculdu = bool(bid_c and ask_c)

        hucreler: List[Dict] = []
        for hz in horizons:
            for d in ("LONG", "SHORT"):
                hucreler.append(_live_cell(
                    art, s, hz, d, df, kapali, prof, fiyat, son_sb,
                    rej, spread, bid_c, ask_c, satir, notional, bayat, olculdu))
        # Tavan burada; SEÇİLEN ufka göre kısaltmayı `pair_card` yapar.
        gecerli_sn = SIGNAL_VALID_SECONDS
        kart = pair_card(s, hucreler, market_price=fiyat,
                         data_quality=(0.4 if bayat else 1.0),
                         liquidity_score=(1.0 if olculdu else 0.5),
                         cost_model=("MEASURED_L2_VWAP" if olculdu else "ESTIMATED"),
                         model_version=_model_version(art),
                         valid_seconds=gecerli_sn)
        kart["regime"] = str(rej["vol_regime"].iloc[-1])
        kart["structure"] = str(rej["structure"].iloc[-1])
        kart["data_age_sec"] = round(yas, 1)
        kart["capacity_curve"] = (
            capacity_curve((float(satir.get("bid_depth_usd") or 0.0),
                            float(satir.get("ask_depth_usd") or 0.0)),
                           spread or 0.0, TARGET_NET_PCT,
                           levels=CAPACITY_LEVELS,
                           bid_curve=bid_c, ask_curve=ask_c)
            if satir is not None else [])
        kart["max_capacity_usd"] = _max_capacity(kart["capacity_curve"])

        # VERİ KALİTESİ — tek skor, bileşenler AYRI görünür
        from .schema import check_card, data_quality
        kart_ihlal = check_card(kart, strict=False)
        kapsam = None
        if satir is not None:
            kum = [satir.get(f"bid_cum_{b:g}bps") for b in (1, 2, 5, 10)]
            var = sum(1 for x in kum if x is not None and x == x)
            kapsam = var / len(kum)
        dq = data_quality(freshness_sec=yas, max_fresh_sec=MAX_DATA_AGE_SEC,
                          completeness=(len(df) / NEEDED_BARS if len(df) else None),
                          book_coverage=kapsam,
                          schema_violations=len(kart_ihlal),
                          cost_model=kart.get("cost_model"))
        kart["data_quality"] = dq.score / 100.0
        kart["data_quality_detail"] = dq.to_dict()
        kartlar.append(kart)
        time.sleep(0.05)                       # aynı gerekçe (bkz. yukarısı)

    # FAIL-FAST ŞEMA — birim/aralık ihlali sessizce geçmez.
    # `strict=False`: canlı serviste tek bir bozuk kart bütün taramayı
    # düşürmemeli; ihlaller SAYILIR, yayımlanır ve veri kalitesini düşürür.
    from .schema import check_scan
    ihlaller = check_scan({"cards": kartlar}, strict=False)

    ozet = scanner_summary(kartlar, eligible=len(kartlar), excluded=0)
    ozet["schema_violations"] = len(ihlaller)
    ozet["schema_violation_sample"] = [str(v) for v in ihlaller[:5]]
    ozet["artifact_errors"] = art.errors
    ozet["artifacts_ok"] = art.ok
    ozet["combinations"] = len(symbols) * len(horizons) * 2
    sirali = rank_pairs(kartlar)

    if ledger is not None:
        _deftere_yaz(ledger, kartlar, sirali)

    return json_safe({
        "scanner": ozet, "cards": kartlar,
        "ranked": [k["symbol"] for k in sirali],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def json_safe(o):
    """JSON'a giren her sayıyı temizle — `inf`/`NaN` JSON'da GEÇERSİZDİR.

    ⚠️ ÖLÇÜLEN HATA: ince defterli bir paritede 1.000 $'lık emir kaydedilen
    derinliği aşınca `vwap_offset_bps` sonsuz döner ve bu değer maliyet
    dökümüne yazılır. Evren 8'den 27 pariteye çıkınca `/api/qualification`
    **500** vermeye başladı (`Out of range float values are not JSON
    compliant: inf`).

    Sonsuz bir maliyet BİLGİ TAŞIR: "bu büyüklükte emir bu defterde
    dolmaz". Bu yüzden değer `None` yapılır ama `warnings` alanındaki
    açıklama korunur — sessizce sıfıra çevrilmez."""
    import math as _m
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if _m.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    return o


def _model_version(art: Artifacts) -> str:
    if not art.models:
        return "yok"
    return f"softmax_l2/{len(art.models)}kafa"


def _max_capacity(curve: List[Dict]) -> Optional[float]:
    uygun = [c["notional_usd"] for c in curve if c.get("feasible")]
    return float(max(uygun)) if uygun else None


def _bos_kart(symbol: str, red: List[str], not_: str) -> Dict:
    return {"symbol": symbol, "guaranteed": False,
            "guarantee_line": GUARANTEE_LINE,
            "best_horizon": None, "status": QualificationState.NO_DATA,
            "rejection_reasons": red, "why_this_horizon": not_,
            "horizons": []}


def _live_cell(art: Artifacts, symbol: str, hz: str, direction: str,
               df: pd.DataFrame, closes: Dict[str, pd.Series],
               prof: CostProfile, fiyat: float, sigma_bar: Optional[float],
               rej: pd.DataFrame, spread: Optional[float],
               bid_c, ask_c, satir, notional: float,
               bayat: bool, olculdu: bool) -> Dict:
    """Araştırma kanıtı + canlı fiyat → tek hücre."""
    arastirma = art.cell(symbol, hz, direction) or {}
    H = HORIZON_MIN[hz] // 5
    saat = HORIZON_MIN[hz] / 60.0

    # maliyet — gerçek eğri varsa GERÇEK VWAP, yoksa yaklaşım (beyan edilir)
    c = estimate_costs(notional,
                       float(satir.get("bid_depth_usd") or 0.0) if satir is not None else 0.0,
                       float(satir.get("ask_depth_usd") or 0.0) if satir is not None else 0.0,
                       spread or 0.0, holding_hours=saat,
                       funding_rate_8h=float(satir.get("funding_rate") or 0.0)
                       if satir is not None else 0.0,
                       direction=direction, bid_curve=bid_c, ask_curve=ask_c)
    brut = required_gross_move_pct(TARGET_NET_PCT, c)

    stop_pct = None
    if sigma_bar is not None and math.isfinite(sigma_bar):
        m = arastirma.get("stop_sigma_mult") or 1.0
        stop_pct = float(np.clip(horizon_sigma_pct(np.array([sigma_bar]), H)[0] * m,
                                 0.15, 6.0))

    if not math.isfinite(brut) or stop_pct is None:
        hedef_f = stop_f = None
    elif direction == "LONG":
        hedef_f = fiyat * (1 + brut / 100.0)
        stop_f = fiyat * (1 - stop_pct / 100.0)
    else:
        hedef_f = fiyat * (1 - brut / 100.0)
        stop_f = fiyat * (1 + stop_pct / 100.0)

    p_live, red = _live_probability(art, symbol, hz, direction, df, closes, prof)

    h = dict(arastirma)
    h.update({
        "horizon": hz, "horizon_minutes": HORIZON_MIN[hz],
        "direction": direction,
        "target_gross_pct": (None if not math.isfinite(brut) else round(brut, 4)),
        "net_1pct_exit": hedef_f, "stop_price": stop_f,
        "stop_pct": (None if stop_pct is None else round(stop_pct, 4)),
        "cost_pct": round(c.total_pct, 5),
        "cost_model": c.model,
        "cost_breakdown": c.to_dict(),
        "p_model_live": p_live,
        "time_exit": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + HORIZON_MIN[hz] * 60)),
    })

    plan = build_entry_plan(
        direction, fiyat,
        None if satir is None else _bid(fiyat, spread),
        None if satir is None else _ask(fiyat, spread),
        None if sigma_bar is None else sigma_bar * math.sqrt(max(1, H)))
    h.update({k: v for k, v in plan.to_dict().items()
              if k in ("entry_low", "optimal_entry", "entry_high",
                       "max_chase_price", "fill_probability", "order_type")})
    h["entry_reason"] = plan.reason
    # §41 — sinyal ömrü: tespit, geçerlilik ve beklenen yarı ömür
    bant = None
    if plan.entry_low is not None and plan.entry_high is not None and fiyat > 0:
        bant = abs(plan.entry_high - plan.entry_low) / fiyat * 100.0
    h["expected_half_life_sec"] = signal_half_life_sec(sigma_bar, bant)
    h["entry_band_pct"] = (None if bant is None else round(bant, 4))

    # canlı kapılar araştırma durumunun ÜSTÜNE eklenir; hiçbiri gevşetmez
    ek: List[str] = list(h.get("rejection_reasons") or []) + list(red)
    if bayat:
        ek.append(RejectionCode.DATA_STALE)
    if not olculdu:
        ek.append(RejectionCode.INSUFFICIENT_L2)
    if not math.isfinite(brut):
        ek.append(RejectionCode.COST_TOO_HIGH)
    if plan.fill_probability is None and plan.order_type == "LIMIT":
        ek.append(RejectionCode.FILL_PROBABILITY_LOW)
    h["rejection_reasons"] = sorted(set(ek))
    h["rejection_reasons_tr"] = [RejectionCode.TR.get(x, x)
                                 for x in h["rejection_reasons"]]
    h["tradable"] = bool(arastirma.get("tradable")) and not h["rejection_reasons"]
    if not h["tradable"] and h.get("status") in QualificationState.TRADABLE:
        h["status"] = QualificationState.UNVERIFIED
    h.setdefault("status", QualificationState.RESEARCH_ONLY)
    h["color"] = QualificationState.COLOR.get(h["status"], "gri")
    return h


def _bid(mid: float, spread_bps: Optional[float]) -> float:
    return mid * (1 - (spread_bps or 0.0) / 20000.0)


def _ask(mid: float, spread_bps: Optional[float]) -> float:
    return mid * (1 + (spread_bps or 0.0) / 20000.0)


def _kaydet(led: Ledger, kart: Dict) -> None:
    """Yalnız YAYIMLANAN (qualified) fırsatlar deftere yazılır."""
    if not kart.get("best_horizon"):
        return
    pid = make_id(kart["symbol"], kart["best_horizon"], kart["direction"],
                  kart["timestamp"])
    if any(p.get("prediction_id") == pid for p in led.predictions()):
        return
    led.record_prediction(Prediction(
        prediction_id=pid, timestamp=kart["timestamp"], symbol=kart["symbol"],
        horizon=kart["best_horizon"], direction=kart["direction"],
        status=kart["status"], entry=float(kart.get("optimal_entry")
                                           or kart.get("market_price") or 0.0),
        net1_exit=float(kart.get("net_1pct_exit") or 0.0),
        stop=float(kart.get("stop") or 0.0),
        p_target_first=kart.get("p_target_first"),
        p_target_lower95=kart.get("p_target_first_lower95"),
        p_stop_first=kart.get("p_stop_first"),
        p_timeout=kart.get("p_timeout"),
        baseline=kart.get("baseline_target_rate"),
        required_lift=kart.get("required_probability_lift"),
        actual_lift=kart.get("actual_probability_lift"),
        robust_ev=kart.get("robust_expected_value"),
        expected_target_hours=kart.get("expected_target_time_hours"),
        cost_model=kart.get("cost_model", "ESTIMATED"),
        cost_pct=None, max_capacity_usd=kart.get("max_capacity_usd"),
        data_quality=kart.get("data_quality"),
        model_version=kart.get("model_version", "yok"),
        features_hash="", valid_until=kart.get("valid_until") or "",
    ))


# ── GÖLGE KAYIT ────────────────────────────────────────────────────────────

def _en_iyi_golge_ufku(kart: Dict) -> Optional[Dict]:
    """Sistem işlem YAPSAYDI hangi ufku seçerdi — nitelendirmeden bağımsız.

    Ölçüt `robust_ev`: kalifiye hücre yokken bile "en az kötü" hücre budur ve
    tahmin↔gerçekleşme sınavı için doğru adaydır. Referans-yalnız ufuklar
    (48h) dışarıda kalır; fiyat düzeyi eksik olan satır değerlendirilemez."""
    adaylar = [h for h in (kart.get("horizons") or [])
               if not h.get("reference_only")
               and h.get("robust_ev") is not None
               and h.get("net_1pct_exit") and h.get("stop_price")
               and h.get("optimal_entry")]
    if not adaylar:
        return None
    return max(adaylar, key=lambda h: h["robust_ev"])


def _deftere_yaz(led: Ledger, kartlar: List[Dict], sirali: List[Dict]) -> None:
    """Yayımlananları ve gölgeleri deftere yazar.

    ⚠️ İKİ FARKLI LİSTE, SEBEBİ VAR: `rank_pairs` YALNIZ kalifiye kartları
    döndürür. Gölge döngüsü de sıralı liste üzerinden koşarsa sıfır kalifiye
    olan bir sistemde HİÇ gölge yazılmaz — ki gölge kaydın var olma sebebi tam
    olarak bu durumdur. Ölçüldü: ilk sürüm sunucuda 0 gölge üretti."""
    for k in sirali:
        _kaydet(led, k)
    acik = {(p.get("symbol"), p.get("horizon"), p.get("direction"))
            for p in led.open_predictions()}
    for k in kartlar:
        _kaydet_golge(led, k, acik)


def _kaydet_golge(led: Ledger, kart: Dict, acik: set) -> None:
    """Kalifiye OLMAYAN en iyi hücreyi `shadow` olarak deftere yazar.

    NE DEĞİLDİR: sinyal değil, tavsiye değil, karneye girmez, `tradable`
    yapmaz. Durum alanı hücrenin GERÇEK durumudur (çoğunlukla NO_EDGE).

    NEDEN VAR: sıfır QUALIFIED fırsat üretilen bir sistemde defter boş kalır;
    boş defterde ana metrik (Gerçekleşen ÷ Tahmin Edilen Net EV) ve kalibrasyon
    tablosu ölçülemez. Model gerçekten yön bilmiyorsa bunu canlı veriyle
    KANITLAMAK gerekir — varsaymak değil.

    HIZ SINIRI: aynı (parite, ufuk, yön) için açık bir gölge kayıt varken
    yenisi yazılmaz. Böylece kayıtlar ufuk uzunluğu kadar aralanır ve
    örtüşen, bağımsız olmayan gözlemler birikmez."""
    if kart.get("best_horizon"):
        return                       # yayımlanan zaten `_kaydet` ile yazıldı
    h = _en_iyi_golge_ufku(kart)
    if h is None:
        return
    anahtar = (kart["symbol"], h["horizon"], h["direction"])
    if anahtar in acik:
        return
    acik.add(anahtar)
    pid = make_id(kart["symbol"], h["horizon"], h["direction"],
                  kart.get("timestamp"), salt=SOURCE_SHADOW)
    led.record_prediction(Prediction(
        prediction_id=pid, timestamp=kart.get("timestamp") or "",
        symbol=kart["symbol"], horizon=h["horizon"], direction=h["direction"],
        status=h.get("status") or QualificationState.RESEARCH_ONLY,
        entry=float(h["optimal_entry"]), net1_exit=float(h["net_1pct_exit"]),
        stop=float(h["stop_price"]),
        p_target_first=h.get("p_target_first"),
        p_target_lower95=h.get("lower95"),
        p_stop_first=h.get("p_stop_first"), p_timeout=h.get("p_timeout"),
        baseline=h.get("baseline"), required_lift=h.get("required_lift"),
        actual_lift=h.get("actual_lift"), robust_ev=h.get("robust_ev"),
        expected_target_hours=h.get("expected_holding_hours"),
        cost_model=h.get("cost_model", "ESTIMATED"), cost_pct=h.get("cost_pct"),
        max_capacity_usd=kart.get("max_capacity_usd"),
        data_quality=kart.get("data_quality"),
        model_version=kart.get("model_version", "yok"), features_hash="",
        valid_until=h.get("time_exit") or "", source=SOURCE_SHADOW,
    ))
