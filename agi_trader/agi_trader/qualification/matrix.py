"""Parite × Ufuk matrisi — şartname 48, 49, 50, 60, 88, 89, 93, 99, 115.

Bu modül ölçüm sonuçlarını KULLANICIYA GÖSTERİLECEK hâle getirir ve tek bir
kuralı uygular: her satır ya gerçek bir ölçümden gelir ya da `UNKNOWN` yazar.
Boşluk doldurmak için sayı üretilmez (şartname 115).

MATRİSİN OKUNUŞU
  satır  = ufuk (5m … 24h; 48h yalnız referans)
  sütun  = LONG / SHORT
  hücre  = P(net +%1 hedef, stop'tan önce) + %95 alt sınır + RobustEV + durum

En üstte HER ZAMAN iki satır bulunur:
  GARANTİ: YOK
  NET +%1 İÇİN EN YÜKSEK DOĞRULANMIŞ OLASILIK: …  (ya da YOK)
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .horizons import HORIZON_MIN, REFERENCE_ONLY, primary_horizons
from .robust import best_horizon, earliest_qualified_horizon, horizon_narrative
from .state import QualificationState, RejectionCode

GUARANTEE_LINE = "GARANTİ: YOK — piyasa sonucu kesin değildir."

# Şartname 99 — aynı faktöre maruz fırsatlar bağımsız gösterilemez.
CORRELATION_CLUSTERS = {
    "CRYPTO_BETA": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
                    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"],
}


def cluster_of(symbol: str) -> Optional[str]:
    for ad, uyeler in CORRELATION_CLUSTERS.items():
        if symbol in uyeler:
            return ad
    return None


def pair_card(symbol: str, cells: List[Dict], *,
              market_price: Optional[float] = None,
              data_quality: Optional[float] = None,
              liquidity_score: Optional[float] = None,
              cost_model: str = "ESTIMATED",
              model_version: str = "yok",
              valid_seconds: int = 480) -> Dict:
    """Şartname 88/89/115 — bir paritenin tam kartı.

    `cells`: bu paritenin (ufuk × yön) hücreleri; her biri `status`,
    `p_target_first`, `robust_utility` vb. taşır.
    """
    ana = [c for c in cells if c["horizon"] in primary_horizons()]
    en_iyi = best_horizon(ana)
    en_erken = earliest_qualified_horizon(ana)
    anlati = horizon_narrative(ana, en_iyi)

    red: List[str] = []
    if en_iyi is None:
        red.append(RejectionCode.NO_QUALIFIED_HORIZON)

    simdi = time.time()
    # GEÇERLİLİK, SEÇİLEN UFKUN KENDİ yarı ömründen gelir.
    # Bütün ufukların maksimumunu almak, 5 dakikalık bir kurulumda 24 saatlik
    # ufkun ömrünü ilan etmek olurdu (ölçüldü: 5m yarı ömrü 30 sn iken kart
    # 480 sn geçerlilik gösteriyordu).
    omur = (en_iyi or {}).get("expected_half_life_sec")
    if omur:
        valid_seconds = int(max(60, min(valid_seconds, 2.0 * float(omur))))
    kart: Dict = {
        "symbol": symbol,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(simdi)),
        # ── DEĞİŞMEZ: bu alan HER ZAMAN False ve şema dışına çıkamaz ──
        "guaranteed": False,
        "guarantee_line": GUARANTEE_LINE,

        "best_horizon": (en_iyi["horizon"] if en_iyi else None),
        "earliest_qualified_horizon": (en_erken["horizon"] if en_erken else None),
        "direction": (en_iyi["direction"] if en_iyi else None),
        "status": (en_iyi["status"] if en_iyi else QualificationState.NO_EDGE
                   if any(c["status"] == QualificationState.NO_EDGE for c in ana)
                   else QualificationState.RESEARCH_ONLY),

        "market_price": market_price,
        "fair_price": None,

        "entry_low": (en_iyi or {}).get("entry_low"),
        "optimal_entry": (en_iyi or {}).get("optimal_entry"),
        "entry_high": (en_iyi or {}).get("entry_high"),
        "max_chase_price": (en_iyi or {}).get("max_chase_price"),

        "net_1pct_exit": (en_iyi or {}).get("net_1pct_exit"),
        "stop": _pick(en_iyi or {}, "stop_price"),
        "time_exit": (en_iyi or {}).get("time_exit"),

        "p_target_first": (en_iyi or {}).get("p_target_first"),
        "p_target_first_lower95": _pick(en_iyi or {}, "p_target_lower95",
                                        "lower95"),
        "p_stop_first": (en_iyi or {}).get("p_stop_first"),
        "p_timeout": (en_iyi or {}).get("p_timeout"),

        "baseline_target_rate": (en_iyi or {}).get("baseline"),
        "required_probability_lift": (en_iyi or {}).get("required_lift"),
        "actual_probability_lift": (en_iyi or {}).get("actual_lift"),

        "expected_net_return": (en_iyi or {}).get("expected_net_return"),
        "expected_value": (en_iyi or {}).get("ev"),
        "robust_expected_value": (en_iyi or {}).get("robust_ev"),
        "expected_target_time_hours": (en_iyi or {}).get("median_hours_to_tp"),
        "target_time_p25_hours": (en_iyi or {}).get("hours_to_tp_p25"),
        "target_time_p75_hours": (en_iyi or {}).get("hours_to_tp_p75"),

        # Aynı iç içe/düz ikiliği kart düzeyinde de vardı; genelleştirilmiş
        # idempotans testi bunu üçüncü örnek olarak yakaladı.
        "expected_mfe": _pick(en_iyi or {}, "mfe_p50",
                              default=((en_iyi or {}).get("mfe") or {}).get("p50")),
        "expected_mae": _pick(en_iyi or {}, "mae_p50",
                              default=((en_iyi or {}).get("mae") or {}).get("p50")),

        "fill_probability": (en_iyi or {}).get("fill_probability"),
        "execution_probability": (en_iyi or {}).get("execution_probability"),
        "max_capacity_usd": (en_iyi or {}).get("max_capacity_usd"),

        "liquidity_score": liquidity_score,
        "data_quality": data_quality,
        "risk_score": (en_iyi or {}).get("risk_score"),

        "effective_sample_size": _pick(en_iyi or {}, "n_eff_used", "n_eff"),
        "brier_score": (en_iyi or {}).get("brier"),
        "calibration_error": (en_iyi or {}).get("ece"),
        "dsr": (en_iyi or {}).get("dsr"),
        "pbo": (en_iyi or {}).get("pbo"),

        "model_version": model_version,
        "cost_model": cost_model,
        "valid_until": (time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                      time.gmtime(simdi + valid_seconds))
                        if en_iyi else None),
        "entry_valid_seconds": valid_seconds,
        "detected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(simdi)),
        "expected_half_life_sec": (en_iyi or {}).get("expected_half_life_sec"),

        "rejection_reasons": red + ((en_iyi or {}).get("rejection_reasons") or []),
        "rejection_reasons_tr": [RejectionCode.TR.get(c, c) for c in
                                 (red + ((en_iyi or {}).get("rejection_reasons") or []))],
        "convergence": (en_iyi or {}).get("convergence"),
        "why_this_horizon": anlati,
        "correlation_cluster": cluster_of(symbol),
        "horizons": flag_horizon_comparability(
            [_horizon_row(c) for c in
             sorted(cells, key=lambda x: x["horizon_minutes"])]),
    }
    return kart


def _conv_summary(c: Optional[Dict]) -> Optional[Dict]:
    """Yakınsamanın panel için yeterli özeti — ağır diziler DIŞARIDA."""
    if not c:
        return None
    return {k: c.get(k) for k in
            ("verdict", "ci_width", "n_effective", "period_spread",
             "regime_spread", "shrink_ratio", "checks", "reasons")}


def _pick(c: Dict, *adaylar, default=None):
    """İlk DOLU alanı seç — fonksiyon İDEMPOTENT olsun diye.

    ⚠️ ÖLÇÜLEN HATA: canlı katman `matrix.json` içindeki HAZIR satırı alıp
    yeniden `pair_card`'a veriyor, yani `_horizon_row` ikinci kez koşuyor.
    İlk geçişte `p_target_lower95` → `lower95` diye yeniden adlandırılıyordu;
    ikinci geçişte eski ad aranınca **None** kalıyor ve panelde ALT95, R/R,
    ETKİN N, HEDEF σ sütunları BOŞ görünüyordu. Artık her iki ad da kabul
    edilir; fonksiyon kendi çıktısını da okuyabilir."""
    for a in adaylar:
        v = c.get(a)
        if v is not None:
            return v
    return default


def _horizon_row(c: Dict) -> Dict:
    """Şartname 89 — bütün ufuklar döner, kalifiye olmayanlar dâhil."""
    return {
        "horizon": c["horizon"],
        "horizon_minutes": c["horizon_minutes"],
        "direction": c["direction"],
        "reference_only": c["horizon"] in REFERENCE_ONLY,
        "p_target_first": c.get("p_target_first"),
        "lower95": _pick(c, "p_target_lower95", "lower95"),
        "p_stop_first": c.get("p_stop_first"),
        "p_timeout": c.get("p_timeout"),
        "baseline": c.get("baseline"),
        "required_lift": c.get("required_lift"),
        "actual_lift": c.get("actual_lift"),
        "net_mean": c.get("net_mean"),
        "net_lower95": c.get("net_lower95"),
        "net_t_stat": c.get("net_t_stat"),
        "blind_net_mean": c.get("blind_net_mean"),
        "selected_stop_pct": c.get("selected_stop_pct"),
        "selected_frac": c.get("selected_frac"),
        "selection_source": c.get("selection_source"),
        "breakeven_probability": c.get("breakeven_probability"),
        "dsr": c.get("dsr"), "dsr_program": c.get("dsr_program"),
        "pbo": c.get("pbo"),
        "brier": c.get("brier"), "brier_base": c.get("brier_base"),
        "ece": c.get("ece"), "calibration_slope": c.get("calibration_slope"),
        "cost_pct": c.get("cost_pct"),
        "robust_ev": c.get("robust_ev"),
        "robust_utility": c.get("robust_utility"),
        "expected_holding_hours": c.get("expected_holding_hours"),
        "median_hours_to_tp": c.get("median_hours_to_tp"),
        "target_distance_sigma": _pick(c, "target_distance_sigma_median",
                                       "target_distance_sigma"),
        "n_eff": _pick(c, "n_eff_used", "n_eff"),
        "n_raw": c.get("n_raw"),
        # ⚠️ AYNI İDEMPOTANS HATASI, İKİ ALAN GEÇ FARK EDİLDİ: araştırma
        # hücresinde bunlar iç içe (`c["mfe"]["p50"]`), ama bu fonksiyonun
        # KENDİ çıktısında düz (`mfe_p50`). Canlı katman matrix.json satırını
        # ikinci kez `pair_card`'a verdiği için ikinci geçişte iç içe ad
        # bulunamıyor ve alan NULL'a düşüyordu. Ölçüldü: matrix.json'da
        # 594/594 dolu, canlı API'de 540/540 boş. Panelde MFE/MAE sütunları
        # bu yüzden hep boştu.
        "mfe_p50": _pick(c, "mfe_p50", default=(c.get("mfe") or {}).get("p50")),
        "mae_p50": _pick(c, "mae_p50", default=(c.get("mae") or {}).get("p50")),
        "ambiguous_pct": c.get("ambiguous_pct"),
        "rr": _pick(c, "rr_median", "rr"),
        "stop_pct": _pick(c, "stop_pct", "stop_pct_median"),
        "stop_sigma_mult": c.get("stop_sigma_mult"),
        "target_gross_pct": c.get("target_gross_pct"),
        # canlı katmanın ürettiği fiyat düzeyleri ve maliyet — araştırma
        # hücresinde yoktur, canlı taramada doldurulur
        "net_1pct_exit": c.get("net_1pct_exit"),
        "stop_price": c.get("stop_price"),
        "entry_low": c.get("entry_low"),
        "optimal_entry": c.get("optimal_entry"),
        "entry_high": c.get("entry_high"),
        "max_chase_price": c.get("max_chase_price"),
        "order_type": c.get("order_type"),
        "entry_reason": c.get("entry_reason"),
        "expected_half_life_sec": c.get("expected_half_life_sec"),
        "entry_band_pct": c.get("entry_band_pct"),
        "fill_probability": c.get("fill_probability"),
        "cost_pct": c.get("cost_pct"),
        "cost_model": c.get("cost_model"),
        "p_model_live": c.get("p_model_live"),
        "time_exit": c.get("time_exit"),
        # Yakınsama ÖZETİ taşınır, tam detay değil. Detay (daralma eğrisi,
        # dönem/rejim tahminleri) hücre başına ~1 KB tutuyor ve 27 parite ×
        # 22 satırda yanıtı 1,7 MB'a çıkarıyordu — panel bunu dakikada bir
        # çekiyor. Tam detay `/api/qualification/convergence` ucundan.
        "convergence": _conv_summary(c.get("convergence")),
        "status": c.get("status"),
        "color": QualificationState.COLOR.get(c.get("status", ""), "gri"),
        "tradable": c.get("tradable", False),
        "rejection_reasons": c.get("rejection_reasons", []),
        "rejection_reasons_tr": [RejectionCode.TR.get(x, x)
                                 for x in c.get("rejection_reasons", [])],
    }


def flag_horizon_comparability(cells: List[Dict]) -> List[Dict]:
    """Ufuklar arası oran karşılaştırmasının GEÇERLİ olup olmadığını işaretler.

    ⚠️ ÖLÇÜLEREK BULUNDU: kör taban ufuk uzadıkça artmalı (kümülatif insidans),
    ama ACEUSDT SHORT'ta 8h %65,0 → 12h %64,6 ile DÜŞTÜ. Sebep ölçüm hatası
    değil: her ufuk için stop AYRI seçiliyor (RobustEV ile) ve 12h'de daha DAR
    bir stop kazandı (k=0,50 vs 0,75). Dar stop daha sık vurulur, hedef-önce
    oranı düşer.

    Aynı k ile karşılaştırınca monotonluk korunuyor: 8h %58,8 → 12h %64,6.

    Yani sayı doğru, KARŞILAŞTIRMA yanıltıcı. Bu fonksiyon farklı stop seçilen
    komşu ufukları işaretler ki panel "neden düştü?" sorusunu cevaplayabilsin.
    Sessizce bırakmak, kullanıcıyı ölçüm hatası var sanmaya iter.
    """
    for d in ("LONG", "SHORT"):
        hs = sorted([c for c in cells if c.get("direction") == d],
                    key=lambda x: x.get("horizon_minutes", 0))
        for a, b in zip(hs, hs[1:]):
            ka = a.get("stop_sigma_mult")
            kb = b.get("stop_sigma_mult")
            ba, bb = a.get("baseline"), b.get("baseline")
            if ba is None or bb is None:
                continue
            dustu = bb < ba - 1e-9
            farkli_stop = (ka is not None and kb is not None and ka != kb)
            if dustu:
                b["baseline_dropped_from_prev"] = True
                b["comparability_note"] = (
                    f"{a['horizon']} → {b['horizon']} arasında kör taban düştü "
                    + (f"çünkü stop çarpanı farklı seçildi (k={ka:g} → {kb:g}); "
                       f"aynı k'da monotonluk korunur."
                       if farkli_stop else
                       "ve stop çarpanı AYNI — bu beklenmeyen bir durumdur, "
                       "kümülatif insidans azalmamalıydı."))
                b["comparability_ok"] = bool(farkli_stop)
            elif farkli_stop:
                b["stop_changed_from_prev"] = True
    return cells


def rank_pairs(cards: List[Dict]) -> List[Dict]:
    """Şartname 60 — ham hedef olasılığına göre SIRALAMA YAPILMAZ.

    Öncelik: RobustEV → alt güven sınırı → gerçekleşme olasılığı → likidite →
    kalibrasyon → fırsat ömrü → kuyruk riski."""
    def anahtar(k: Dict):
        return (
            -(k.get("robust_expected_value") or -1e18),
            -(k.get("p_target_first_lower95") or 0.0),
            -(k.get("execution_probability") or 0.0),
            -(k.get("liquidity_score") or 0.0),
            (k.get("calibration_error") if k.get("calibration_error") is not None else 1e9),
        )
    kalifiye = [k for k in cards if k.get("best_horizon")]
    return sorted(kalifiye, key=anahtar)


# Bilerek üretilmeyen alanlar ve GEREKÇELERİ. Buradaki bir alanın boş olması
# kusur değildir; burada OLMAYAN bir alanın tamamen boş olması kusurdur.
# (gerekçe, beklenen_mi). `beklenen=True` → tasarım kararı, arıza değil.
# `beklenen=False` → gerçek eksiklik ama SEBEBİ BİLİNİYOR; listede olmayan bir
# boş alan ise teşhis edilmemiş arızadır. Üç durum üç ayrı şeydir.
KNOWN_ABSENT = {
    "fill_probability": (
        "kuyruk/akış geçmişi yok — dolum olasılığı ölçülemez; taker varsayılır",
        True),
    "execution_probability": (
        "yürütme katmanı kurulmadı — sistem ölçüm modunda", True),
    "dsr_program": (
        "artefaktlar bu alanı üreten koddan ÖNCEKİ koşumdan geliyor; model-fazı "
        "getirileri saklanmadığı için sonradan hesaplanamaz. Sonraki tam "
        "koşumda dolar. Matematiksel sınır: n_trials büyüdükçe DSR küçülür, "
        "`dsr` 540/540 hücrede 0 olduğu için `dsr_program` da 0'dır — ama "
        "ÖLÇÜLMEDİĞİ için değer yazılmaz.",
        False),
}


def field_coverage(cards: List[Dict]) -> Dict:
    """HİÇBİR hücrede dolmayan alanları bulur.

    ⚠️ BU FONKSİYON BİR KUSUR SINIFINI GÖRÜNÜR KILMAK İÇİN VAR. Ölçüldü:
    `mfe_p50`/`mae_p50` canlı API'de 540/540 boştu (idempotans hatası),
    `dsr_program` 594/594 boştu (artefaktlar bu alanı üreten koddan ÖNCEKİ
    koşumdan geliyor). İkisi de panelde sessizce boş sütun olarak görünüyordu
    ve yalnızca elle denetimde fark edildi.

    'Hesaplandı ve boş çıktı' ile 'hiç hesaplanmadı' aynı şey değildir;
    ikincisi bir arıza bildirimidir."""
    satir = [h for k in cards for h in k.get("horizons", [])]
    if not satir:
        return {"n_rows": 0, "always_null": [], "unexplained": [], "note": ""}
    alanlar: Dict[str, int] = {}
    for h in satir:
        for a, v in h.items():
            if v is None:
                alanlar[a] = alanlar.get(a, 0) + 1
    hep_bos = sorted(a for a, n in alanlar.items() if n == len(satir))
    aciklanmamis = [a for a in hep_bos if a not in KNOWN_ABSENT]
    kayit = []
    for a in hep_bos:
        gerekce, beklenen = KNOWN_ABSENT.get(
            a, ("TEŞHİS EDİLMEDİ — ya hiç hesaplanmıyor ya da katmanlar "
                "arasında taşınırken kayboluyor", False))
        kayit.append({"field": a, "reason": gerekce, "expected": beklenen,
                      "diagnosed": a in KNOWN_ABSENT})
    return {
        "n_rows": len(satir),
        "always_null": kayit,
        "unexplained": aciklanmamis,
        "n_undiagnosed": len(aciklanmamis),
        "note": ("Bu alanlar hiçbir hücrede dolmadı. 'expected' olanlar bilerek "
                 "kapsam dışıdır; 'diagnosed' ama beklenmeyen olanlar sebebi "
                 "bilinen eksiklerdir; hiçbiri değilse ARIZADIR."),
    }


def scanner_summary(cards: List[Dict], eligible: int, excluded: int,
                    last_update: Optional[str] = None) -> Dict:
    """Şartname 119 — ana ekranın en üst bandı."""
    kalifiye = [k for k in cards if k.get("best_horizon")]
    kumeler: Dict[str, List[str]] = {}
    for k in kalifiye:
        kumeler.setdefault(k.get("correlation_cluster") or "TEKIL", []).append(k["symbol"])
    return {
        "title": "NET +%1 FIRSAT TARAYICISI",
        "guarantee": "YOK",
        "guarantee_line": GUARANTEE_LINE,
        "markets_scanned": eligible + excluded,
        "markets_eligible": eligible,
        "markets_excluded": excluded,
        "qualified_markets": len(kalifiye),
        "current_opportunities": len(kalifiye),
        "last_update": last_update or time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime()),
        "empty_message": ("ŞU AN NET +%1 İÇİN QUALIFIED FIRSAT YOK — "
                          "alternatif düşük kaliteli sinyal üretilmez."),
        "correlation_warning": ([
            f"{ad} kümesinde {len(u)} fırsat aynı piyasa faktörüne maruz: "
            f"{', '.join(u)}" for ad, u in kumeler.items() if len(u) > 1]),
        # Boş sütun sessizce geçmesin: hiç dolmayan alanlar burada raporlanır.
        "field_coverage": field_coverage(cards),
    }
