"""
Maliyet motoru — "net %1" hesabının tek doğruluk kaynağı.

Şartnamenin 1., 7. ve 8. maddelerinin karşılığı: sistem hiçbir yerde
`profit = sell - buy` kullanmaz. Her fırsat, GERÇEKLEŞEBİLİR fiyat üzerinden
ve tüm maliyetler düşüldükten sonra değerlendirilir.

İKİ MALİYET MODELİ — hangisinin kullanıldığı `CostEstimate.model` ile beyan edilir.

1. `gercek_L2_vwap` — TERCİH EDİLEN. `recorder.collect_book_ladder` kümülatif
   derinlik eğrisini kaydeder; `vwap_offset_bps` bu eğri üzerinde gerçek bir
   VWAP yürüyüşü yapar. Bu bir tahmin değil, kaydedilmiş defterin kendisidir.

2. `dogrusal_defter_yaklasimi` — GERİ DÜŞÜŞ. Eğri yoksa ±%2 bandındaki toplam
   derinlikten doğrusal etki varsayılır:  etki ≈ (bant/2) × (N/D) × güvenlik.

   Bu yaklaşımın hatası ÖLÇÜLDÜ ve İKİ YÖNLÜDÜR:
     BTC 1 M$   → gerçek 0,50 bps · doğrusal 1,08 bps  (2× ABARTIYOR)
     AVAX 10 k$ → gerçek 1,88 bps · doğrusal 0,54 bps  (3,5× EKSİK gösteriyor)
   Yani "yaklaşım hep güvenli taraftadır" demek YANLIŞTIR; ince defterlerde
   maliyeti sistematik olarak az gösterir. Eğri varsa mutlaka eğri kullanılır.

ÇİFT SAYIM: gerçek eğri mid'den ölçüldüğü için giriş+çıkış uzaklıkları TAM
spread'i zaten içerir; o modda `spread_bps` AYRICA eklenmez.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

# Varsayılanlar — Binance taker referans alındı. Venue bazında ezilebilir.
DEFAULT_TAKER_BPS = 4.0       # %0,04
DEFAULT_MAKER_BPS = 2.0       # %0,02
DEPTH_BAND_PCT = 2.0          # recorder ±%2 derinlik kaydediyor
IMPACT_SAFETY = 1.5           # doğrusal defter varsayımı etkiyi az gösterir
LATENCY_RESERVE_BPS = 2.0     # emir gönderimi ile dolum arası fiyat kayması
FAILURE_RESERVE_BPS = 3.0     # kısmi dolum / red / yeniden deneme karşılığı


@dataclass
class CostEstimate:
    """Tüm maliyetler BAZ PUAN (bps) cinsinden. 100 bps = %1."""
    entry_fee_bps: float = 0.0
    exit_fee_bps: float = 0.0
    spread_bps: float = 0.0
    entry_impact_bps: float = 0.0
    exit_impact_bps: float = 0.0
    funding_bps: float = 0.0
    borrow_bps: float = 0.0
    transfer_bps: float = 0.0
    latency_bps: float = 0.0
    failure_bps: float = 0.0
    # Beklenen funding KAZANCI — toplama GİRMEZ, yalnız raporlanır.
    funding_credit_bps: float = 0.0
    model: str = "dogrusal_defter_yaklasimi"
    warnings: List[str] = field(default_factory=list)

    @property
    def total_bps(self) -> float:
        """Toplam maliyet. `funding_credit_bps` KASITLI olarak dışarıda —
        beklenen bir kazanç, ödenmiş bir maliyeti azaltmaz."""
        return float(
            self.entry_fee_bps + self.exit_fee_bps + self.spread_bps
            + self.entry_impact_bps + self.exit_impact_bps
            + self.funding_bps + self.borrow_bps + self.transfer_bps
            + self.latency_bps + self.failure_bps)

    @property
    def total_pct(self) -> float:
        return self.total_bps / 100.0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["total_bps"] = round(self.total_bps, 3)
        d["total_pct"] = round(self.total_pct, 4)
        return d


def vwap_offset_bps(curve: Dict[float, Optional[float]], notional: float,
                    max_covered_bps: Optional[float] = None,
                    min_offset_bps: float = 0.0) -> Tuple[float, Optional[str]]:
    """GERÇEK VWAP — kümülatif derinlik eğrisinden ortalama dolum uzaklığı (bps).

    `curve`: {uzaklık_bps: o uzaklığa kadar birikmiş DOLAR nominal}
    Örn: {1: 120_000, 2: 260_000, 5: 700_000, ...}

    Yöntem: eğri kovalar arasında doğrusal kabul edilir; her dilimde ortalama
    uzaklık (x0+x1)/2'dir. N nominali dolana kadar dilimler yürünür ve
    nominal-ağırlıklı ortalama uzaklık döner. Bu, doğrusal-defter YAKLAŞIMI
    değil, kaydedilmiş defterin ta kendisidir.

    None değerli kova = SANSÜRLÜ (defter o uzaklığa ulaşmıyor); dolu sayılmaz.
    """
    if notional <= 0:
        return 0.0, None
    noktalar = sorted((b, v) for b, v in curve.items() if v is not None)
    if not noktalar:
        return float("inf"), "derinlik eğrisi yok"

    # İlk dilim mid'den DEĞİL, en iyi fiyattan başlar: küçük bir emir tamamen
    # best ask'te dolar ve mid'e uzaklığı yarım spread'dir. `min_offset_bps`
    # verilmezse taban 0 alınır ve küçük emirlerde etki fazla çıkar.
    onceki_bps, onceki_kum = float(max(0.0, min_offset_bps)), 0.0
    dolan, agirlikli = 0.0, 0.0
    for b, kum in noktalar:
        dilim = kum - onceki_kum
        if dilim <= 0:
            onceki_bps, onceki_kum = b, kum
            continue
        kalan = notional - dolan
        orta = (onceki_bps + b) / 2.0
        if dilim >= kalan:
            agirlikli += kalan * orta
            dolan += kalan
            return float(agirlikli / notional), None
        agirlikli += dilim * orta
        dolan += dilim
        onceki_bps, onceki_kum = b, kum

    kapsam = noktalar[-1][0]
    trunc = (max_covered_bps is not None and kapsam < max_covered_bps)
    return float("inf"), (
        f"emir kaydedilen derinliği aşıyor ({dolan:,.0f} $ / {notional:,.0f} $ "
        f"— eğri {kapsam:g} bps'e kadar" + (", SANSÜRLÜ)" if trunc else ")"))


def ladder_from_row(row, side: str) -> Dict[float, Optional[float]]:
    """Kaydedici satırından bir tarafın kümülatif eğrisini çıkarır."""
    out: Dict[float, Optional[float]] = {}
    for b in (1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0):
        k = f"{side}_cum_{b:g}bps"
        try:
            v = row[k]
        except Exception:
            continue
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        out[b] = float(v)
    return out


def linear_impact_bps(notional: float, depth: float,
                      band_pct: float = DEPTH_BAND_PCT,
                      safety: float = IMPACT_SAFETY) -> Tuple[float, Optional[str]]:
    """Doğrusal defter varsayımıyla fiyat etkisi (bps) + uyarı.

    notional > depth ise emir bandı tüketir; bu durumda etki üst sınırdan
    hesaplanır ve fırsat "kapasite aşıldı" uyarısı alır."""
    if depth <= 0:
        return float("inf"), "derinlik verisi yok"
    oran = notional / depth
    if oran >= 1.0:
        return float("inf"), f"emir ±%{band_pct:.0f} bandındaki derinliği aşıyor ({oran:.1f}×)"
    # ortalama etki = bandın yarısı × doluluk oranı
    bps = (band_pct * 100.0 / 2.0) * oran * safety
    uyari = None
    if oran > 0.25:
        uyari = f"emir bant derinliğinin %{oran*100:.0f}'ini yiyor — etki tahmini kırılgan"
    return float(bps), uyari


def estimate_costs(notional_usd: float,
                   bid_depth_usd: float,
                   ask_depth_usd: float,
                   spread_bps: float,
                   *,
                   taker: bool = True,
                   holding_hours: float = 4.0,
                   funding_rate_8h: float = 0.0,
                   direction: str = "LONG",
                   borrow_bps_per_day: float = 0.0,
                   transfer_bps: float = 0.0,
                   bid_curve: Optional[Dict[float, Optional[float]]] = None,
                   ask_curve: Optional[Dict[float, Optional[float]]] = None) -> CostEstimate:
    """Bir gidiş-dönüş işlemin toplam maliyeti.

    funding_rate_8h: perp funding (oran, ör. 0.0001 = %0,01). LONG pozisyon
    pozitif funding ÖDER; SHORT tahsil eder. İşaret buna göre uygulanır."""
    c = CostEstimate()
    fee = DEFAULT_TAKER_BPS if taker else DEFAULT_MAKER_BPS
    c.entry_fee_bps = fee
    c.exit_fee_bps = fee
    gercek_egri = bool(bid_curve and ask_curve)
    # ÇİFT SAYIM UYARISI: gerçek L2 eğrisi mid'den ölçülür, yani giriş ve çıkış
    # uzaklıklarının toplamı TAM SPREAD'i zaten içerir. Bu durumda spread'i
    # ayrıca eklemek maliyeti iki kez saymak olur. Yalnız yaklaşım modunda eklenir.
    c.spread_bps = 0.0 if gercek_egri else max(0.0, float(spread_bps))

    # GERÇEK VWAP eğrisi varsa onu kullan; yoksa doğrusal defter yaklaşımına düş.
    giris_egri = ask_curve if direction == "LONG" else bid_curve
    cikis_egri = bid_curve if direction == "LONG" else ask_curve
    if gercek_egri:
        yarim = max(0.0, float(spread_bps)) / 2.0
        gi, u1 = vwap_offset_bps(giris_egri, notional_usd, min_offset_bps=yarim)
        ci, u2 = vwap_offset_bps(cikis_egri, notional_usd, min_offset_bps=yarim)
        c.model = "gercek_L2_vwap"
    else:
        gi, u1 = linear_impact_bps(
            notional_usd, ask_depth_usd if direction == "LONG" else bid_depth_usd)
        ci, u2 = linear_impact_bps(
            notional_usd, bid_depth_usd if direction == "LONG" else ask_depth_usd)
        c.warnings.append("L2 eğrisi yok — doğrusal defter YAKLAŞIMI kullanıldı")
    c.entry_impact_bps, c.exit_impact_bps = gi, ci
    for u in (u1, u2):
        if u:
            c.warnings.append(u)

    # FUNDING — ASİMETRİK VE KASITLI
    #
    # Funding 8 saatte bir ödenir; 24 saatlik bir ufukta üç kez. Ama gelecekteki
    # funding BİLİNMEZ — elimizde yalnız ŞU ANKİ oran var.
    #
    # Bu belirsizlik lehte değil ALEYHTE yorumlanır:
    #   • funding ÖDENECEKSE (maliyet) → tam uygulanır
    #   • funding TAHSİL EDİLECEKSE (kazanç) → SIFIR sayılır
    #
    # Neden: kazancı hedefe saymak, brüt hedefi net hedefin ALTINA indirir
    # ("net %1 için %0,9 hareket yeter") ve bu ancak funding üç periyot boyunca
    # aynı işarette kalırsa doğrudur. Ölçüldü: ACEUSDT/BICOUSDT'de canlı
    # negatif funding `cost_pct`'i eksiye düşürüp hedefi indiriyordu.
    #
    # Beklenen kazanç YOK SAYILMAZ, ayrı alanda raporlanır (`funding_credit_bps`)
    # — bilgi kaybolmaz, yalnız hedefi indirmez.
    if funding_rate_8h:
        periyot = holding_hours / 8.0
        isaret = 1.0 if direction == "LONG" else -1.0
        ham = isaret * float(funding_rate_8h) * periyot * 10000.0
        if ham >= 0:
            c.funding_bps = ham                 # ödenecek: tam uygula
        else:
            c.funding_bps = 0.0                 # tahsil edilecek: SAYMA
            c.funding_credit_bps = -ham
            c.warnings.append(
                f"funding {(-ham):.1f} bps KAZANÇ bekleniyor ama hedefe "
                f"sayılmadı — gelecekteki funding bilinmez, belirsizlik "
                f"aleyhte yorumlanır")

    c.borrow_bps = borrow_bps_per_day * (holding_hours / 24.0)
    c.transfer_bps = transfer_bps
    c.latency_bps = LATENCY_RESERVE_BPS
    c.failure_bps = FAILURE_RESERVE_BPS
    return c


def required_gross_move_pct(target_net_pct: float, costs: CostEstimate) -> float:
    """Net hedefe ulaşmak için fiyatın YÜZDE KAÇ hareket etmesi gerekir.

    Toplama değil ÇARPMA ile: net %1 hedefi, maliyetler düşüldükten sonra
    kalan orandır. (1+g)(1-c) = 1+n  →  g = (1+n)/(1-c) - 1"""
    n = target_net_pct / 100.0
    cst = costs.total_pct / 100.0
    if cst >= 1.0:
        return float("inf")
    return ((1.0 + n) / (1.0 - cst) - 1.0) * 100.0


def capacity_curve(depths: Tuple[float, float], spread_bps: float,
                   target_net_pct: float = 1.0,
                   levels: Tuple[float, ...] = (100, 500, 1_000, 2_500, 5_000,
                                                10_000, 25_000, 50_000, 100_000),
                   **kw) -> List[Dict]:
    """Şartname 8: fırsat TEK bir yüzde değildir, sermayeye göre eğridir."""
    bid_d, ask_d = depths
    out = []
    for n in levels:
        c = estimate_costs(n, bid_d, ask_d, spread_bps, **kw)
        g = required_gross_move_pct(target_net_pct, c)
        out.append({"notional_usd": n,
                    "cost_bps": round(c.total_bps, 2),
                    "required_gross_pct": (None if not math.isfinite(g) else round(g, 3)),
                    "feasible": math.isfinite(g),
                    "warnings": c.warnings})
    return out
