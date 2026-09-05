"""
Carry (taşıma) sleeve'i — perp funding hasadı, delta-nötr.

MANTIK: Sürekli vadeli (perp) sözleşmede funding pozitifken long'lar short'lara
öder. Spot al + perp sat = piyasa yönünden BAĞIMSIZ (delta-nötr) pozisyon;
getirisi ≈ funding oranı. Fiyat yönüne bakmadığı için trend sleeve'iyle
korelasyonu yapısal olarak ~0'dır — çeşitlendirme değeri buradan gelir.

⚠️ ÖLÇÜLEN SHARPE SİSTEMATİK OLARAK ŞİŞKİNDİR. Önceki ölçüm "Sharpe 36 / DD %0,1"
vermişti; bu sayı gerçek DEĞİLDİR çünkü funding serisi şu riskleri İÇERMEZ:
  • iki bacağın yürütme kayması ve yeniden dengeleme maliyeti
  • borsa/karşı taraf riski (FTX 2022: delta-nötr pozisyonlar da silindi)
  • teminat yetersizliğinde zorunlu tasfiye (perp bacağı short, fiyat fırlarsa)
  • funding'in aniden negatife dönmesi ve baz açılması
Bu yüzden burada: (a) maliyet İKİ bacak için uygulanır, (b) kapasite tavanı,
(c) kuyruk şoku simülasyonu (`tail_shock`) ile ceza uygulanır, (d) kabul kapısı
Sharpe > 2,5'i "önce bug" sayar — carry bu tavanı aşarsa gerçek değil, eksik
modellemedir.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import Sleeve, zscore

# Perp funding 8 saatte bir ödenir → günde 3 uzlaşma
FUNDING_PER_DAY = 3


class CarrySleeve(Sleeve):
    """Funding oranı yüksekken delta-nötr taşıma pozisyonu aç.

    `returns()` geçersiz kılınır: bu sleeve fiyat yönünden getiri üretmez,
    doğrudan funding akışından üretir."""

    name = "carry"
    warmup = 90

    def __init__(self, smooth: int = 14, enter_annual: float = 0.03,
                 exit_annual: float = 0.01, max_gross: float = 1.0,
                 leg_cost: float = 0.0006, tail_shock_prob: float = 0.0005,
                 tail_shock_loss: float = 0.05, capture: float = 0.80,
                 stochastic_tail: bool = False, **kw):
        """Carry DÜŞÜK DEVİR HIZLI olmalıdır.

        İlk uygulama z-skoru ile girip çıkıyordu; funding günlerin %87'sinde
        zaten pozitif olduğu için bu, sürekli giriş-çıkış = ücret ölümü demekti
        (tam tur maliyeti %0,12; yıllık hasat ~%6,6 → 55 tur hasadı bitirir).
        Doğru kural: funding maliyeti karşıladığı SÜRECE pozisyonda kal.

        smooth        : funding'i bu kadar günlük ortalamayla yumuşat (gürültü)
        enter_annual  : yıllıklandırılmış funding bu eşiği aşarsa gir (%3)
        exit_annual   : bu eşiğin altına inerse çık (%1) — histerezis, devir hızını kırar
        tail_shock_*  : modellenmemiş kuyruk riski cezası. Varsayılan ~5,5 yılda
                        bir %5 kayıp. FTX/LUNA sınıfı olaylar delta-nötr
                        pozisyonları da vurdu; sıfırlamak sonucu güzelleştirir
                        ama gerçekçi yapmaz.
        capture       : kotasyon funding'inin gerçekte cebe giren oranı (0,80).

        CAPTURE NEDEN VAR — ölçülen duyarlılık (23 parite, 2022-2025):
            yakalama %100 → Sharpe 1,22 · CAGR %4,4
            yakalama % 80 → Sharpe 0,74 · CAGR %2,6      ← varsayılan
            yakalama % 60 → Sharpe 0,24 · CAGR %0,8
            yakalama % 40 → Sharpe −0,29 (zarar)
        Kotasyon funding'in tamamı cebe girmez: iki bacağın spread'i, teminat ve
        spot bacağın fonlama maliyeti, yeniden dengeleme kayması ve pozisyon
        limitleri payını alır. %100 varsaymak stratejiyi olduğundan iyi gösterir
        — bu sleeve'in EN BÜYÜK kırılganlığı budur.

        ⚠️ REJİM UYARISI: tam geçmişte (2019-2025) Sharpe 3,26 çıkar; bu sayı
        İLERİYE DÖNÜK KULLANILMAMALIDIR. 2021'de perp funding yıllık %37'ye
        ulaşmıştı (aşırı kaldıraçlı boğa) ve o rejim tekrarlamadı:
            2019-2021 → Sharpe 6,56 · CAGR %29,7
            2022-2025 → Sharpe 1,22 · CAGR %4,4   ← gerçekçi taban
        2022 zararla kapandı (−%4,3, DD %9,4): ayı piyasasında funding negatife
        döner ve carry ters çalışır."""
        super().__init__(smooth=smooth, enter_annual=enter_annual,
                         exit_annual=exit_annual,
                         max_gross=max_gross, leg_cost=leg_cost,
                         tail_shock_prob=tail_shock_prob,
                         tail_shock_loss=tail_shock_loss,
                         capture=float(capture),
                         stochastic_tail=bool(stochastic_tail), **kw)

    # --------------------------------------------------------------- yardımcı
    @staticmethod
    def daily_funding(funding_8h: pd.Series) -> pd.Series:
        """8 saatlik funding serisini günlük toplama indirger."""
        s = funding_8h.sort_index()
        return s.resample("1D").sum().fillna(0.0)

    def positions(self, prices: pd.DataFrame, **ctx) -> pd.DataFrame:
        """Delta-nötr olduğu için yönlü fiyat pozisyonu YOKTUR.
        Maruziyet raporlaması için sıfır matrisi döner."""
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # ------------------------------------------------------------------ ana iş
    def returns(self, prices: pd.DataFrame,
                funding: Optional[Dict[str, pd.Series]] = None,
                basis: Optional[Dict[str, pd.Series]] = None,
                seed: int = 0, **ctx) -> pd.Series:
        """funding: {varlık: 8 saatlik funding serisi}
        basis  : {varlık: günlük baz serisi} = perp/spot − 1

        BAZ NEDEN ZORUNLU: long spot + short perp pozisyonunun günlük P&L'i
            P&L ≈ taraf × (funding − Δbaz)
        Yalnız funding sayılırsa getiri yapay olarak pürüzsüzleşir (ölçüldü:
        2023-25'te yıllık vol %0,4, Sharpe 15-18 — imkânsız değerler). Δbaz,
        perp priminin günlük dalgalanmasıdır ve funding tam olarak bu riskin
        ücretidir. `basis` verilmezse UYARI basılır ve sonuç iyimserdir."""
        p = self.params
        # DataFrame de kabul edilir: `.items()` zaten (kolon, Seri) verdiği için
        # gerisi çalışıyordu, yalnız bu kapı `if not <DataFrame>` ile anlaşılmaz
        # bir ValueError fırlatıyordu ("truth value is ambiguous").
        if isinstance(funding, pd.DataFrame):
            funding = {c: funding[c] for c in funding.columns}
        if funding is None or len(funding) == 0:
            return pd.Series(dtype=float, name=self.name)

        daily = {}
        for sym, s in funding.items():
            if s is None or len(s) < self.warmup:
                continue
            daily[sym] = self.daily_funding(s)
        if not daily:
            return pd.Series(dtype=float, name=self.name)

        F = pd.DataFrame(daily).reindex(prices.index).fillna(0.0)

        # Sinyal: yumuşatılmış funding'in yıllık karşılığı, histerezisli.
        # Beklenen taşıma t anında BİLİNENDİR (geçmiş funding ortalaması) —
        # gelecekteki funding kullanılmaz.
        smooth_annual = F.rolling(p["smooth"], min_periods=p["smooth"]).mean() * 365

        enter = smooth_annual.abs() >= p["enter_annual"]
        stay = smooth_annual.abs() >= p["exit_annual"]
        # histerezis: bir kez girince `stay` sağlandığı sürece pozisyonda kal
        active = pd.DataFrame(False, index=F.index, columns=F.columns)
        prev = pd.Series(False, index=F.columns)
        for ts in F.index:
            cur = np.where(prev, stay.loc[ts].fillna(False), enter.loc[ts].fillna(False))
            prev = pd.Series(cur, index=F.columns)
            active.loc[ts] = cur

        # yön: funding pozitifse short-perp/long-spot (+funding kazanılır),
        # negatifse ters pozisyon → işaret yumuşatılmış funding ile aynı
        side = np.sign(smooth_annual).where(active, 0.0).fillna(0.0)

        n_active = active.sum(axis=1).replace(0, np.nan)
        w = (active.astype(float)).div(n_active, axis=0).fillna(0.0) * p["max_gross"]

        held = w.shift(1).fillna(0.0)
        side_held = side.shift(1).fillna(0.0)

        # Baz mark-to-market: Δb. Pozisyonun günlük P&L'i taraf×(funding − Δb).
        if basis:
            B = pd.DataFrame({k: v for k, v in basis.items() if k in F.columns})
            B = B.reindex(F.index).ffill()
            dB = B.diff().reindex(columns=F.columns).fillna(0.0)
        else:
            print("[carry] ⚠️ baz (basis) verilmedi — sonuç İYİMSER; "
                  "Δbaz riski modellenmiyor.", flush=True)
            dB = pd.DataFrame(0.0, index=F.index, columns=F.columns)

        # `capture` < 1: kotasyon funding'in tamamı cebe girmez (bkz. __init__).
        pnl = (held * side_held * (F * p["capture"] - dB)).sum(axis=1)
        harvest = pnl

        # maliyet: İKİ bacak (spot + perp), her yeniden dengelemede
        turnover = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
        cost = turnover * p["leg_cost"] * 2

        ret = harvest - cost

        # Modellenmemiş kuyruk riski cezası — DETERMİNİSTİK sürüklenme olarak.
        #
        # İlk uygulama rastgele şok enjekte ediyordu (p olasılıkla −loss). Sabit
        # tohuma rağmen dizi uzunluğu değişince şoklar BAŞKA GÜNLERE düşüyordu:
        # aynı 2022-2025 dilimi, 2019'dan ısıtılınca Sharpe 0,82 · 2022'de
        # sıfırdan başlayınca 0,38. Ölçümün başlangıç tarihine bağlı olması
        # kabul edilemez. Beklenen kayıp oranı günlük sürüklenme olarak
        # uygulanır: aynı beklenen getiri cezası, sıfır yapay varyans.
        #
        # Kuyruk riskinin VARYANS etkisini görmek için `stochastic_tail=True`
        # senaryo modunu kullan — ama nokta tahmininde değil.
        gross_on = (held.abs().sum(axis=1) > 1e-6).astype(float)
        drag = p["tail_shock_prob"] * p["tail_shock_loss"]
        if p.get("stochastic_tail"):
            rng = np.random.default_rng(seed)
            shock = (rng.random(len(ret)) < p["tail_shock_prob"]).astype(float)
            ret = ret - pd.Series(shock * p["tail_shock_loss"], index=ret.index) * gross_on
        elif drag > 0:
            ret = ret - drag * gross_on

        # ------------------------------------------------- RİSK ÖLÇÜLEBİLİR Mİ?
        # HAYIR — ve bu, sonucun en önemli kısmıdır.
        #
        # Bu model carry'nin GETİRİSİNİ güvenilir ölçer (funding tahakkuku,
        # maliyet, baz MTM, kuyruk sürüklenmesi hepsi dahil): 2022-2025 olgun
        # piyasada net ≈ +%4,7/yıl. Bu sayı tüm varyantlarda kararlı.
        #
        # RİSKİ ise ölçemez. Ölçülen volatilite yıllık %1 çıkar ve Sharpe 4-8
        # görünür; böyle bir fırsat piyasada kalmaz. Modelde OLMAYAN riskler:
        #   • gün-içi baz dalgalanması (günlük kapanış farkı bunu gösteremez)
        #   • teminat/tasfiye riski (short perp bacağında fiyat fırlarsa)
        #   • borsa ve karşı taraf riski (FTX 2022'de delta-nötr de silindi)
        #   • yürütme kayması varyansı ve pozisyon limitleri
        #
        # Eksik varyansı sentetik gürültüyle "tamamlamak" DENENDİ ve REDDEDİLDİ:
        # bu, riski ölçmek değil uydurmaktır ve tabanı seçen kişi Sharpe'ı da
        # seçmiş olur. Doğru davranış: Sharpe'ı RAPORLAMAMAK.
        #
        # `risk_measurable` bayrağı aşağı akışa bunu bildirir; kabul kapısı ve
        # panel bu sleeve için Sharpe göstermemelidir. Gerçek ölçüm ancak gün-içi
        # baz + tasfiye verisi biriktiğinde mümkün olur (data/recorder.py bunu
        # bugünden itibaren topluyor).
        ret.attrs["risk_measurable"] = False
        ret.attrs["risk_note"] = (
            "Getiri ölçüldü, risk ölçülmedi: gün-içi baz, teminat/tasfiye ve "
            "karşı taraf riski günlük kapanış verisiyle modellenemez. "
            "Sharpe raporlanmamalıdır.")
        return ret.rename(self.name)
