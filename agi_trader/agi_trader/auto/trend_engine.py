"""
Trend-Takip Otonom Motoru (PAPER) — FAZ6.

OOS-doğrulanmış günlük trend-takip stratejisini (strategies/trend_follow.py)
canlı/paper işleme bağlar. Sinyal-başı değil, GÜNLÜK HEDEF-AĞIRLIK yeniden
dengeleme modeli:
  Her gün: her parite için hedef ağırlık = trend_position (vol-targeted) / N.
  Paper portföy bu ağırlıklara çekilir; turnover'da gerçekçi maliyet düşülür.

GÜVENLİK: yalnız paper. Canlı emir yok.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..strategies.trend_follow import to_daily, trend_position, current_signal


# ---------------------------------------------------------------------------
# RISK RAYLARI (2026-09-05) - ayni sinyal, farkli risk istahi.
#
# Olculmus temel (48 gercek paper gunu): ort +%0,0884/gun - sapma %0,647 -
# Sharpe 2,61 - maksimum dusus %2,08. Kaldirac L bu ucunu de L ile carpar
# (Sharpe DEGISMEZ). Yani "daha yuksek kazanc" bu katmanda ancak DAHA YUKSEK
# DUSUS satin alinarak elde edilir - olculen supurme:
#     L=1 -> yillik ~%37,  DD %2,1      L=3 -> ~%146, DD %6,2
#     L=5 -> ~%315, DD %10,4            L=11,3 -> ~%1362, DD %23,1 (= %1/gun)
# UYARI: %1/gun icin gereken 11,3x yalniz GERCEKLESEN Sharpe 2,61 kaliciysa
# gecerli. Backtest beklentisi 1,36 idi; onunla ayni hedef 21,7x ister ve
# %40+ dusus demektir. Bu yuzden raylar 3x-5x bandinda tutuldu.
# max_lev ile max_exposure FARKLI seylerdir ve bu fark 2026-09-05'te canli olcumle bulundu:
# `max_lev` HAM hedeflerin carpanidir, ham hedefler ise (varliklarin bir kismi piyasa disi,
# kalanlar vol-olcekli oldugu icin) tipik olarak yalnizca ~0,42 toplar. Yani `max_lev = 16`
# gercekte 16x maruziyet degil ~6,7x maruziyet verir - uc rayin ucu de tavanina erken carpip
# reklam ettikleri bandi TUTMUYORDU. Guvenlik siniri portfoyun GERCEK maruziyeti olmali:
# `max_exposure` = sum(agirliklar) tavani. `max_lev` yalnizca ikincil bir emniyet kalir.
# Bandlar temel rayin olculmus maruziyetine (~1,065x) gore ifade edilir; supurmedeki
# 3,2x / 5,33x / 15,94x carpanlari sirasiyla 3,4 / 5,7 / 17,0 maruziyete denk gelir.
RISK_TRACKS: Dict[str, Dict[str, float]] = {
    "base":       {"target_vol": 0.15, "max_lev": 2.5,  "max_exposure": 3.0,
                   "dd_soft": 0.10, "dd_hard": 0.20, "dd_kill": 0.30},
    "aggressive": {"target_vol": 0.45, "max_lev": 12.0, "max_exposure": 3.4,
                   "dd_soft": 0.10, "dd_hard": 0.20, "dd_kill": 0.35},
    "extreme":    {"target_vol": 0.75, "max_lev": 20.0, "max_exposure": 5.7,
                   "dd_soft": 0.12, "dd_hard": 0.25, "dd_kill": 0.45},
    # MAKSIMUM: %1/GUN hedefinin olculmus fiyati. 15,9x kaldirac, finansman dahil, GERCEKLESEN
    # Sharpe 2,61'in kalici oldugu varsayimiyla gunluk net ~%1 verir. Bedeli:
    #   gunluk sapma %10,3 · olculen 48 gunde maks dusus ~%35 · cokus stresinde (3 gun -4 sigma)
    #   -%47 ve %49 dusus · TAM KAYIP icin dayanakta -%6,3 (9,7 sigma) yeter.
    # Backtest beklentisi (Sharpe 1,36) dogruysa bu ray %1/gun VERMEZ: ayni kapilarla yillik
    # -%88 uretir ve Monte Carlo'da medyan sonuc 0,77x (yani ZARAR) cikar.
    # DD kapilari olculerek secildi (%12/25/45 ... %20/35/55 supurmesi): %15/30/50, cokus
    # korumasinin cogunu tutup olculen senaryoda %452/yil biraktigi icin.
    "max":        {"target_vol": 2.25, "max_lev": 45.0, "max_exposure": 17.0,
                   "dd_soft": 0.15, "dd_hard": 0.30, "dd_kill": 0.50},
}


class TrendTrader:
    def __init__(self, config, pairs: Optional[List[str]] = None, initial: float = 10_000.0,
                 track: str = "base", target_vol: Optional[float] = None,
                 max_lev: Optional[float] = None, dd_soft: Optional[float] = None,
                 dd_hard: Optional[float] = None, dd_kill: Optional[float] = None,
                 max_exposure: Optional[float] = None):
        self.config = config
        self.pairs = pairs or list(config.symbols)
        self.initial = float(initial)
        rc = config.get("risk", {})
        self.cost = float(rc.get("fee_taker", 0.0004)) + float(rc.get("slippage_base", 0.00015))
        tr = dict(RISK_TRACKS.get(track, RISK_TRACKS["base"]))
        self.track = str(track)
        # config yalniz "base" rayinin varsayilanini ezebilir; agresif raylar acikca secilir.
        if track == "base":
            tr["target_vol"] = float(rc.get("trend_target_vol", tr["target_vol"]))
            tr["max_lev"] = float(rc.get("trend_max_lev", tr["max_lev"]))
        if target_vol is not None: tr["target_vol"] = float(target_vol)
        if max_lev is not None:    tr["max_lev"] = float(max_lev)
        if max_exposure is not None: tr["max_exposure"] = float(max_exposure)
        if dd_soft is not None:    tr["dd_soft"] = float(dd_soft)
        if dd_hard is not None:    tr["dd_hard"] = float(dd_hard)
        if dd_kill is not None:    tr["dd_kill"] = float(dd_kill)
        self.target_vol_a = float(tr["target_vol"])                      # yillik
        self.target_vol_d = self.target_vol_a / np.sqrt(365)             # gunluk
        self.max_lev = float(tr["max_lev"])
        self.max_exposure = float(tr.get("max_exposure", tr["max_lev"]))
        self.dd_soft, self.dd_hard, self.dd_kill = float(tr["dd_soft"]), float(tr["dd_hard"]), float(tr["dd_kill"])
        # KALDIRAC FINANSMAN MALIYETI (yillik). 1x'in ustundeki kisim borctur ve bedava
        # degildir: kripto perp funding tarihsel olarak ~%11/yil, ETF/marj ~%6-12/yil.
        # Saymazsak agresif raylar kayirilir - 4,2x kaldiracta yilda ~%32'lik gercek bir
        # gider gorunmez olur ve "yuksek getiri" olcumu sahte cikar.
        self.finance_rate = float(rc.get("leverage_finance_rate", 0.10))
        self.weights: Dict[str, float] = {s: 0.0 for s in self.pairs}   # equity kesri (long)
        self.equity = self.initial
        self.peak_equity = self.initial
        self.dd_locked = False            # dd_kill asildi -> nakit; dd_hard altina donene dek kilitli
        # IFLAS: 16x kaldiracta dayanakta -%6,3 ozsermayeyi SIFIRA indirir. Model bunu
        # yakalamazsa equity NEGATIFE duser ve sistem "eksi sermayeyle" islem yapmaya devam
        # eder - gercek hesap tasfiye olurdu. Dusuk kaldiracta hic ortaya cikmayan bu durum
        # MAKSIMUM rayinda gercek bir olasilik oldugu icin acikca modellenir.
        self.ruined = False
        self.last_rebalance: Optional[str] = None
        self.last_close: Dict[str, float] = {}
        self.last_signals: Dict[str, Dict] = {}
        self.history: List[Dict] = []
        self.recovered_from_nan = False

    # ---------------------------------------------------- hedef ağırlıklar
    def compute_targets(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Her parite için bugünkü hedef ağırlık (0..1/N). Eşit-ağırlık + vol-target."""
        n = max(1, len(self.pairs))
        targets: Dict[str, float] = {}
        for s in self.pairs:
            df = data.get(s)
            if df is None or len(df) < 10:
                targets[s] = 0.0; continue
            sig = current_signal(df, vol_target=True)
            targets[s] = float(sig["target"]) / n          # portföy payı
        # portföy hedef-vol kaldıraç: güncel ağırlıkların son-60g portföy vol'ünü
        # tahmin et, hedef vol'e ölçekle (kaldıraç tavanıyla). OOS: CAGR ~2×.
        lev = self._target_vol_leverage(data, targets) * self.dd_multiplier()
        out = {s: v * lev for s, v in targets.items()}
        # MARUZIYET TAVANI: gercek risk sum(agirliklar)'dir, carpan degil. Ham hedefler
        # gunden gune degistigi icin (kac varlik piyasada, vol ne kadar) ayni carpan cok
        # farkli maruziyetler uretir - tavan bu yuzden maruziyetin kendisine konur.
        tot = sum(out.values())
        if tot > self.max_exposure > 0:
            k = self.max_exposure / tot
            out = {s: v * k for s, v in out.items()}
        return out

    # ------------------------------------------------ dususe bagli kaldirac
    def dd_multiplier(self) -> float:
        """Tepe-alti dusus buyudukce kaldiraci KISAR.

        Yuksek kaldiracta sabit vol hedefi yetmez: vol hedefi oynakligi olcer,
        SERMAYE KAYBINI olcmez. 5x kaldiracta arka arkaya birkac kotu gun, vol
        henuz artmadan sermayenin beste birini alir. Bu yuzden dusus kendisi bir
        kisicidir: dd_soft'a kadar tam, dd_hard'da yari, dd_kill'de SIFIR.
        Kilit histerezislidir - dd_kill'i asan ray, dd_hard'in ALTINA donene
        kadar nakitte kalir (esigin etrafinda gidip gelip maliyet uretmesin)."""
        if self.ruined or self.equity <= 0.0:
            return 0.0                              # tasfiye edilmis ray pozisyon acamaz
        peak = float(self.peak_equity or self.initial)
        if not math.isfinite(peak) or peak <= 0 or not math.isfinite(self.equity):
            return 1.0
        dd = max(0.0, 1.0 - self.equity / peak)
        if self.dd_locked:
            if dd < self.dd_hard:
                self.dd_locked = False
            else:
                return 0.0
        if dd >= self.dd_kill:
            self.dd_locked = True
            return 0.0
        if dd <= self.dd_soft:
            return 1.0
        if dd <= self.dd_hard:
            f = (dd - self.dd_soft) / max(1e-9, self.dd_hard - self.dd_soft)
            return float(1.0 - 0.5 * f)                    # 1,0 -> 0,5
        f = (dd - self.dd_hard) / max(1e-9, self.dd_kill - self.dd_hard)
        return float(max(0.0, 0.5 * (1.0 - f)))            # 0,5 -> 0,0

    def drawdown_pct(self) -> float:
        peak = float(self.peak_equity or self.initial)
        if not math.isfinite(peak) or peak <= 0 or not math.isfinite(self.equity):
            return 0.0
        return round(max(0.0, 1.0 - self.equity / peak) * 100, 2)

    def _target_vol_leverage(self, data: Dict[str, pd.DataFrame], targets: Dict[str, float]) -> float:
        try:
            rets = {}
            for s in self.pairs:
                df = data.get(s)
                if df is not None and len(df) > 65 and targets.get(s, 0) != 0:
                    rets[s] = to_daily(df)["close"].pct_change().tail(60)
            if not rets:
                return 1.0
            R = pd.DataFrame(rets).fillna(0)
            w = pd.Series({s: targets.get(s, 0.0) for s in R.columns})
            port_ret = (R * w).sum(axis=1)                 # güncel ağırlıklı portföy getirisi
            est_vol = float(port_ret.std())
            if est_vol < 1e-9:
                return 1.0
            return float(np.clip(self.target_vol_d / est_vol, 0.0, self.max_lev))
        except Exception:
            return 1.0

    def signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Açıklamalı bugünkü sinyaller (UI/rapor için)."""
        out = {}
        for s in self.pairs:
            df = data.get(s)
            out[s] = current_signal(df, vol_target=True) if df is not None and len(df) > 10 \
                else {"target": 0.0, "in_market": False, "reason": "veri yok"}
        return out

    # ---------------------------------------------------- canlı yeniden dengele
    def rebalance(self, data: Dict[str, pd.DataFrame], date_str: Optional[str] = None) -> Dict:
        """Bir günlük yeniden dengeleme (paper): hedeflere çek, turnover maliyeti uygula."""
        targets = self.compute_targets(data)
        turnover = sum(abs(targets.get(s, 0.0) - self.weights.get(s, 0.0)) for s in self.pairs)
        if not math.isfinite(turnover):
            turnover = 0.0
        if not math.isfinite(self.equity):
            self.equity = self._last_finite_equity()
        cost = turnover * self.cost * self.equity
        cost = cost if math.isfinite(cost) else 0.0
        self.equity -= cost
        self.weights = targets
        ev = {"date": date_str, "equity": round(self.equity, 2),
              "targets": {s: round(v, 4) for s, v in targets.items()},
              "invested_pct": round(sum(targets.values()) * 100, 1),
              "turnover": round(turnover, 3), "cost": round(cost, 2)}
        self.history.append(ev)
        self.last_rebalance = date_str
        return ev

    def mark(self, day_returns: Dict[str, float]) -> Dict:
        """Gün sonu piyasa hareketiyle equity'yi işaretle (ağırlık × getiri).

        NaN KORUMASI (2026-09-04 canlı arıza): bir varlığın fiyatı çekilemezse getirisi NaN olur;
        NaN tek bir çarpımla equity'yi kalıcı olarak bozar (`equity *= NaN`) ve diske yazılır —
        45 günlük paper kaydı bu yüzden okunamaz hâle geldi. Artık NaN getiriler HESABA KATILMAZ,
        o varlığın o günkü katkısı 0 sayılır ve durum `skipped` olarak raporlanır."""
        pnl = 0.0
        skipped = []
        for s in self.pairs:
            w = self.weights.get(s, 0.0)
            r = day_returns.get(s)
            if r is None or not math.isfinite(float(r)):
                if abs(w) > 1e-9:
                    skipped.append(s)
                continue
            pnl += w * float(r)
        if not math.isfinite(pnl):
            pnl = 0.0
        # Gun boyunca TASINAN kaldiracin finansman gideri (agirliklar bu gunun agirliklari;
        # rebalance() bundan SONRA calisir, dolayisiyla dogru gune yazilir).
        lev = sum(float(v) for v in self.weights.values() if math.isfinite(float(v)))
        borrowed = max(0.0, lev - 1.0)
        fin = borrowed * self.finance_rate / 365.0
        if not math.isfinite(fin):
            fin = 0.0
        pnl -= fin
        yeni = self.equity * (1 + pnl)
        if math.isfinite(yeni) and yeni <= 0.0:
            # Tasfiye: sermaye tukendi. Negatif ozsermayeyle "islem" yapilamaz.
            self.equity = 0.0
            self.ruined = True
            self.dd_locked = True
            return {"pnl": pnl, "skipped": skipped, "finance_pct": round(fin * 100.0, 5),
                    "leverage": round(lev, 4), "ruined": True}
        self.equity = yeni
        if not math.isfinite(self.equity):          # son çare: hiçbir koşulda NaN diske yazılmaz
            self.equity = self._last_finite_equity()
        if not math.isfinite(float(self.peak_equity or 0)):
            self.peak_equity = self.equity
        self.peak_equity = max(float(self.peak_equity or self.initial), float(self.equity))
        return {"pnl": pnl, "skipped": skipped,
                "finance_pct": round(fin * 100.0, 5), "leverage": round(lev, 4)}

    def _last_finite_equity(self) -> float:
        """Geçmişteki SON sağlam özsermaye (kurtarma). Yoksa başlangıç sermayesi."""
        for ev in reversed(self.history):
            e = ev.get("equity")
            if isinstance(e, (int, float)) and math.isfinite(e):
                return float(e)
        return float(self.initial)

    def step(self, data: Dict[str, pd.DataFrame], date_str: Optional[str] = None) -> Dict:
        """Bir günlük TAM adım (kalıcı paper): önce dünkü ağırlıklarla piyasa getirisini
        işaretle, sonra bugünkü hedeflere yeniden dengele. Günlük cron/döngü için."""
        closes = {}
        missing = []
        for s in self.pairs:
            df = data.get(s)
            v = float(df["close"].iloc[-1]) if (df is not None and len(df)) else float("nan")
            if math.isfinite(v) and v > 0:
                closes[s] = v
            else:
                missing.append(s)                      # veri gelmedi ya da NaN — sessizce yutma
        # 1) mark-to-market (önceki kapanışa göre) — yalnız iki ucu da sonlu olanlar
        marked = {}
        if self.last_close:
            day_ret = {}
            for s in self.pairs:
                a, b = closes.get(s), self.last_close.get(s)
                if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)) or b <= 0:
                    continue
                day_ret[s] = a / b - 1
            marked = self.mark(day_ret)
        # 2) yeniden dengele
        ev = self.rebalance(data, date_str)
        # eksik fiyatlı varlığın SON BİLİNEN kapanışı korunur (silinirse ertesi gün de işaretlenemez)
        for s in missing:
            if s in self.last_close and math.isfinite(self.last_close[s]):
                closes[s] = self.last_close[s]
        self.last_close = closes
        self.last_signals = self.signals(data)
        ev.update({"return_pct": round((self.equity / self.initial - 1) * 100, 2),
                   "missing_prices": missing,
                   "marked_skipped": marked.get("skipped", []),
                   "data_ok": not missing})
        return ev

    # ---------------------------------------------------- kalıcılık
    def to_dict(self) -> Dict:
        return {"pairs": self.pairs, "initial": self.initial, "equity": self.equity,
                "track": self.track, "peak_equity": self.peak_equity, "dd_locked": self.dd_locked,
                "ruined": self.ruined,
                "risk": {"target_vol": self.target_vol_a, "max_lev": self.max_lev,
                         "max_exposure": self.max_exposure,
                         "dd_soft": self.dd_soft, "dd_hard": self.dd_hard, "dd_kill": self.dd_kill},
                "weights": self.weights, "last_close": self.last_close,
                "last_signals": self.last_signals,
                "last_rebalance": self.last_rebalance, "history": self.history[-400:]}

    def load_state(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            self.history = list(d.get("history", []))          # kurtarma için önce geçmiş
            eq = float(d.get("equity", self.initial))
            if not math.isfinite(eq):                          # bozuk durum → son sağlam güne dön
                eq = self._last_finite_equity()
                self.recovered_from_nan = True
            self.equity = eq
            self.weights = {k: float(v) for k, v in d.get("weights", {}).items()
                            if math.isfinite(float(v))}
            self.last_close = {k: float(v) for k, v in d.get("last_close", {}).items()
                               if math.isfinite(float(v)) and float(v) > 0}
            self.last_rebalance = d.get("last_rebalance")
            pk = d.get("peak_equity")
            try:
                pk = float(pk)
            except Exception:
                pk = None
            # Eski durum dosyalarinda tepe yok -> gecmisten yeniden kur (kaldirac
            # kisicisi ilk gunden dogru calissin diye; yoksa dusus 0 sanilir).
            if pk is None or not math.isfinite(pk) or pk <= 0:
                pk = max([float(e.get("equity")) for e in self.history
                          if isinstance(e.get("equity"), (int, float)) and math.isfinite(float(e.get("equity")))]
                         + [float(self.initial), float(self.equity)])
            self.peak_equity = max(pk, float(self.equity))
            self.dd_locked = bool(d.get("dd_locked", False))
            self.ruined = bool(d.get("ruined", False)) or self.equity <= 0.0
            return True
        except Exception:
            return False

    def save_state(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                              encoding="utf-8")

    def status(self) -> Dict:
        return {"mode": "trend_follow_paper", "pairs": self.pairs,
                "track": self.track,
                "equity": round(self.equity, 2),
                "return_pct": round((self.equity / self.initial - 1) * 100, 2),
                "weights": {s: round(v, 4) for s, v in self.weights.items()},
                "invested_pct": round(sum(self.weights.values()) * 100, 1),
                "peak_equity": round(float(self.peak_equity or self.initial), 2),
                "drawdown_pct": self.drawdown_pct(),
                "dd_multiplier": round(self.dd_multiplier(), 3),
                "dd_locked": self.dd_locked,
                "ruined": self.ruined,
                "finance_rate_pct": round(self.finance_rate * 100, 2),
                "borrowed_pct": round(max(0.0, sum(self.weights.values()) - 1.0) * 100, 1),
                "risk": {"target_vol_pct": round(self.target_vol_a * 100, 1),
                         "max_lev": self.max_lev,
                         "max_exposure": self.max_exposure,
                         "dd_soft_pct": round(self.dd_soft * 100, 1),
                         "dd_hard_pct": round(self.dd_hard * 100, 1),
                         "dd_kill_pct": round(self.dd_kill * 100, 1)},
                "last_rebalance": self.last_rebalance}

    # ---------------------------------------------------- entegrasyon doğrulama
    def backtest(self, price_daily: Dict[str, pd.Series]) -> Dict:
        """Motorun kendi mantığıyla portföy backtest'i (parity: portfolio_trend ≈ Sharpe 1.05).
        price_daily: {symbol: günlük close serisi}."""
        idx = None
        for s in self.pairs:
            if s in price_daily:
                idx = price_daily[s].index if idx is None else idx.union(price_daily[s].index)
        if idx is None:
            return {"error": "veri yok"}
        n = max(1, len(self.pairs))
        strat = pd.Series(0.0, index=idx)
        for s in self.pairs:
            c = price_daily.get(s)
            if c is None:
                continue
            c = c.reindex(idx).ffill()
            r = c.pct_change().fillna(0)
            pos = trend_position(pd.DataFrame({"close": c}), vol_target=True) / n
            p = pos.shift(1).fillna(0)
            turn = p.diff().abs().fillna(0)
            strat = strat.add(p * r - turn * self.cost, fill_value=0.0)
        eq = (1 + strat).cumprod()
        total = float((eq.iloc[-1] - 1) * 100)
        dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
        sharpe = float(strat.mean() / (strat.std() + 1e-12) * np.sqrt(365))
        cagr = float((eq.iloc[-1] ** (365 / max(1, len(eq))) - 1) * 100)
        return {"total_return_pct": round(total, 1), "cagr_pct": round(cagr, 1),
                "sharpe": round(sharpe, 2), "max_drawdown_pct": round(dd, 1), "days": len(eq)}
