"""
Çoklu-Parite Tarama Sistemi + Twitter Sosyal Entegrasyon.

Likit pariteleri toplu tarar; her parite için hızlı bir skor üretir:
  teknik + SMC + formasyon + alış/satış baskısı + (opsiyonel) whale akışı
  + Twitter kritik hesaplardan SOSYAL ISI (coin başına mention + nedensel yön).

Sosyal entegrasyon: tüm parite taranırken Twitter takip süreci tek seferde
çalışır (scan_social_heat) ve sonuç her paritenin temel coin'ine eşlenir.
Böylece tarama, hem grafik hem de "kritik hesapların ne konuştuğunu" birlikte
gösterir; yükseliş/azalış öngören tweet'ler doğrudan taramayı etkiler.

Hız için derin (torch) topluluk taramada KULLANILMAZ; tek tıkla derin analiz
için panelden ilgili pariteye "Analiz Et" denir.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import re

import numpy as np

from ..analysis import compute_all_indicators, technical_vote, detect_patterns, smc_vote
from ..analysis.forecast import buy_sell_pressure

# Taramadan dışlanacak stablecoin / sarmalanmış baz varlıklar (gürültü)
EXCLUDE_BASE = {
    "USDC", "USDT", "BUSD", "TUSD", "FDUSD", "DAI", "USDD", "USDP", "USD1",
    "PYUSD", "EUR", "EURI", "GUSD", "USTC", "AEUR", "XUSD", "FRAX",
}


class MultiPairScanner:
    def __init__(self, orchestrator):
        self.orch = orchestrator

    # ----------------------------------------------------- evren
    def universe(self, top: int = 25) -> List[str]:
        em = self.orch.data
        if not getattr(em, "live_enabled", False):
            return self.orch.config.symbols
        ex = next(iter(em.exchanges.values()), None)
        if ex is None:
            return self.orch.config.symbols
        try:
            tickers = ex.fetch_tickers()
        except Exception:
            return self.orch.config.symbols
        rows = []
        for sym, t in tickers.items():
            if "/USDT" not in sym or ":" in sym:
                continue
            base = sym.split("/")[0].upper()
            if base in EXCLUDE_BASE:
                continue
            # yalnızca standart sembol harfleri (isimli/garip tokenları ele)
            if not re.fullmatch(r"[A-Z0-9]{2,15}", base):
                continue
            qv = t.get("quoteVolume") or 0
            if qv and qv > 2_000_000:
                rows.append((sym, qv))
        rows.sort(key=lambda x: -x[1])
        return [s for s, _ in rows[:top]] or self.orch.config.symbols

    # ----------------------------------------------------- tek parite hızlı skor
    def _quick(self, symbol: str, tf: str, social: Dict, include_onchain: bool) -> Optional[Dict]:
        try:
            df = self.orch.data.fetch_ohlcv(symbol, tf, limit=200)
        except Exception:
            return None
        if df is None or len(df) < 60:
            return None

        ind = compute_all_indicators(df)
        tv = technical_vote(df, ind)
        sv = smc_vote(df)
        _, pv = detect_patterns(df)

        agg = tv.score * 0.45 + sv.score * 0.25 + pv.score * 0.30
        buy, sell, plabel = buy_sell_pressure(df, agg)

        # sosyal ısı (temel coin)
        coin = symbol.split("/")[0].upper()
        soc = social.get(coin, {})
        soc_score = soc.get("score", 0.0)
        mentions = soc.get("mentions", 0)

        onchain_score = 0.0
        onchain_note = ""
        if include_onchain:
            try:
                flow = self.orch.whale.large_trade_flow(symbol)
                if flow:
                    onchain_score = flow["flow_score"]
                    c = flow["counts"]
                    onchain_note = f"{c['mega']}M/{c['whale']}W/{c['shark']}S net ${flow['net_usd']/1e6:.1f}M"
            except Exception:
                pass

        # birleşik tarama skoru (sosyal + on-chain dahil)
        w_tech, w_soc, w_oc = (0.6, 0.25, 0.15) if include_onchain else (0.75, 0.25, 0.0)
        combined = agg * w_tech + soc_score * w_soc + onchain_score * w_oc
        combined = float(np.clip(combined, -1, 1))

        direction = "LONG" if combined > 0.12 else "SHORT" if combined < -0.12 else "FLAT"
        # tarama güveni: katman hizalanması
        sigs = [tv.score, sv.score, pv.score, soc_score, onchain_score]
        aligned = sum(1 for s in sigs if (s > 0) == (combined > 0) and abs(s) > 0.1)
        confidence = round(min(0.95, 0.35 + 0.12 * aligned), 3)

        return {
            "symbol": symbol,
            "score": round(combined, 3),
            "direction": direction,
            "confidence": confidence,
            "price": round(float(df["close"].iloc[-1]), 6),
            "buy_pct": buy, "sell_pct": sell, "pressure": plabel,
            "technical": round(tv.score, 2),
            "smc": round(sv.score, 2),
            "pattern": round(pv.score, 2),
            "social_score": round(soc_score, 3),
            "social_mentions": mentions,
            "social_reasons": soc.get("reasons", []),
            "onchain_score": round(onchain_score, 3),
            "onchain_note": onchain_note,
        }

    # ----------------------------------------------------- toplu tarama
    def scan(self, symbols: Optional[List[str]] = None, tf: Optional[str] = None,
             top: int = 25, include_onchain: bool = False,
             min_abs_score: float = 0.0) -> Dict:
        tf = tf or self.orch.primary_tf
        syms = symbols or self.universe(top)

        # Twitter takip süreci: tek seferde sosyal ısı (çoklu-parite entegrasyonu)
        social = {}
        try:
            social = self.orch.twitter.scan_social_heat()
        except Exception:
            social = {}

        results: List[Dict] = []
        for sym in syms:
            r = self._quick(sym, tf, social, include_onchain)
            if r and abs(r["score"]) >= min_abs_score:
                results.append(r)

        # En güçlü mutlak skora göre sırala
        results.sort(key=lambda x: -abs(x["score"]))

        trending = sorted(
            ({"coin": c, **v} for c, v in social.items()),
            key=lambda x: -abs(x.get("score", 0)) * (1 + x.get("mentions", 0)))[:10]

        return {
            "timeframe": tf,
            "scanned": len(results),
            "social_live": self.orch.twitter.live,
            "social_coins": len(social),
            "trending": trending,
            "results": results,
        }
