"""
Broker — tek arayüz, üç mod:

  paper    sanal bakiye, dolum = son fiyat, komisyon simüle (videodaki "sanal 5.000 $")
  testnet  borsanın sandbox'ı (ccxt set_sandbox_mode) — sahte para, gerçek API
  live     gerçek hesap — YALNIZ kasa'dan gelen İŞLEM kapsamlı anahtarla

GÜVENLİK SÖZLEŞMESİ
  • Bu sınıf anahtarı yalnız ctor'da alır, hiçbir metot onu döndürmez/loglamaz.
  • Her emir `clientOrderId` taşır → aynı döngü iki kez emir gönderemez (idempotent).
  • Her emir `max_order_usdt` tavanıyla kırpılır; tavan aşan istek REDDEDİLİR
    (sessizce küçültülmez — kullanıcı ne istediğini bilmeli).
  • Borsa minimum notional/precision kuralları uygulanır; sağlanamıyorsa emir yok.
  • Para çekme ile ilgili HİÇBİR ccxt metodu bu sınıfta çağrılmaz.

Public veri (fiyat/OHLCV) anahtarsız istemciyle çekilir; özel istemci yalnız
bakiye/emir için kurulur ve yalnız testnet/live modunda vardır.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

MODES = ("paper", "testnet", "live")
# Kâğıt modda borsanın market tablosu YÜKLENMEZ (MEXC ~2.500 market, on MB'larca RSS),
# bu yüzden muhafazakâr bir asgari emir varsayılır. Ama bu varsayım ÖLÇÜMÜ ÇARPITIR:
# `max_order_usdt` de 10 $ olduğunda hiçbir kısmi/basamaklı kâr alımı YAPILAMAZ — dilim
# her zaman asgarinin altında kalır. (200 canlı işlemin yalnız 7'sinde kısmi kâr
# alınabilmesinin sebeplerinden biri budur.) Borsanın gerçek asgarisi ortam değişkeniyle
# verilebilir; VARSAYILAN DEĞİŞMEDİ — bilinçli bir karar olmadan gevşemesin.
MIN_NOTIONAL_FALLBACK = float(os.environ.get("CRYPTOMIND_PAPER_MIN_NOTIONAL", "10.0"))


def make_client_order_id(user_id: int, exchange_id: str, symbol: str,
                         cycle: int, side: str) -> str:
    """Deterministik, ≤32 karakter, yalnız [a-z0-9]. Aynı (kullanıcı, borsa, parite,
    döngü, yön) → aynı kimlik; borsa ikinciyi 'duplicate' diye reddeder."""
    raw = f"{user_id}|{exchange_id}|{symbol}|{cycle}|{side}".encode("utf-8")
    return "cm" + hashlib.sha1(raw).hexdigest()[:24]


class BrokerError(RuntimeError):
    pass


class Broker:
    def __init__(self, exchange_id: str, mode: str = "paper",
                 creds: Optional[Dict[str, str]] = None,
                 market_type: str = "spot", fee_bps: float = 10.0,
                 maker_fee_bps: Optional[float] = None,
                 max_order_usdt: float = 100.0,
                 paper_capital: float = 5000.0,
                 client_factory: Optional[Callable[..., Any]] = None,
                 timeout_ms: int = 15000):
        if mode not in MODES:
            raise BrokerError(f"bilinmeyen mod: {mode}")
        self.exchange_id = exchange_id
        self.mode = mode
        self.market_type = market_type
        self.fee_bps = float(fee_bps)                      # taker (piyasa emri)
        self.maker_fee_bps = float(fee_bps if maker_fee_bps is None else maker_fee_bps)
        self.max_order_usdt = float(max_order_usdt)
        self._creds = dict(creds or {})
        self._factory = client_factory or self._ccxt_factory
        self._timeout = timeout_ms
        self._public = None
        self._private = None
        self._markets: Dict[str, Dict] = {}
        # paper defteri
        self.paper_cash = float(paper_capital)
        self.paper_holdings: Dict[str, float] = {}     # symbol -> base miktar (+long/−short)
        self.last_prices: Dict[str, float] = {}
        self.orders: List[Dict] = []
        self._seen_ids: set = set()

        if mode in ("testnet", "live") and not (self._creds.get("apiKey") and
                                               self._creds.get("secret")):
            raise BrokerError(f"{mode} modu için apiKey + secret gerekli")

    # ---------------------------------------------------------------- istemci
    @staticmethod
    def _ccxt_factory(exchange_id: str, params: Dict):
        import ccxt
        try:
            cls = getattr(ccxt, exchange_id)
        except AttributeError:
            raise BrokerError(f"ccxt'de bilinmeyen borsa: {exchange_id}")
        return cls(params)

    def _pub(self):
        if self._public is None:
            self._public = self._factory(self.exchange_id, {
                "enableRateLimit": True, "timeout": self._timeout,
                "options": {"defaultType": "future" if self.market_type != "spot" else "spot"}})
        return self._public

    def _priv(self):
        if self.mode == "paper":
            raise BrokerError("paper modunda özel istemci yok")
        if self._private is None:
            params = {"enableRateLimit": True, "timeout": self._timeout,
                      "apiKey": self._creds["apiKey"], "secret": self._creds["secret"],
                      "options": {"defaultType": "future" if self.market_type != "spot" else "spot"}}
            if self._creds.get("password"):
                params["password"] = self._creds["password"]
            ex = self._factory(self.exchange_id, params)
            if self.mode == "testnet":
                try:
                    ex.set_sandbox_mode(True)
                except Exception as e:
                    raise BrokerError(f"{self.exchange_id} sandbox desteklemiyor: "
                                      f"{type(e).__name__}")
            self._private = ex
        return self._private

    def close(self) -> None:
        """Anahtar materyalini bellekten düşür."""
        self._creds = {}
        self._private = None

    # ---------------------------------------------------------------- piyasa
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 120) -> pd.DataFrame:
        rows = self._pub().fetch_ohlcv(symbol, timeframe, limit=limit)
        if not rows:
            raise BrokerError(f"{symbol} OHLCV boş")
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts")
        self.last_prices[symbol] = float(df["close"].iloc[-1])
        return df

    def fetch_price(self, symbol: str) -> float:
        t = self._pub().fetch_ticker(symbol)
        px = float(t.get("last") or t.get("close") or 0.0)
        if px <= 0:
            raise BrokerError(f"{symbol} fiyat yok")
        self.last_prices[symbol] = px
        return px

    def fetch_book_top(self, symbol: str, depth: int = 20) -> Dict:
        """Spread (bps) + ±%2 bant derinliği (USDT) — maliyet modeli için."""
        ob = self._pub().fetch_order_book(symbol, limit=depth)
        bids, asks = ob.get("bids") or [], ob.get("asks") or []
        if not bids or not asks:
            return {"spread_bps": 0.0, "bid_depth_usd": 0.0, "ask_depth_usd": 0.0}
        bb, ba = float(bids[0][0]), float(asks[0][0])
        mid = (bb + ba) / 2.0
        spread_bps = (ba - bb) / mid * 1e4 if mid > 0 else 0.0
        bd = sum(float(p) * float(q) for p, q in bids if float(p) >= mid * 0.98)
        ad = sum(float(p) * float(q) for p, q in asks if float(p) <= mid * 1.02)
        return {"spread_bps": round(spread_bps, 3), "bid_depth_usd": round(bd, 2),
                "ask_depth_usd": round(ad, 2), "bid": bb, "ask": ba}

    def fetch_trades(self, symbol: str, limit: int = 200) -> list:
        """Son işlemler (taker yönü + miktar) — CVD / saldırgan alım-satım için. Hata → boş liste (fail-safe)."""
        try:
            return self._pub().fetch_trades(symbol, limit=limit) or []
        except Exception:
            return []

    def fetch_tickers(self) -> Dict:
        """Tüm tickerlar (tek çağrı) — üçgen arbitraj gölge taraması için."""
        return self._pub().fetch_tickers() or {}

    def market_rules(self, symbol: str) -> Dict:
        if symbol in self._markets:
            return self._markets[symbol]
        rules = {"min_notional": MIN_NOTIONAL_FALLBACK, "amount_precision": 6,
                 "min_amount": 0.0}
        # PAPER: borsanın tüm market tablosunu (MEXC ~2.500 market, on MB'larca RSS)
        # yüklemek gereksiz — muhafazakâr varsayılanlar yeter. Testnet/canlıda gerçek
        # kurallar okunur (asgari notional / hassasiyet emir reddine yol açar).
        if self.mode == "paper":
            self._markets[symbol] = rules
            return rules
        try:
            ex = self._pub()
            if not getattr(ex, "markets", None):
                ex.load_markets()
            m = ex.market(symbol)
            lim = (m.get("limits") or {})
            cost_min = ((lim.get("cost") or {}).get("min"))
            amt_min = ((lim.get("amount") or {}).get("min"))
            prec = (m.get("precision") or {}).get("amount")
            if cost_min:
                rules["min_notional"] = float(cost_min)
            if amt_min:
                rules["min_amount"] = float(amt_min)
            if prec is not None:
                rules["amount_precision"] = prec
        except Exception:
            pass
        self._markets[symbol] = rules
        return rules

    def round_amount(self, symbol: str, amount: float) -> float:
        rules = self.market_rules(symbol)
        prec = rules.get("amount_precision", 6)
        try:
            ex = self._pub()
            if getattr(ex, "markets", None) and symbol in ex.markets:
                return float(ex.amount_to_precision(symbol, amount))
        except Exception:
            pass
        if isinstance(prec, int):
            f = 10 ** prec
            return math.floor(amount * f) / f
        if isinstance(prec, float) and prec > 0:     # tick-size biçimi
            return math.floor(amount / prec) * prec
        return float(amount)

    # ---------------------------------------------------------------- hesap
    def fetch_balance_usdt(self) -> Dict:
        if self.mode == "paper":
            return {"free": round(self.paper_cash, 2), "total": round(self.paper_cash, 2),
                    "source": "paper"}
        b = self._priv().fetch_balance()
        usdt = (b.get("USDT") or {})
        return {"free": float(usdt.get("free") or 0.0),
                "total": float(usdt.get("total") or 0.0), "source": self.mode}

    def fetch_positions(self) -> Dict[str, float]:
        """symbol -> taban miktar. Spot'ta bakiye, vadelide pozisyon."""
        if self.mode == "paper":
            return {s: q for s, q in self.paper_holdings.items() if abs(q) > 0}
        ex = self._priv()
        out: Dict[str, float] = {}
        if self.market_type == "spot":
            b = ex.fetch_balance()
            for cur, v in (b.get("total") or {}).items():
                try:
                    q = float(v or 0.0)
                except (TypeError, ValueError):
                    continue
                if q > 0 and cur not in ("USDT", "BUSD", "USDC", "BNB"):
                    out[f"{cur}/USDT"] = q
            return out
        for p in ex.fetch_positions() or []:
            q = float(p.get("contracts") or 0.0)
            if q:
                side = -1.0 if str(p.get("side", "")).lower() == "short" else 1.0
                out[p.get("symbol")] = side * q
        return out

    # ---------------------------------------------------------------- emir
    def market_order(self, symbol: str, side: str, notional_usdt: float,
                     client_id: str, ref_price: Optional[float] = None,
                     reduce_only: bool = False, amount: Optional[float] = None) -> Dict:
        """Piyasa emri. Tavan aşımı → RET (reduce_only hariç: pozisyon KAPATMAK
        riski azaltır, tavan onu engellemez). Aynı client_id ikinci kez → RET.
        `amount` verilirse tam o miktar gönderilir (kapanışta dust kalmasın)."""
        side = side.lower()
        if side not in ("buy", "sell"):
            raise BrokerError(f"geçersiz yön: {side}")
        if not reduce_only and notional_usdt > self.max_order_usdt + 1e-9:
            raise BrokerError(f"emir tavanı aşıldı: {notional_usdt:.2f} > "
                              f"{self.max_order_usdt:.2f} USDT")
        if client_id in self._seen_ids:
            raise BrokerError(f"yinelenen emir kimliği: {client_id}")
        px = float(ref_price or self.last_prices.get(symbol) or 0.0)
        if px <= 0:
            px = self.fetch_price(symbol)
        rules = self.market_rules(symbol)
        if not reduce_only and notional_usdt < rules["min_notional"] - 1e-9:
            raise BrokerError(f"{symbol} asgari notional {rules['min_notional']} USDT "
                              f"(istenen {notional_usdt:.2f})")
        if amount is not None:
            amount = float(amount) if reduce_only else self.round_amount(symbol, float(amount))
        else:
            amount = self.round_amount(symbol, notional_usdt / px)
        if amount <= 0 or (not reduce_only and amount < rules.get("min_amount", 0.0)):
            raise BrokerError(f"{symbol} miktar hassasiyet altında")
        self._seen_ids.add(client_id)

        if self.mode == "paper":
            fee = notional_usdt * self.fee_bps / 1e4
            signed = amount if side == "buy" else -amount
            self.paper_holdings[symbol] = self.paper_holdings.get(symbol, 0.0) + signed
            if abs(self.paper_holdings[symbol]) < 1e-12:
                self.paper_holdings.pop(symbol, None)
            self.paper_cash -= (amount * px if side == "buy" else -amount * px) + fee
            rec = {"ok": True, "mode": "paper", "id": client_id, "client_id": client_id,
                   "symbol": symbol, "side": side, "amount": amount, "avg_price": px,
                   "filled_usdt": round(amount * px, 4), "fee_usdt": round(fee, 6),
                   "status": "closed", "ts": time.time()}
            self.orders.append(rec)
            return rec

        ex = self._priv()
        params: Dict[str, Any] = {}
        # ccxt borsaya göre doğru alanı seçer; Binance 'newClientOrderId' kabul eder
        params["clientOrderId"] = client_id
        if reduce_only and self.market_type != "spot":
            params["reduceOnly"] = True
        try:
            o = ex.create_order(symbol, "market", side, amount, None, params)
        except Exception as e:
            self._seen_ids.discard(client_id)      # gönderilemedi → tekrar denenebilir
            raise BrokerError(f"emir reddedildi: {type(e).__name__}: {str(e)[:160]}")
        filled = float(o.get("filled") or amount)
        avg = float(o.get("average") or o.get("price") or px)
        fee_usdt = 0.0
        fee = o.get("fee") or {}
        if fee and fee.get("cost") is not None:
            fee_usdt = float(fee["cost"])
            if str(fee.get("currency", "USDT")).upper() not in ("USDT", "USD", "BUSD"):
                fee_usdt *= avg if str(fee.get("currency")).upper() == symbol.split("/")[0].upper() else 1.0
        if fee_usdt == 0.0:
            fee_usdt = filled * avg * self.fee_bps / 1e4     # borsa döndürmediyse tahmin
        rec = {"ok": True, "mode": self.mode, "id": o.get("id"), "client_id": client_id,
               "symbol": symbol, "side": side, "amount": filled, "avg_price": avg,
               "filled_usdt": round(filled * avg, 4), "fee_usdt": round(fee_usdt, 6),
               "status": o.get("status", "?"), "ts": time.time()}
        self.orders.append(rec)
        return rec

    # ---------------------------------------------------------------- limit (maker)
    def limit_order(self, symbol: str, side: str, notional_usdt: float, price: float,
                    client_id: str, post_only: bool = True) -> Dict:
        """Maker emri. Paper'da BEKLEYEN kayıt döner; dolum `paper_limit_fill` ile
        bar low/high'ına göre kontrol edilir (muhafazakâr: fiyat limitin İÇİNDEN
        geçmeli). Testnet/canlıda borsaya post-only limit gönderilir."""
        side = side.lower()
        if side not in ("buy", "sell"):
            raise BrokerError(f"geçersiz yön: {side}")
        if notional_usdt > self.max_order_usdt + 1e-9:
            raise BrokerError(f"emir tavanı aşıldı: {notional_usdt:.2f} > {self.max_order_usdt:.2f} USDT")
        if client_id in self._seen_ids:
            raise BrokerError(f"yinelenen emir kimliği: {client_id}")
        rules = self.market_rules(symbol)
        if notional_usdt < rules["min_notional"] - 1e-9:
            raise BrokerError(f"{symbol} asgari notional {rules['min_notional']} USDT")
        amount = self.round_amount(symbol, notional_usdt / float(price))
        if amount <= 0:
            raise BrokerError(f"{symbol} miktar hassasiyet altında")
        self._seen_ids.add(client_id)
        rec = {"ok": True, "mode": self.mode, "id": client_id, "client_id": client_id,
               "symbol": symbol, "side": side, "amount": amount, "price": float(price),
               "notional": round(amount * float(price), 4), "status": "open",
               "type": "limit", "ts": time.time()}
        if self.mode == "paper":
            self.orders.append(rec)
            return rec
        ex = self._priv()
        params: Dict[str, Any] = {"clientOrderId": client_id}
        if post_only:
            params["postOnly"] = True
        try:
            o = ex.create_order(symbol, "limit", side, amount, float(price), params)
        except Exception as e:
            self._seen_ids.discard(client_id)
            raise BrokerError(f"limit emir reddedildi: {type(e).__name__}: {str(e)[:160]}")
        rec["id"] = o.get("id") or client_id
        rec["status"] = o.get("status", "open")
        self.orders.append(rec)
        return rec

    def paper_limit_fill(self, order: Dict, bar_low: float, bar_high: float) -> Optional[Dict]:
        """Paper: alış limiti barın low'u limitin ALTINA inerse, satış limiti high'ı
        limitin ÜSTÜNE çıkarsa dolar (limite dokunmak yetmez — kuyruk önceliği yok
        sayılır). Dolum fiyatı = limit; ücret = maker."""
        if order.get("status") != "open" or self.mode != "paper":
            return None
        px = float(order["price"])
        filled = (bar_low < px) if order["side"] == "buy" else (bar_high > px)
        if not filled:
            return None
        amount = float(order["amount"])
        fee = amount * px * self.maker_fee_bps / 1e4
        signed = amount if order["side"] == "buy" else -amount
        self.paper_holdings[order["symbol"]] = self.paper_holdings.get(order["symbol"], 0.0) + signed
        if abs(self.paper_holdings[order["symbol"]]) < 1e-12:
            self.paper_holdings.pop(order["symbol"], None)
        self.paper_cash -= (amount * px if order["side"] == "buy" else -amount * px) + fee
        order["status"] = "closed"
        return {**order, "avg_price": px, "filled_usdt": round(amount * px, 4),
                "fee_usdt": round(fee, 6), "status": "closed"}

    def order_status(self, order: Dict) -> Dict:
        """Testnet/canlı: borsadaki durumu oku; dolduysa dolum bilgisiyle döner."""
        if self.mode == "paper":
            return order
        try:
            o = self._priv().fetch_order(order["id"], order["symbol"])
        except Exception as e:
            return {**order, "error": f"{type(e).__name__}"}
        st = o.get("status", "open")
        out = {**order, "status": st}
        if st == "closed":
            avg = float(o.get("average") or o.get("price") or order["price"])
            filled = float(o.get("filled") or order["amount"])
            fee = o.get("fee") or {}
            fee_usdt = float(fee.get("cost") or 0.0) or filled * avg * self.maker_fee_bps / 1e4
            out.update({"avg_price": avg, "amount": filled, "filled_usdt": round(filled * avg, 4),
                        "fee_usdt": round(fee_usdt, 6)})
        return out

    def cancel_order(self, order: Dict) -> bool:
        if order.get("status") != "open":
            return False
        order["status"] = "canceled"
        if self.mode == "paper":
            return True
        try:
            self._priv().cancel_order(order["id"], order["symbol"])
            return True
        except Exception:
            return False

    # ---------------------------------------------------------------- durum
    def describe(self) -> Dict:
        return {"exchange": self.exchange_id, "mode": self.mode,
                "market_type": self.market_type, "fee_bps": self.fee_bps,
                "maker_fee_bps": self.maker_fee_bps,
                "max_order_usdt": self.max_order_usdt,
                "has_private": self._private is not None,
                "orders_sent": len(self.orders)}
