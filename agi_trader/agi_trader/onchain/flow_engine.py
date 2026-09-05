"""
On-chain / Whale & Shark Akış Motoru (Blockchain / On-chain Analist rolü).

3-TIER MIMARI (her tier anahtar varsa otomatik aktifleşir):

  TIER 1 — Her zaman çalışır (anahtarsız, ücretsiz):
    1) Borsa büyük-işlem akışı (ccxt fetch_trades)
       - shark / whale / mega-whale tier sınıflandırması (USD büyüklüğüne göre)
       - whale CVD: büyük taker ALIM vs SATIM hacmi → net whale akışı
    2) Funding rate (ccxt): aşırı funding → long/short sıkışma riski
    3) Open Interest (ccxt): OI + fiyat ilişkisi → trend gücü / zayıflığı
    4) Order book imbalance + büyük likidite duvarları (ccxt)
    5) BTC mempool tıkanıklığı (mempool.space, anahtarsız)

  TIER 2 — Ücretsiz API anahtarlarıyla yükselir (opsiyonel):
    6) Etherscan: ETH/ERC-20 büyük transferler + borsa cüzdan etiketlemesi
    7) CoinGecko: piyasa trend, korku/açgözlülük proxy'si
    8) CoinMarketCap: piyasa verisi, ranking
    9) Blockchain.com: BTC zincir-üstü büyük transferler

  TIER 3 — Premium API anahtarlarıyla yükselir (paralı, opsiyonel):
    10) Whale Alert: entity-labeled zincir-üstü transferler

Veri çekilemezse fiyat/hacim türevinden PROXY üretir ve düşük güvenle oy verir.
Her tier kendi güven çarpanıyla skora katkı yapar.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.models import LayerVote

try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

# Whale/shark eşikleri (USD)
TIER_SHARK = 50_000
TIER_WHALE = 250_000
TIER_MEGA = 1_000_000

# ---------------------------------------------------------------------------
# Bilinen borsa cüzdan adresleri (Etherscan etiketlemesi için)
# ---------------------------------------------------------------------------
EXCHANGE_ADDRESSES: Dict[str, str] = {
    # Binance
    "0x28c6c06298d514db089934071355e5743bf21d60": "Binance 1",
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549": "Binance 2",
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": "Binance 3",
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f": "Binance 4",
    "0x9696f59e4d72e237be84ffd425dcad154bf96976": "Binance 5",
    # Coinbase
    "0x71660c4005ba85c37ccec55d0c4493e3fe3351a3": "Coinbase 1",
    "0x503828976d22510aad0201ac7ec88293211d23da": "Coinbase 2",
    "0xddfabcdc4d8ffc6d5beaf154f18b778f892a0740": "Coinbase 3",
    # Kraken
    "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0": "Kraken 1",
    "0xe853c56864a2ebe4576a807d26fdc4a0ada51919": "Kraken 2",
    # OKX
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": "OKX 1",
    "0x5041ed759dd4afc3a72b8192c143f72f4724081a": "OKX 2",
    # Bybit
    "0xf89d7b9c864f589bbf53a82105107622b35eaa40": "Bybit 1",
    # Gate.io
    "0x0d0707963952f2fba59dd06f2b425ace40b492fe": "Gate.io 1",
    # Kucoin
    "0x2b5634c42055806a59e910ded3249682a10daab8": "Kucoin 1",
    # Bitfinex
    "0x1151314c646ce4e0efd76d1af4760ae66a2fe30f": "Bitfinex 1",
    # Wintermute (market maker)
    "0xdbf5e9c5206d0db70a90108bf936da60221dc080": "Wintermute",
    # Jump Trading
    "0xf584f8728b874a6a5c7a8d4d387c9aae9172d621": "Jump Trading",
}

# USDC / USDT contract addresses (Ethereum mainnet)
USDC_ADDRESS = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
USDT_ADDRESS = "0xdac17f958d2ee523a2206206994597c13d831ec7"

# BTC whale eşiği
BTC_WHALE_THRESH = 100  # BTC

# ---------------------------------------------------------------------------
# Tier güven çarpanları
# ---------------------------------------------------------------------------
CONFIDENCE_TIER1 = 0.70  # ccxt canlı veri — yüksek güven
CONFIDENCE_TIER2 = 0.50  # ücretsiz API — orta güven
CONFIDENCE_TIER3 = 0.65  # premium API — orta-yüksek


class WhaleFlowEngine:
    def __init__(self, config, exchange_manager=None):
        self.config = config
        self.em = exchange_manager
        # Tier 2 keys
        self.etherscan_key = config.secret("ETHERSCAN_API_KEY")
        self.coingecko_key = config.secret("COINGECKO_API_KEY")
        self.coinmarketcap_key = config.secret("COINMARKETCAP_API_KEY")
        self.blockchaincom_key = config.secret("BLOCKCHAINCOM_API_KEY")
        # Tier 3 keys
        self.whale_alert_key = config.secret("WHALE_ALERT_API_KEY")

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------
    def _spot_client(self):
        if not self.em or not getattr(self.em, "exchanges", None):
            return None
        for pref in ("binance", "bybit", "okx", "kraken"):
            if pref in self.em.exchanges:
                return self.em.exchanges[pref]
        return next(iter(self.em.exchanges.values()), None)

    @staticmethod
    def _perp_symbol(symbol: str) -> str:
        base, quote = symbol.split("/")
        return f"{base}/{quote}:{quote}"

    def _get(self, url: str, params: dict = None, timeout: int = 15,
             headers: dict = None) -> Optional[dict]:
        """Güvenli HTTP GET. Hata durumunda None."""
        if not _HAS_REQUESTS:
            return None
        try:
            r = requests.get(url, params=params or {}, timeout=timeout,
                             headers=headers or {})
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    @staticmethod
    def _classify_address(addr: str) -> str:
        """Adresi borsa/kurumsal olarak etiketle."""
        al = addr.lower()
        if al in EXCHANGE_ADDRESSES:
            return f"borsa:{EXCHANGE_ADDRESSES[al]}"
        for known_al, label in EXCHANGE_ADDRESSES.items():
            if known_al.lower() == al:
                return f"borsa:{label}"
        return "unknown"

    # ══════════════════════════════════════════════════════════════════
    # TIER 1 — Her zaman çalışır (anahtarsız, ücretsiz)
    # ══════════════════════════════════════════════════════════════════

    # ----------------------------------------------------- 1) büyük işlem akışı
    def large_trade_flow(self, symbol: str) -> Optional[Dict]:
        ex = self._spot_client()
        if ex is None:
            return None
        try:
            trades = ex.fetch_trades(symbol, limit=1000)
        except Exception:
            return None
        if not trades:
            return None

        shark_buy = shark_sell = whale_buy = whale_sell = mega_buy = mega_sell = 0.0
        counts = {"shark": 0, "whale": 0, "mega": 0}
        biggest = []
        for t in trades:
            cost = float(t.get("cost") or (t.get("price", 0) * t.get("amount", 0)))
            side = t.get("side", "")
            if cost < TIER_SHARK:
                continue
            tier = "mega" if cost >= TIER_MEGA else "whale" if cost >= TIER_WHALE else "shark"
            counts[tier] += 1
            if tier == "mega":
                mega_buy += cost if side == "buy" else 0
                mega_sell += cost if side == "sell" else 0
            elif tier == "whale":
                whale_buy += cost if side == "buy" else 0
                whale_sell += cost if side == "sell" else 0
            else:
                shark_buy += cost if side == "buy" else 0
                shark_sell += cost if side == "sell" else 0
            biggest.append((cost, side))

        total_buy = shark_buy + whale_buy + mega_buy
        total_sell = shark_sell + whale_sell + mega_sell
        net = total_buy - total_sell
        weighted_net = ((mega_buy - mega_sell) * 3 + (whale_buy - whale_sell) * 2 +
                        (shark_buy - shark_sell)) / (
            (mega_buy + mega_sell) * 3 + (whale_buy + whale_sell) * 2 +
            (shark_buy + shark_sell) + 1e-9)
        biggest.sort(reverse=True)
        return {
            "tier": 1,
            "counts": counts,
            "total_buy_usd": total_buy,
            "total_sell_usd": total_sell,
            "net_usd": net,
            "flow_score": float(np.clip(weighted_net, -1, 1)),
            "biggest_trades": biggest[:3],
            "n_large": sum(counts.values()),
        }

    # ----------------------------------------------------- 2) funding
    def funding(self, symbol: str) -> Optional[float]:
        ex = self._spot_client()
        if ex is None or not ex.has.get("fetchFundingRate"):
            return None
        try:
            fr = ex.fetch_funding_rate(self._perp_symbol(symbol))
            return float(fr.get("fundingRate"))
        except Exception:
            return None

    # ----------------------------------------------------- 3) open interest
    def open_interest(self, symbol: str) -> Optional[float]:
        ex = self._spot_client()
        if ex is None or not ex.has.get("fetchOpenInterest"):
            return None
        try:
            oi = ex.fetch_open_interest(self._perp_symbol(symbol))
            return float(oi.get("openInterestAmount") or oi.get("openInterestValue") or 0)
        except Exception:
            return None

    # ----------------------------------------------------- 4) order book
    def book_imbalance(self, symbol: str) -> Optional[Dict]:
        ex = self._spot_client()
        if ex is None:
            return None
        try:
            ob = ex.fetch_order_book(symbol, limit=50)
        except Exception:
            return None
        bid_vol = sum(b[1] for b in ob["bids"][:25])
        ask_vol = sum(a[1] for a in ob["asks"][:25])
        imb = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
        bid_wall = max(ob["bids"][:25], key=lambda x: x[1], default=[0, 0])
        ask_wall = max(ob["asks"][:25], key=lambda x: x[1], default=[0, 0])
        return {"imbalance": float(imb), "bid_wall": bid_wall, "ask_wall": ask_wall}

    # ----------------------------------------------------- 5) BTC mempool
    def btc_mempool(self) -> Optional[Dict]:
        return self._get("https://mempool.space/api/mempool", timeout=10)

    # ══════════════════════════════════════════════════════════════════
    # TIER 2 — Ücretsiz API anahtarları (opsiyonel)
    # ══════════════════════════════════════════════════════════════════

    # ----------------------------------------------------- 6) Etherscan
    def etherscan_large_transfers(self, symbol: str) -> Optional[Dict]:
        """ETH ve ERC-20 stablecoin büyük transferlerini tarar."""
        if not (self.etherscan_key and _HAS_REQUESTS):
            return None

        coin = symbol.split("/")[0].upper()
        results = {"eth_transfers": [], "stable_transfers": [], "exchange_flows": {}}

        # ETH büyük transferler (son 1000 bloktan sample)
        try:
            # Son blok numarasını al
            bn = self._get(
                "https://api.etherscan.io/api",
                {"module": "proxy", "action": "eth_blockNumber",
                 "apikey": self.etherscan_key},
            )
            if bn and bn.get("result"):
                latest = int(bn["result"], 16)
                # Son 500 bloktaki işlemleri tara (sadece birkaç blok sample)
                for block_hex in [hex(latest - i) for i in range(0, 500, 100)]:
                    block = self._get(
                        "https://api.etherscan.io/api",
                        {"module": "proxy", "action": "eth_getBlockByNumber",
                         "tag": block_hex, "boolean": "true",
                         "apikey": self.etherscan_key},
                    )
                    if not block or "result" not in block:
                        continue
                    txs = (block["result"].get("transactions") or [])[:20]
                    for tx in txs:
                        val_eth = int(tx.get("value", "0"), 16) / 1e18
                        if val_eth < 50:  # min 50 ETH
                            continue
                        from_a = tx.get("from", "")
                        to_a = tx.get("to", "")
                        from_tag = self._classify_address(from_a)
                        to_tag = self._classify_address(to_a)
                        rec = {
                            "value_eth": round(val_eth, 2),
                            "value_usd": round(val_eth * 2000, 0),  # yaklaşık
                            "from": from_a[:10] + "...",
                            "to": to_a[:10] + "...",
                            "from_label": from_tag,
                            "to_label": to_tag,
                        }
                        results["eth_transfers"].append(rec)
                        # Borsa akış takibi
                        if "borsa" in from_tag:
                            tag = from_tag.split(":")[1] if ":" in from_tag else from_tag
                            results["exchange_flows"].setdefault("out_" + tag, 0)
                            results["exchange_flows"]["out_" + tag] += val_eth
                        if "borsa" in to_tag:
                            tag = to_tag.split(":")[1] if ":" in to_tag else to_tag
                            results["exchange_flows"].setdefault("in_" + tag, 0)
                            results["exchange_flows"]["in_" + tag] += val_eth
        except Exception:
            pass

        # Stablecoin büyük transferler (USDC)
        if coin in ("ETH", "BTC", "SOL") and results["eth_transfers"]:
            try:
                usdc_tx = self._get(
                    "https://api.etherscan.io/api",
                    {"module": "account", "action": "tokentx",
                     "contractaddress": USDC_ADDRESS, "page": 1, "offset": 20,
                     "sort": "desc", "apikey": self.etherscan_key},
                )
                if usdc_tx and usdc_tx.get("result"):
                    for tx in usdc_tx["result"]:
                        val = int(tx.get("value", "0")) / 1e6
                        if val < 1_000_000:  # min $1M USDC
                            continue
                        from_a = tx.get("from", "")
                        to_a = tx.get("to", "")
                        results["stable_transfers"].append({
                            "token": "USDC",
                            "value_usd": round(val, 0),
                            "from_label": self._classify_address(from_a),
                            "to_label": self._classify_address(to_a),
                        })
            except Exception:
                pass

        n_eth = len(results["eth_transfers"])
        n_stable = len(results["stable_transfers"])
        total = n_eth + n_stable
        if total == 0:
            return None

        # Net akış yönü: borsaya giriş (+) = potansiyel satış, borsadan çıkış (-) = potansiyel alım
        exchange_net = 0.0
        for flow_key, val in results.get("exchange_flows", {}).items():
            if flow_key.startswith("in_"):
                exchange_net += val  # borsaya giriş → satış baskısı → negatif
            else:
                exchange_net -= val  # borsadan çıkış → alım → pozitif

        flow_score = float(np.tanh(-exchange_net / 1000))  # normalize

        results["n_transfers"] = total
        results["exchange_net_eth"] = round(exchange_net, 2)
        results["flow_score"] = flow_score
        results["tier"] = 2
        return results

    # ----------------------------------------------------- 7) CoinGecko
    def coingecko_market(self) -> Optional[Dict]:
        """CoinGecko piyasa duyarlılığı (korku/açgözlülük proxy'si)."""
        if not _HAS_REQUESTS:
            return None

        headers = {}
        base = "https://api.coingecko.com/api/v3"
        if self.coingecko_key:
            headers["x-cg-demo-api-key"] = self.coingecko_key
            base = "https://pro-api.coingecko.com/api/v3"

        try:
            # Global market data (BTC dominance, total market cap değişimi)
            global_data = self._get(f"{base}/global", timeout=15)
            # Trending coins
            trending = self._get(f"{base}/search/trending", timeout=15)
        except Exception:
            return None

        result = {"tier": 2}
        if global_data and "data" in global_data:
            gd = global_data["data"]
            mcap_change = gd.get("market_cap_change_percentage_24h_usd", 0)
            btc_dom = gd.get("market_cap_percentage", {}).get("btc", 50)
            result["market_cap_change_24h"] = mcap_change
            result["btc_dominance"] = btc_dom
            # BTC dominance yüksek + mcap düşüş → risk-off (ayı)
            # BTC dominance düşük + mcap yükseliş → risk-on (boğa)
            fear_greed = float(np.tanh((mcap_change / 5) - (btc_dom - 50) / 10))
            result["fear_greed_proxy"] = round(fear_greed, 3)

        if trending and "coins" in trending:
            top_coins = [c["item"]["name"] for c in trending["coins"][:10]]
            result["trending_coins"] = top_coins
            result["trending_count"] = len(top_coins)

        if "fear_greed_proxy" not in result:
            result["fear_greed_proxy"] = 0.0

        return result

    # ----------------------------------------------------- 8) CoinMarketCap
    def coinmarketcap_fear_greed(self) -> Optional[Dict]:
        """CoinMarketCap Fear & Greed Index."""
        if not (self.coinmarketcap_key and _HAS_REQUESTS):
            return None
        data = self._get(
            "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest",
            headers={"X-CMC_PRO_API_KEY": self.coinmarketcap_key},
            timeout=15,
        )
        if not data or "data" not in data:
            return None
        try:
            d = data["data"]
            result = {
                "tier": 2,
                "btc_dominance": d.get("btc_dominance"),
                "eth_dominance": d.get("eth_dominance"),
                "total_mcap": d.get("quote", {}).get("USD", {}).get("total_market_cap"),
            }
        except Exception:
            return None
        return result

    # ----------------------------------------------------- 9) Blockchain.com
    def blockchain_btc_large_txs(self) -> Optional[Dict]:
        """BTC zincir-üstü büyük işlemler (Blockchain.com API)."""
        if not _HAS_REQUESTS:
            return None
        # Blockchain.com'un ücretsiz WebSocket'i veya REST API'si
        # API key opsiyonel, yoksa da temel endpoint çalışır
        try:
            params = {"format": "json"}
            if self.blockchaincom_key:
                params["api_code"] = self.blockchaincom_key
            # Son bloktaki büyük işlemleri al (unconfirmed + latest block)
            latest = self._get(
                "https://blockchain.info/latestblock", params=params, timeout=10
            )
            if not latest:
                return None

            # Son blok hash ile raw block al
            block_hash = latest.get("hash")
            if not block_hash:
                return None

            raw = self._get(
                f"https://blockchain.info/rawblock/{block_hash}", timeout=10
            )
            if not raw or "tx" not in raw:
                return None

            whale_txs = []
            for tx in raw["tx"][:200]:
                # Toplam output değerini hesapla (BTC)
                total_out = sum(o.get("value", 0) for o in tx.get("out", [])) / 1e8
                if total_out < BTC_WHALE_THRESH:
                    continue
                whale_txs.append({
                    "value_btc": round(total_out, 2),
                    "value_usd": round(total_out * 60000, 0),  # yaklaşık
                    "n_outputs": len(tx.get("out", [])),
                    "size_bytes": tx.get("size", 0),
                })

            if not whale_txs:
                return None

            total_whale_btc = sum(t["value_btc"] for t in whale_txs)
            # Çok büyük BTC hareketi → genelde borsa/OTC → nötr veya hafif ayı
            flow_score = float(np.tanh((total_whale_btc / BTC_WHALE_THRESH - 1) * -0.5))

            return {
                "tier": 2,
                "n_whale_txs": len(whale_txs),
                "total_btc": round(total_whale_btc, 2),
                "avg_btc": round(total_whale_btc / len(whale_txs), 2),
                "flow_score": flow_score,
            }
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════
    # TIER 3 — Premium (paralı, opsiyonel)
    # ══════════════════════════════════════════════════════════════════

    # ----------------------------------------------------- 10) Whale Alert
    def whale_alert(self, symbol: str) -> Optional[List[Dict]]:
        """Whale Alert API — entity-labeled zincir-üstü transferler."""
        if not (_HAS_REQUESTS and self.whale_alert_key):
            return None
        coin = symbol.split("/")[0].upper()
        try:
            import time
            r = requests.get(
                "https://api.whale-alert.io/v1/transactions",
                params={
                    "api_key": self.whale_alert_key,
                    "min_value": 1_000_000,
                    "start": int(time.time()) - 3600,
                },
                timeout=15,
            )
            if r.status_code == 200:
                txs = r.json().get("transactions", [])
                # Filtrele: sadece ilgili coin
                filtered = []
                for tx in txs:
                    sym = (tx.get("symbol") or "").upper()
                    if sym == coin or coin in ("BTC", "ETH") and sym in ("BTC", "ETH"):
                        filtered.append({
                            "symbol": sym,
                            "amount": tx.get("amount"),
                            "amount_usd": tx.get("amount_usd"),
                            "from_owner": tx.get("from", {}).get("owner", "unknown"),
                            "to_owner": tx.get("to", {}).get("owner", "unknown"),
                            "blockchain": tx.get("blockchain"),
                        })
                return filtered if filtered else None
        except Exception:
            return None
        return None


# ══════════════════════════════════════════════════════════════════════
# BİRLEŞİK ON-CHAIN OYLAMA (3-tier skorlama)
# ══════════════════════════════════════════════════════════════════════

def onchain_vote(engine: "WhaleFlowEngine", symbol: str, df: pd.DataFrame) -> LayerVote:
    """Tüm on-chain/whale sinyallerini 3-tier ağırlıklı tek oya indirir."""
    reasons: List[str] = []
    tier_scores: Dict[str, List[float]] = {"tier1": [], "tier2": [], "tier3": []}
    tier_reasons: Dict[str, List[str]] = {"tier1": [], "tier2": [], "tier3": []}
    detail: Dict = {"live": False, "tiers_active": []}

    # ── TIER 1 ──────────────────────────────────────────────────────
    t1_active = False

    flow = engine.large_trade_flow(symbol)
    if flow:
        t1_active = True
        detail["flow"] = {k: flow[k] for k in ("counts", "net_usd", "flow_score", "n_large")}
        tier_scores["tier1"].append(flow["flow_score"] * 1.5)
        c = flow["counts"]
        side = "ALIM" if flow["net_usd"] > 0 else "SATIM"
        tier_reasons["tier1"].append(
            f"[T1] Büyük işlem akışı: {c['mega']} mega / {c['whale']} whale / "
            f"{c['shark']} shark → net {side} ${abs(flow['net_usd'])/1e6:.2f}M "
            f"(skor {flow['flow_score']:+.2f})"
        )

    fr = engine.funding(symbol)
    if fr is not None:
        t1_active = True
        detail["funding_rate"] = fr
        if fr > 0.0005:
            tier_scores["tier1"].append(-0.4)
            tier_reasons["tier1"].append(
                f"[T1] Funding aşırı pozitif (%{fr*100:.4f}) → long sıkışma riski (ayı)"
            )
        elif fr < -0.0005:
            tier_scores["tier1"].append(0.4)
            tier_reasons["tier1"].append(
                f"[T1] Funding aşırı negatif (%{fr*100:.4f}) → short sıkışma (boğa)"
            )
        else:
            tier_reasons["tier1"].append(f"[T1] Funding nötr (%{fr*100:.4f})")

    oi = engine.open_interest(symbol)
    if oi is not None and len(df) > 2:
        t1_active = True
        detail["open_interest"] = oi
        price_chg = df["close"].pct_change().iloc[-1]
        if price_chg > 0:
            tier_scores["tier1"].append(0.2)
            tier_reasons["tier1"].append(
                f"[T1] OI {oi:,.0f} + fiyat ↑ → boğa pozisyon birikimi"
            )
        elif price_chg < 0:
            tier_scores["tier1"].append(-0.2)
            tier_reasons["tier1"].append(
                f"[T1] OI {oi:,.0f} + fiyat ↓ → ayı pozisyon birikimi"
            )

    book = engine.book_imbalance(symbol)
    if book:
        t1_active = True
        detail["book_imbalance"] = round(book["imbalance"], 3)
        tier_scores["tier1"].append(book["imbalance"] * 0.6)
        bias = "alıcı" if book["imbalance"] > 0 else "satıcı"
        tier_reasons["tier1"].append(
            f"[T1] Emir defteri dengesizliği {book['imbalance']:+.2f} ({bias} ağırlıklı)"
        )

    mp = engine.btc_mempool()
    if mp and symbol.startswith("BTC"):
        detail["btc_mempool_count"] = mp.get("count")
        tier_reasons["tier1"].append(
            f"[T1] BTC mempool: {mp.get('count', '?'):,} bekleyen işlem"
        )

    if t1_active:
        detail["tiers_active"].append("T1")

    # ── TIER 2 ──────────────────────────────────────────────────────
    t2_active = False

    eth_xfer = engine.etherscan_large_transfers(symbol)
    if eth_xfer:
        t2_active = True
        detail["etherscan"] = {
            "n_transfers": eth_xfer.get("n_transfers", 0),
            "exchange_net_eth": eth_xfer.get("exchange_net_eth", 0),
        }
        if eth_xfer.get("flow_score") is not None:
            tier_scores["tier2"].append(eth_xfer["flow_score"] * 1.0)
        tier_reasons["tier2"].append(
            f"[T2] Etherscan: {eth_xfer.get('n_transfers', 0)} büyük ETH/ERC-20 transfer "
            f"(borsa net: {eth_xfer.get('exchange_net_eth', 0):+.1f} ETH)"
        )

    cg = engine.coingecko_market()
    if cg:
        t2_active = True
        detail["coingecko"] = {
            "fear_greed_proxy": cg.get("fear_greed_proxy"),
            "trending_count": cg.get("trending_count", 0),
        }
        tier_scores["tier2"].append(cg.get("fear_greed_proxy", 0) * 0.8)
        trend_coins = cg.get("trending_coins", [])
        mcap_chg = cg.get("market_cap_change_24h", 0)
        tier_reasons["tier2"].append(
            f"[T2] CoinGecko: korku/açgözlülük {cg.get('fear_greed_proxy', 0):+.2f} | "
            f"mcap %24s %{mcap_chg:+.2f} | trend: {', '.join(trend_coins[:5])}"
        )

    cmc = engine.coinmarketcap_fear_greed()
    if cmc:
        t2_active = True
        detail["coinmarketcap"] = {
            "btc_dominance": cmc.get("btc_dominance"),
            "eth_dominance": cmc.get("eth_dominance"),
        }
        btc_dom = cmc.get("btc_dominance", 50)
        # BTC dominance > 55 → altcoin zayıf, risk-off
        dom_score = float(np.tanh((50 - btc_dom) / 10)) * 0.4
        tier_scores["tier2"].append(dom_score)
        tier_reasons["tier2"].append(
            f"[T2] CoinMarketCap: BTC dom %{btc_dom:.1f} | ETH dom %{cmc.get('eth_dominance', 0):.1f}"
        )

    btc_txs = engine.blockchain_btc_large_txs()
    if btc_txs and symbol.startswith("BTC"):
        t2_active = True
        detail["blockchaincom"] = {
            "n_whale_txs": btc_txs.get("n_whale_txs", 0),
            "total_btc": btc_txs.get("total_btc", 0),
        }
        tier_scores["tier2"].append(btc_txs.get("flow_score", 0) * 0.7)
        tier_reasons["tier2"].append(
            f"[T2] Blockchain.com: {btc_txs.get('n_whale_txs', 0)} whale işlem "
            f"({btc_txs.get('total_btc', 0):.1f} BTC)"
        )

    if t2_active:
        detail["tiers_active"].append("T2")

    # ── TIER 3 ──────────────────────────────────────────────────────
    t3_active = False

    wa = engine.whale_alert(symbol)
    if wa:
        t3_active = True
        detail["whale_alert_txs"] = len(wa)
        # Whale Alert transferlerinin yön analizi
        exchange_to_unknown = sum(
            1 for t in wa
            if t.get("from_owner", "").lower() in ("exchange", "binance", "coinbase", "kraken", "okx")
            and t.get("to_owner", "").lower() == "unknown"
        )
        unknown_to_exchange = sum(
            1 for t in wa
            if t.get("from_owner", "").lower() == "unknown"
            and t.get("to_owner", "").lower() in ("exchange", "binance", "coinbase", "kraken", "okx")
        )
        # unknown→exchange = potansiyel satış (ayı)
        # exchange→unknown = cüzdana çekme (boğa)
        wa_score = float(np.tanh((exchange_to_unknown - unknown_to_exchange) * 0.5))
        tier_scores["tier3"].append(wa_score * 1.0)
        tier_reasons["tier3"].append(
            f"[T3] Whale Alert: {len(wa)} büyük zincir-üstü transfer "
            f"(borsa→cüzdan: {exchange_to_unknown}, cüzdan→borsa: {unknown_to_exchange})"
        )

    if t3_active:
        detail["tiers_active"].append("T3")

    # ── SKOR HESAPLAMA ──────────────────────────────────────────────
    all_scores = []
    all_weights = []

    # Tier 1: yüksek güven
    for s in tier_scores["tier1"]:
        all_scores.append(s)
        all_weights.append(CONFIDENCE_TIER1)

    # Tier 2: orta güven
    for s in tier_scores["tier2"]:
        all_scores.append(s)
        all_weights.append(CONFIDENCE_TIER2)

    # Tier 3: premium güven
    for s in tier_scores["tier3"]:
        all_scores.append(s)
        all_weights.append(CONFIDENCE_TIER3)

    if all_scores:
        # Ağırlıklı ortalama
        total_w = sum(all_weights)
        score = float(np.clip(sum(s * w for s, w in zip(all_scores, all_weights)) / total_w, -1, 1))
        # Güven: aktif tier sayısına ve en yüksek tier'a göre
        max_confidence = CONFIDENCE_TIER1
        if tier_scores["tier3"]:
            max_confidence = max(max_confidence, CONFIDENCE_TIER3)
        if tier_scores["tier2"]:
            max_confidence = max(max_confidence, CONFIDENCE_TIER2)
        confidence = float(min(0.85, 0.4 + 0.06 * len(all_scores) + max_confidence * 0.3))
    else:
        # PROXY fallback (hiçbir canlı veri yok)
        from ..agents.extra_layers import onchain_proxy_vote
        return onchain_proxy_vote(df)

    # Tüm tier'ların sebeplerini sırayla birleştir
    all_reasons = (
        tier_reasons["tier3"] + tier_reasons["tier2"] + tier_reasons["tier1"]
    )

    return LayerVote(
        name="onchain",
        score=score,
        confidence=confidence,
        reasons=all_reasons[:10],
        detail=detail,
    )
