"""
STRATEJİ ARAŞTIRMA FABRİKASI — 250 kayıtlı hipotez, tek maliyetli OOS boru hattı.

Her kayıt: id · aile · ad · edge mekanizması · beklenen rejim · veri ihtiyacı · öncelik ·
uygulama durumu (kod sleeve'i / gölge modülü / veri yok / yalnız kayıt) · boru hattı aşaması.

DURUM SÖZLÜĞÜ (dürüst):
  IMPLEMENTED   kodda komiteye bağlı sleeve/rol/filtre/çıkış/tahsis olarak çalışıyor (paper)
  SHADOW        araştırma modülü sinyal üretiyor, emir YOK (spot'ta short/perp yok ya da kanıt yok)
  RESEARCH      veri mevcut, hipotez kayıtlı; kod yok — sıradaki adaylar
  DATA_NOT_WIRED veri ccxt ile çekilebilir ama koşucuya bağlı değil (OI, likidasyon, büyük işlem)
  NO_DATA       anahtarsız kaynak yok (zincir-üstü, DEX, sosyal hacim API) — aktifleştirilemez

Boru hattı: REGISTERED → LICENSE_CHECK → STATIC_REVIEW → LOGIC_EXTRACT → NORMALIZE → LOOKAHEAD_TEST
→ COST_MODEL → BACKTEST → PURGED_WALK_FORWARD → OOS → DSR_PBO → SHADOW → PAPER → QUALIFIED.
Kodda çalışan sleeve'ler PAPER aşamasındadır ve OOS/DSR kanıtı simülatörde BİRİKMEKTEDİR;
"QUALIFIED" yalnız lifecycle kapıları (OOS>0, CI>0, maliyet×2, DSR>0, PBO<0,5, n≥30) geçilince.
"""
from __future__ import annotations

from typing import Dict, List, Optional

PIPELINE = ["REGISTERED", "LICENSE_CHECK", "STATIC_REVIEW", "LOGIC_EXTRACT", "NORMALIZE", "LOOKAHEAD_TEST",
            "COST_MODEL", "BACKTEST", "PURGED_WALK_FORWARD", "OOS", "DSR_PBO", "SHADOW", "PAPER", "QUALIFIED"]
FAMILIES = {
    "A": "Trend / Momentum", "B": "Geri Çekilme / Trend Girişi", "C": "Kırılım / Genişleme",
    "D": "Başarısız Kırılım / Dönüş", "E": "Ortalamaya Dönüş", "F": "İstatistiksel Arbitraj / Göreli Değer",
    "G": "Funding / Vadeli / Baz", "H": "Saf Arbitraj", "I": "Emir Defteri / Mikroyapı",
    "J": "Likidite / Defter Olayları", "K": "Türev Pozisyonlanma", "L": "Zincir-üstü", "M": "Haber / Olay",
    "N": "Duygu / Dikkat", "O": "Oynaklık", "P": "Piyasa Yapıcılık", "Q": "Yürütme", "R": "Çıkış / Kâr Maksimizasyonu",
    "S": "Portföy / Meta",
}
# Sistemde bugün gerçekten akan veri
AVAILABLE_DATA = {"ohlcv", "book", "news", "social", "multi_venue", "tickers"}
OPTIONAL_DATA = {"funding": "CRYPTOMIND_RESEARCH_CARRY=1 ile ccxt swap funding (gölge)"}
WIRABLE_DATA = {"oi", "liquidations", "trades"}                 # ccxt public ile çekilebilir, koşucuya bağlı değil
NO_SOURCE_DATA = {"onchain", "dex", "social_volume", "perp_exec", "short_exec"}

# id|aile|ad|rejim|veri(virgül)|öncelik|uygulama(sleeve:key | shadow:mod | role:x | filter:x | exit:x | alloc:x | exec:x | -)
_RAW = """
1|A|Adaptive Multi-Timeframe Trend Following|TREND|ohlcv|P0|sleeve:adaptive_trend
2|A|Time-Series Momentum|TREND|ohlcv|P0|sleeve:momentum
3|A|Volatility-Scaled Momentum|TREND|ohlcv|P1|alloc:sizing_vol_target
4|A|EMA Slope Momentum|TREND|ohlcv|P1|sleeve:momentum
5|A|Dual Moving Average Trend|TREND|ohlcv|P2|-
6|A|Triple EMA Alignment|TREND|ohlcv|P2|-
7|A|Donchian Trend Following|TREND|ohlcv|P0|sleeve:donchian_breakout
8|A|ADX Trend Continuation|TREND|ohlcv|P1|sleeve:adaptive_trend
9|A|Relative Strength Momentum|TREND|ohlcv|P0|sleeve:rs_momentum
10|A|Cross-Sectional Momentum|ANY|ohlcv|P0|sleeve:rs_momentum
11|A|Momentum Acceleration|TREND|ohlcv|P0|-
12|A|Risk-Adjusted Momentum|TREND|ohlcv|P1|-
13|A|Residual Momentum (BTC-beta arındırılmış)|ANY|ohlcv|P2|-
14|A|Breakout Momentum|TREND|ohlcv|P0|sleeve:breakout
15|A|Trend Persistence Score|TREND|ohlcv|P1|sleeve:adaptive_trend
16|B|EMA20 Trend Pullback|TREND|ohlcv|P0|sleeve:pullback
17|B|EMA50 Trend Pullback|TREND|ohlcv|P1|-
18|B|VWAP Pullback Continuation|TREND|ohlcv|P0|sleeve:vwap_continuation
19|B|Anchored VWAP Pullback|TREND|ohlcv|P2|-
20|B|ATR Pullback|TREND|ohlcv|P1|sleeve:adaptive_trend
21|B|Structure Support Pullback|TREND|ohlcv|P1|sleeve:bos_retest
22|B|Breakout-Retest Continuation|TREND|ohlcv|P0|sleeve:bos_retest
23|B|Fibonacci Pullback (kanıt kapılı)|TREND|ohlcv|P2|-
24|B|Relative-Strength Pullback|TREND|ohlcv|P0|sleeve:adaptive_trend
25|B|Low-Volume Pullback + Volume Re-expansion|TREND|ohlcv|P1|-
26|C|Donchian Breakout|TREND|ohlcv|P0|sleeve:donchian_breakout
27|C|Bollinger Squeeze Breakout|SQUEEZE|ohlcv|P0|sleeve:squeeze_breakout
28|C|ATR Compression Breakout|SQUEEZE|ohlcv|P1|sleeve:squeeze_breakout
29|C|Keltner Squeeze|SQUEEZE|ohlcv|P2|-
30|C|Bollinger-Keltner Squeeze|SQUEEZE|ohlcv|P1|-
31|C|Range Expansion Breakout|SQUEEZE|ohlcv|P1|-
32|C|Volume-Surprise Breakout|TREND|ohlcv|P0|sleeve:breakout
33|C|New-High Breakout|TREND|ohlcv|P1|sleeve:donchian_breakout
34|C|New-Low Short Breakout|TREND_DOWN|ohlcv,short_exec|P2|-
35|C|Session Breakout (kripto seans eşdeğeri)|ANY|ohlcv|P2|-
36|C|Multi-Timeframe Resistance Break|TREND|ohlcv|P1|-
37|C|Volatility Regime Transition Breakout|SQUEEZE|ohlcv|P1|-
38|C|Compression → Expansion|SQUEEZE|ohlcv|P0|sleeve:squeeze_breakout
39|C|Breakout + Retest|TREND|ohlcv|P0|sleeve:bos_retest
40|C|Relative-Strength Breakout|TREND|ohlcv|P0|sleeve:rs_momentum
41|D|Failed Breakout Reversal|RANGE|ohlcv,short_exec|P0|sleeve:failed_breakout
42|D|Failed Breakdown Reversal|RANGE|ohlcv|P0|sleeve:failed_breakdown
43|D|Liquidity Sweep Long|ANY|ohlcv|P0|sleeve:sweep_reversal
44|D|Liquidity Sweep Short|ANY|ohlcv,short_exec|P1|-
45|D|Swing-Low Reclaim|RANGE|ohlcv|P0|sleeve:sweep_reversal
46|D|Swing-High Rejection|RANGE|ohlcv,short_exec|P1|-
47|D|VWAP Reclaim|RANGE|ohlcv|P1|sleeve:vwap_reversion
48|D|Support Undercut + Reclaim|RANGE|ohlcv|P0|sleeve:failed_breakdown
49|D|Resistance Overshoot + Failure|RANGE|ohlcv,short_exec|P1|sleeve:failed_breakout
50|D|Volume-Climax Reversal|VOLATILE|ohlcv|P1|-
51|E|Z-Score Mean Reversion|RANGE|ohlcv|P0|sleeve:dip
52|E|VWAP Mean Reversion|RANGE|ohlcv|P0|sleeve:vwap_reversion
53|E|Bollinger Mean Reversion|RANGE|ohlcv|P1|-
54|E|RSI Extreme + Regime Filter|RANGE|ohlcv|P0|sleeve:dip_moderate
55|E|Stochastic Reversion|RANGE|ohlcv|P2|-
56|E|Distance-from-EMA Reversion|RANGE|ohlcv|P1|-
57|E|ATR Extreme Reversion|VOLATILE|ohlcv|P1|-
58|E|Intraday Return Reversal|RANGE|ohlcv|P1|-
59|E|Multi-Timeframe Oversold Reversion|RANGE|ohlcv|P1|-
60|E|Multi-Timeframe Overbought Reversion|RANGE|ohlcv,short_exec|P2|-
61|E|Residual Mean Reversion|RANGE|ohlcv|P2|-
62|E|Regime-Adaptive Mean Reversion|RANGE|ohlcv|P0|sleeve:range_edge
63|E|Volatility-Normalized Reversion|RANGE|ohlcv|P1|sleeve:dip
64|E|Liquidity-Adjusted Reversion|RANGE|ohlcv,book|P1|-
65|E|Microprice Reversion|RANGE|book|P1|-
66|F|Engle-Granger Cointegration Pairs|ANY|ohlcv,short_exec|P0|shadow:pairs
67|F|Johansen Multi-Asset Cointegration|ANY|ohlcv,short_exec|P2|-
68|F|Kalman Dynamic Hedge Ratio|ANY|ohlcv,short_exec|P0|shadow:pairs
69|F|Ornstein-Uhlenbeck Spread Trading|ANY|ohlcv,short_exec|P0|shadow:pairs
70|F|Residual Z-Score Pairs|ANY|ohlcv,short_exec|P0|shadow:pairs
71|F|PCA Statistical Arbitrage|ANY|ohlcv,short_exec|P2|-
72|F|Factor-Neutral Residual Trading|ANY|ohlcv,short_exec|P2|-
73|F|BTC-Beta Neutral Pairs|ANY|ohlcv,short_exec|P1|shadow:pairs
74|F|ETH-Beta Neutral Pairs|ANY|ohlcv,short_exec|P1|shadow:pairs
75|F|Sector-Neutral Crypto Pairs|ANY|ohlcv,short_exec|P2|-
76|F|Stablecoin Relative Value|ANY|tickers|P2|-
77|F|Cross-Exchange Relative Value|ANY|multi_venue|P1|shadow:venue_compare
78|F|Basket-vs-Asset Mean Reversion|ANY|ohlcv,short_exec|P2|-
79|F|Multi-Asset Market-Neutral Basket|ANY|ohlcv,short_exec|P2|-
80|F|Convex-Optimization Statistical Arbitrage|ANY|ohlcv,short_exec|P2|-
81|G|Spot–Perpetual Funding Carry|ANY|funding,perp_exec|P0|shadow:carry
82|G|Cash-and-Carry Futures Basis|ANY|funding,perp_exec|P0|shadow:carry
83|G|Reverse Cash-and-Carry|ANY|funding,perp_exec,short_exec|P1|-
84|G|Cross-Exchange Funding Arbitrage|ANY|funding,multi_venue,perp_exec|P0|shadow:carry
85|G|Funding Dispersion|ANY|funding,multi_venue|P0|shadow:carry
86|G|Funding Mean Reversion|ANY|funding|P1|-
87|G|Extreme Positive Funding Short|ANY|funding,short_exec|P1|-
88|G|Extreme Negative Funding Long|ANY|funding|P1|shadow:carry
89|G|Funding + OI Squeeze|ANY|funding,oi|P1|-
90|G|Basis Convergence|ANY|funding,perp_exec|P1|-
91|G|Term-Structure Relative Value|ANY|funding,perp_exec|P2|-
92|G|Delta-Neutral Funding Portfolio|ANY|funding,perp_exec|P1|-
93|H|CEX–CEX Arbitrage|ANY|multi_venue,tickers|P1|shadow:venue_compare
94|H|DEX–CEX Arbitrage|ANY|dex|P2|-
95|H|DEX–DEX Arbitrage|ANY|dex|P2|-
96|H|Triangular Arbitrage|ANY|tickers|P1|shadow:triangular
97|H|Multi-Hop Graph Arbitrage (Bellman-Ford)|ANY|tickers|P1|shadow:triangular
98|H|Spot–Perpetual Price Arbitrage|ANY|funding,perp_exec|P1|-
99|H|Futures Calendar Arbitrage|ANY|perp_exec|P2|-
100|H|Stablecoin Cross-Market Arbitrage|ANY|tickers,multi_venue|P2|-
101|H|AMM-vs-Orderbook Arbitrage|ANY|dex|P2|-
102|H|Oracle/Market Temporary Divergence|ANY|dex|P2|-
103|I|Order Book Imbalance Momentum|ANY|book|P0|sleeve:obi_momentum
104|I|Multi-Level OBI Momentum|ANY|book|P1|-
105|I|OBI Reversal|RANGE|book|P1|-
106|I|Microprice Momentum|ANY|book|P0|sleeve:obi_momentum
107|I|Microprice Mean Reversion|RANGE|book|P1|-
108|I|Bid-Ask Pressure|ANY|book|P1|sleeve:obi_momentum
109|I|Trade-Flow Imbalance|ANY|trades|P0|-
110|I|CVD Momentum|ANY|trades|P0|-
111|I|CVD Divergence|ANY|trades|P1|-
112|I|Aggressive Buyer Burst|ANY|trades|P1|-
113|I|Aggressive Seller Burst|ANY|trades|P1|-
114|I|VWAP-to-Mid Pressure|ANY|book,trades|P1|-
115|I|Queue-Depletion Breakout|ANY|book|P2|-
116|I|Spread Compression Signal|ANY|book|P1|filter:spread_gate
117|I|Spread Expansion Risk-Off|ANY|book|P1|filter:spread_gate
118|J|Bid Wall Absorption|ANY|book,trades|P1|-
119|J|Ask Wall Absorption|ANY|book,trades|P1|-
120|J|Bid Wall Failure|ANY|book|P2|-
121|J|Ask Wall Failure|ANY|book|P2|-
122|J|Liquidity Vacuum Breakout|ANY|book|P1|-
123|J|Depth Collapse|ANY|book|P1|filter:depth_gate
124|J|Depth Recovery|ANY|book|P2|-
125|J|Hidden Liquidity Inference|ANY|book,trades|P2|-
126|J|Spoof-Filtered OBI|ANY|book|P2|-
127|J|Book Refill Momentum|ANY|book|P2|-
128|J|Sweep-and-Reclaim|ANY|ohlcv,book|P0|sleeve:sweep_reversal
129|J|Queue Imbalance Execution|ANY|book|P1|exec:entry_optimizer
130|J|Depth-Adjusted Entry|ANY|book|P1|exec:entry_optimizer
131|K|Price↑ + OI↑ Continuation|TREND|oi|P1|-
132|K|Price↑ + OI↓ Short-Covering Filter|TREND|oi|P1|-
133|K|Price↓ + OI↑ Short Build-Up|TREND_DOWN|oi|P1|-
134|K|Price↓ + OI↓ Long Liquidation Exhaustion|VOLATILE|oi|P1|-
135|K|OI Breakout|TREND|oi|P2|-
136|K|OI Divergence|ANY|oi|P2|-
137|K|Funding + OI Composite|ANY|funding,oi|P1|-
138|K|Liquidation Cascade Long|VOLATILE|liquidations|P1|-
139|K|Liquidation Cascade Short|VOLATILE|liquidations,short_exec|P2|-
140|K|Liquidation Exhaustion Reversal|VOLATILE|liquidations|P1|-
141|K|Long/Short Ratio Contrarian|ANY|oi|P2|-
142|K|Basis + OI Composite|ANY|funding,oi|P2|-
143|K|Mark/Index Divergence|ANY|funding|P2|-
144|K|Perp-Spot Lead/Lag|ANY|funding,tickers|P2|-
145|K|Funding Regime Transition|ANY|funding|P2|-
146|L|USDT Exchange Inflow Momentum|ANY|onchain|P1|-
147|L|USDC Exchange Inflow Momentum|ANY|onchain|P2|-
148|L|Stablecoin Dry-Powder Expansion|ANY|onchain|P1|-
149|L|BTC Exchange Inflow Bearish Pressure|ANY|onchain|P1|-
150|L|ETH Exchange Inflow Pressure|ANY|onchain|P1|-
151|L|Exchange Outflow Accumulation|ANY|onchain|P1|-
152|L|Whale Deposit Event|ANY|onchain|P2|-
153|L|Whale Withdrawal Event|ANY|onchain|P2|-
154|L|Miner-to-Exchange Flow|ANY|onchain|P2|-
155|L|Stablecoin Mint Impulse|ANY|onchain|P2|-
156|L|Stablecoin Burn/Redemption Risk|ANY|onchain|P2|-
157|L|Exchange Reserve Trend|ANY|onchain|P2|-
158|L|Active Address Acceleration|ANY|onchain|P2|-
159|L|Transaction Volume Surprise|ANY|onchain|P2|-
160|L|On-chain Flow + Price Confirmation|ANY|onchain,ohlcv|P1|-
161|M|Exchange Listing Momentum|ANY|news|P0|sleeve:catalyst
162|M|Exchange Delisting Short/Risk-Off|ANY|news|P0|filter:news_severe_risk
163|M|Protocol Upgrade Momentum|ANY|news|P1|sleeve:catalyst
164|M|Mainnet Launch|ANY|news|P1|sleeve:catalyst
165|M|Token Unlock Pressure|ANY|news|P1|role:haber_sosyal
166|M|Token Burn Catalyst|ANY|news|P1|sleeve:catalyst
167|M|ETF/Institutional News|ANY|news|P0|sleeve:catalyst
168|M|Regulatory Shock|ANY|news|P0|filter:market_risk_level
169|M|Hack/Exploit Risk-Off|ANY|news|P0|filter:news_severe_risk
170|M|Exchange Outage Dislocation|ANY|news|P1|filter:news_severe_risk
171|M|Partnership Catalyst|ANY|news|P1|sleeve:catalyst
172|M|Treasury Purchase Event|ANY|news|P1|sleeve:catalyst
173|M|Lawsuit/Enforcement Event|ANY|news|P1|role:haber_sosyal
174|M|News Surprise + Volume Confirmation|ANY|news,ohlcv|P0|sleeve:catalyst
175|M|News Overreaction Reversal|RANGE|news,ohlcv|P1|sleeve:news_overreaction
176|N|Social Sentiment Momentum|ANY|social|P2|role:haber_sosyal
177|N|Sentiment Reversal|ANY|social|P2|-
178|N|Google Search Attention Spike|ANY|social_volume|P2|-
179|N|Reddit Attention Acceleration|ANY|social|P2|role:haber_sosyal
180|N|News Volume Shock|ANY|news|P1|role:haber_sosyal
181|N|Social Volume Shock|ANY|social_volume|P2|-
182|N|Sentiment-Price Divergence|ANY|social,ohlcv|P2|-
183|N|Fear/Greed Contrarian|ANY|news|P2|-
184|N|Sentiment + Order Flow Confirmation|ANY|social,trades|P2|-
185|N|Multimodal News/Market Confirmation|ANY|news,ohlcv|P1|sleeve:catalyst
186|O|Realized Volatility Breakout|SQUEEZE|ohlcv|P1|sleeve:squeeze_breakout
187|O|ATR Regime Expansion|ANY|ohlcv|P1|role:rejim_oynaklik
188|O|Volatility Compression|SQUEEZE|ohlcv|P0|sleeve:squeeze_breakout
189|O|Volatility-of-Volatility Shock|VOLATILE|ohlcv|P2|-
190|O|HAR Volatility Forecast Filter|ANY|ohlcv|P1|-
191|O|GARCH Volatility Filter|ANY|ohlcv|P1|-
192|O|EGARCH Shock Filter|ANY|ohlcv|P2|-
193|O|Quantile Volatility Forecast (P50/P90)|ANY|ohlcv|P1|-
194|O|Volatility Targeting|ANY|ohlcv|P0|alloc:sizing_vol_target
195|O|Risk-Parity Volatility Scaling|ANY|ohlcv|P1|-
196|O|High-Vol Momentum|VOLATILE|ohlcv|P2|-
197|O|Low-Vol Breakout Preparation|SQUEEZE|ohlcv|P1|sleeve:squeeze_breakout
198|O|Volatility Shock Cash Mode|ANY|ohlcv|P0|alloc:portfolio_mode
199|P|Pure Market Making|ANY|book|P1|shadow:market_making
200|P|Inventory-Skew Market Making|ANY|book|P1|shadow:market_making
201|P|Avellaneda–Stoikov|ANY|book|P1|shadow:market_making
202|P|Adaptive Avellaneda–Stoikov|ANY|book|P1|shadow:market_making
203|P|OBI-Skewed Market Making|ANY|book|P2|-
204|P|Microprice-Skewed Market Making|ANY|book|P2|-
205|P|Volatility-Adaptive Market Making|ANY|book|P1|shadow:market_making
206|P|Cross-Exchange Market Making|ANY|multi_venue,book|P2|-
207|P|Perpetual Market Making|ANY|perp_exec|P2|-
208|P|Multi-Level Market Making|ANY|book|P2|-
209|P|Adverse-Selection-Aware MM|ANY|book,trades|P1|shadow:market_making
210|P|Alpha-Adjusted Market Making|ANY|book|P2|-
211|Q|Maker-First Smart Entry|ANY|book|P0|exec:entry_optimizer
212|Q|Maker → Taker Adaptive Chase|ANY|book|P0|exec:runner_chase
213|Q|TWAP|ANY|book|P2|-
214|Q|Adaptive TWAP|ANY|book|P2|-
215|Q|POV Execution|ANY|trades|P2|-
216|Q|Liquidity-Aware Order Slicing|ANY|book|P1|-
217|Q|OBI-Aware Execution|ANY|book|P1|exec:entry_optimizer
218|Q|Smart Venue Routing|ANY|multi_venue|P0|exec:venue_router
219|Q|Adverse-Selection-Aware Limit Placement|ANY|book|P1|exec:entry_optimizer
220|Q|Dynamic Cancel/Replace|ANY|book|P1|exec:runner_chase
221|R|Fixed Net TP|ANY|ohlcv|P0|exit:FIXED_TARGET
222|R|ATR TP|ANY|ohlcv|P1|-
223|R|MFE-Quantile TP|ANY|ohlcv|P1|-
224|R|Partial TP + Breakeven|ANY|ohlcv|P0|exit:PARTIAL_AND_RUN
225|R|Multi-Level Partial TP|ANY|ohlcv|P1|-
226|R|ATR Trailing|ANY|ohlcv|P0|exit:chandelier
227|R|Chandelier Exit|ANY|ohlcv|P0|exit:chandelier
228|R|Structure Trailing|ANY|ohlcv|P1|-
229|R|EMA Trailing|ANY|ohlcv|P2|-
230|R|Half-Peak Giveback|ANY|ohlcv|P0|exit:giveback_net
231|R|Dynamic Peak Giveback|ANY|ohlcv|P0|exit:DYNAMIC_PEAK
232|R|Continuation-Probability Exit|ANY|ohlcv|P0|exit:MODEL_EXIT
233|R|Edge-Decay Exit|ANY|ohlcv|P0|exit:EDGE_DECAY
234|R|Time Stop|ANY|ohlcv|P0|exit:TIME_STOP
235|R|Regime-Change Exit|ANY|ohlcv|P1|-
236|S|Equal-Risk Sleeve Allocation|ANY|ohlcv|P0|alloc:risk_per_trade
237|S|Volatility-Parity Allocation|ANY|ohlcv|P1|-
238|S|EV-Weighted Allocation|ANY|ohlcv|P0|alloc:ev_competition
239|S|Robust-EV Allocation|ANY|ohlcv|P0|alloc:ev_competition
240|S|Correlation-Penalized Allocation|ANY|ohlcv|P1|alloc:portfolio_mode
241|S|Strategy Risk Parity|ANY|ohlcv|P2|-
242|S|Regime-Switched Allocation|ANY|ohlcv|P0|alloc:regime_sleeves
243|S|Bayesian Strategy Reliability Allocation|ANY|ohlcv|P0|alloc:meta_allocator
244|S|Thompson Strategy Allocation|ANY|ohlcv|P1|alloc:meta_allocator
245|S|Champion–Challenger Allocation|ANY|ohlcv|P0|alloc:challenger
246|S|Drawdown-Controlled Allocation|ANY|ohlcv|P0|alloc:portfolio_mode
247|S|Market-Beta-Neutral Allocation|ANY|ohlcv,short_exec|P2|-
248|S|Dynamic Cash Allocation|ANY|ohlcv,news|P0|alloc:portfolio_mode
249|S|Breadth-Controlled Exposure|ANY|ohlcv|P0|alloc:portfolio_mode
250|S|Multi-Strategy Meta-Allocator|ANY|ohlcv|P0|alloc:meta_allocator
251|B|Fair Value Gap Fill (ICT)|TREND|ohlcv|P0|sleeve:fvg_fill
252|D|Inverse FVG Reclaim|ANY|ohlcv|P0|sleeve:ifvg_reclaim
253|D|HTF Range Break-and-Reclaim|ANY|ohlcv|P0|sleeve:range_reclaim
254|J|Manipulation Candle (prior-bar sweep + close-through)|ANY|ohlcv|P0|sleeve:manipulation_candle
255|C|Opening Range Breakout + Retest|TREND|ohlcv|P1|sleeve:opening_range
256|B|EMA Stack Pullback + Engulfing|TREND|ohlcv|P0|sleeve:ema_engulf
257|E|Volume Point-of-Control Reversion|RANGE|ohlcv|P1|sleeve:poc_reversion
258|B|Order Block Retest after MSS|TREND|ohlcv|P1|sleeve:order_block
259|E|EMA Cross-Back + Stochastic|RANGE|ohlcv|P2|sleeve:stoch_cross_back
260|E|Bollinger Lower Band to Mid (range only)|RANGE|ohlcv|P2|sleeve:bb_lower_band
"""

# 251-260: YouTube video transkriptlerinden çıkarılan kurulumlar (2026-09-04).
# İddialar ALINMADI, mekanik çekirdek alındı; kaynak künyesi strategies/sleeves_video.SOURCES.
VIDEO_SOURCED_IDS = set(range(251, 261))

IMPL_KIND_TR = {"sleeve": "komite sleeve'i", "shadow": "gölge araştırma modülü", "role": "komite rolü",
                "filter": "kapı/filtre", "exit": "çıkış motoru", "alloc": "tahsis/portföy", "exec": "yürütme"}


def _parse() -> List[Dict]:
    rows = []
    for line in _RAW.strip().splitlines():
        sid, fam, name, regime, data, prio, impl = [x.strip() for x in line.split("|")]
        needs = set(x for x in data.split(",") if x)
        impl_kind, impl_key = (impl.split(":", 1) if ":" in impl else (None, None))
        if impl_kind:
            status = "SHADOW" if impl_kind == "shadow" else "IMPLEMENTED"
        elif needs & NO_SOURCE_DATA:
            status = "NO_DATA"
        elif needs & WIRABLE_DATA:
            status = "DATA_NOT_WIRED"
        elif needs & set(OPTIONAL_DATA):
            status = "RESEARCH"
        else:
            status = "RESEARCH"
        stage = ("PAPER" if status == "IMPLEMENTED" else "SHADOW" if status == "SHADOW" else "REGISTERED")
        rows.append({"id": int(sid), "family": fam, "family_tr": FAMILIES[fam], "name": name, "regime": regime,
                     "data": sorted(needs), "priority": prio, "impl_kind": impl_kind, "impl_key": impl_key,
                     "impl_tr": (IMPL_KIND_TR.get(impl_kind) if impl_kind else None),
                     "video_sourced": int(sid) in VIDEO_SOURCED_IDS,
                     "status": status, "pipeline_stage": stage,
                     "blocker": (", ".join(sorted(needs & NO_SOURCE_DATA)) + " kaynağı yok" if status == "NO_DATA"
                                 else ", ".join(sorted(needs & WIRABLE_DATA)) + " ccxt'den çekilebilir, koşucuya bağlanmadı" if status == "DATA_NOT_WIRED"
                                 else "spot simülatörde short/perp yürütme yok — yalnız gölge" if status == "SHADOW" and (needs & {"short_exec", "perp_exec"})
                                 else None)})
    return rows


LIBRARY: List[Dict] = _parse()
assert len(LIBRARY) == 260 and len({r["id"] for r in LIBRARY}) == 260


def by_status() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in LIBRARY:
        out[r["status"]] = out.get(r["status"], 0) + 1
    return out


def funnel() -> List[Dict]:
    """Eleme hunisi — 'yüzlerce strateji çalıştırmak' değil, kanıtla azaltmak."""
    n = len(LIBRARY)
    st = by_status()
    active = st.get("IMPLEMENTED", 0) + st.get("SHADOW", 0)
    return [{"stage": "kayıtlı hipotez", "n": n},
            {"stage": "veri mevcut", "n": n - st.get("NO_DATA", 0) - st.get("DATA_NOT_WIRED", 0)},
            {"stage": "kodda çalışan (paper) + gölge", "n": active},
            {"stage": "komite sleeve'i (EV yarışmasında)", "n": sum(1 for r in LIBRARY if r["impl_kind"] == "sleeve")},
            {"stage": "OOS/DSR/PBO kapıları geçmiş (QUALIFIED)", "n": sum(1 for r in LIBRARY if r["pipeline_stage"] == "QUALIFIED")}]


def sleeves_implemented() -> List[str]:
    return sorted({r["impl_key"] for r in LIBRARY if r["impl_kind"] == "sleeve"})


def apply_lifecycle(lifecycle_status: Optional[List[Dict]]) -> List[Dict]:
    """Lifecycle kayıt defterindeki aşamayı kütüphane satırına yansıtır (sleeve'ler için)."""
    if not lifecycle_status:
        return LIBRARY
    stage = {x["sleeve"]: x.get("stage") for x in lifecycle_status}
    gates = {x["sleeve"]: x.get("gates_passed") for x in lifecycle_status}
    out = []
    for r in LIBRARY:
        r = dict(r)
        if r["impl_kind"] == "sleeve" and r["impl_key"] in stage:
            r["lifecycle_stage"] = stage[r["impl_key"]]
            r["gates_passed"] = gates.get(r["impl_key"])
            if gates.get(r["impl_key"]):
                r["pipeline_stage"] = "QUALIFIED"
        out.append(r)
    return out


def summary(lifecycle_status: Optional[List[Dict]] = None) -> Dict:
    rows = apply_lifecycle(lifecycle_status)
    fam = {}
    for r in rows:
        f = fam.setdefault(r["family"], {"family": r["family"], "family_tr": r["family_tr"], "n": 0, "implemented": 0, "shadow": 0, "no_data": 0})
        f["n"] += 1
        if r["status"] == "IMPLEMENTED":
            f["implemented"] += 1
        elif r["status"] == "SHADOW":
            f["shadow"] += 1
        elif r["status"] in ("NO_DATA", "DATA_NOT_WIRED"):
            f["no_data"] += 1
    return {"n": len(rows), "by_status": by_status(), "funnel": funnel(), "families": list(fam.values()),
            "pipeline": PIPELINE, "available_data": sorted(AVAILABLE_DATA), "optional_data": OPTIONAL_DATA,
            "rows": rows,
            "note": ("Durumlar dürüsttür: IMPLEMENTED = komitede paper çalışıyor (OOS/DSR kanıtı birikiyor, QUALIFIED değil); "
                     "SHADOW = sinyal var emir yok; NO_DATA = kaynak yok, aktifleştirilemez.")}
