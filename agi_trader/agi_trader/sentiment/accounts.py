"""
Kritik Twitter/X hesapları veritabanı (NLP/Sentiment rolü).

Spec: piyasayı hareket ettirebilen yüzlerce hesap, önem derecesine göre.
İki büyük blok:
  • KRİPTO/EXCHANGE (~400 hedef): borsa CEO/yöneticileri, proje kurucuları,
    VC/fonlar, market maker, whale tracker, on-chain analist, trader, haber,
    developer, influencer.
  • SİYASİ/MAKRO (~200 hedef): ABD yönetimi, Fed/merkez bankaları, regülatörler
    (SEC/CFTC), dünya liderleri, maliye bakanları, makro ekonomistler.

Her hesap: weight (0-10 etki gücü), category, assets (etkilediği varlıklar).
Liste küratörlüdür ve genişletilebilir; weight ve kategori yapısı motorun
ağırlıklandırma mantığını besler. (handle'lar @ olmadan yazılır.)
"""
from __future__ import annotations

from typing import Dict

# ===========================================================================
# BLOK 1 — KRİPTO / EXCHANGE
# ===========================================================================
CRYPTO_ACCOUNTS: Dict[str, Dict[str, Dict]] = {
    "exchange_leaders": {
        "cz_binance": {"weight": 9.6, "category": "exchange_ceo", "assets": ["BNB", "all"]},
        "binance": {"weight": 9.4, "category": "exchange", "assets": ["all"]},
        "brian_armstrong": {"weight": 9.2, "category": "exchange_ceo", "assets": ["all"]},
        "coinbase": {"weight": 9.0, "category": "exchange", "assets": ["all"]},
        "tyler": {"weight": 8.2, "category": "exchange_ceo", "assets": ["all"]},
        "cameron": {"weight": 8.2, "category": "exchange_ceo", "assets": ["all"]},
        "paoloardoino": {"weight": 8.6, "category": "stablecoin_ceo", "assets": ["USDT", "all"]},
        "tether_to": {"weight": 8.3, "category": "stablecoin", "assets": ["USDT", "all"]},
        "circle": {"weight": 8.0, "category": "stablecoin", "assets": ["USDC", "all"]},
        "jeremyallaire": {"weight": 8.0, "category": "stablecoin_ceo", "assets": ["USDC", "all"]},
        "kucoincom": {"weight": 7.5, "category": "exchange", "assets": ["all"]},
        "bybit_official": {"weight": 8.2, "category": "exchange", "assets": ["all"]},
        "benbybit": {"weight": 7.8, "category": "exchange_ceo", "assets": ["all"]},
        "okx": {"weight": 8.2, "category": "exchange", "assets": ["all"]},
        "star_okx": {"weight": 7.6, "category": "exchange_ceo", "assets": ["all"]},
        "krakenfx": {"weight": 8.0, "category": "exchange", "assets": ["all"]},
        "gate_io": {"weight": 7.2, "category": "exchange", "assets": ["all"]},
        "bitget_global": {"weight": 7.2, "category": "exchange", "assets": ["all"]},
        "htx_global": {"weight": 7.0, "category": "exchange", "assets": ["all"]},
        "bitfinex": {"weight": 7.6, "category": "exchange", "assets": ["all"]},
        "gemini": {"weight": 7.5, "category": "exchange", "assets": ["all"]},
        "cryptocom": {"weight": 7.6, "category": "exchange", "assets": ["CRO", "all"]},
        "kris": {"weight": 7.2, "category": "exchange_ceo", "assets": ["CRO", "all"]},
    },
    "project_founders": {
        "VitalikButerin": {"weight": 9.8, "category": "founder", "assets": ["ETH", "all"]},
        "aeyakovenko": {"weight": 9.0, "category": "founder", "assets": ["SOL"]},
        "rajgokal": {"weight": 8.2, "category": "founder", "assets": ["SOL"]},
        "haydenzadams": {"weight": 8.2, "category": "founder", "assets": ["UNI"]},
        "gavofyork": {"weight": 8.2, "category": "founder", "assets": ["DOT", "KSM"]},
        "rune_christensen": {"weight": 7.8, "category": "founder", "assets": ["MKR", "DAI"]},
        "justinsuntron": {"weight": 7.8, "category": "founder", "assets": ["TRX"]},
        "stani_kulechov": {"weight": 7.8, "category": "founder", "assets": ["AAVE"]},
        "sandeepnailwal": {"weight": 7.8, "category": "founder", "assets": ["MATIC", "POL"]},
        "0xPolygon": {"weight": 7.6, "category": "project", "assets": ["MATIC", "POL"]},
        "tarunchitra": {"weight": 7.0, "category": "researcher", "assets": ["all"]},
        "zhusu": {"weight": 7.2, "category": "founder", "assets": ["all"]},
        "Rewkang": {"weight": 7.4, "category": "trader_fund", "assets": ["all"]},
        "danheld": {"weight": 7.0, "category": "advocate", "assets": ["BTC"]},
        "ethereum": {"weight": 8.6, "category": "project", "assets": ["ETH"]},
        "solana": {"weight": 8.6, "category": "project", "assets": ["SOL"]},
        "avax": {"weight": 7.6, "category": "project", "assets": ["AVAX"]},
        "el33th4xor": {"weight": 7.6, "category": "founder", "assets": ["AVAX"]},
        "cosmos": {"weight": 7.2, "category": "project", "assets": ["ATOM"]},
        "chainlink": {"weight": 7.8, "category": "project", "assets": ["LINK"]},
        "sergeynazarov": {"weight": 7.8, "category": "founder", "assets": ["LINK"]},
        "dydx": {"weight": 7.0, "category": "project", "assets": ["DYDX"]},
        "arbitrum": {"weight": 7.6, "category": "project", "assets": ["ARB"]},
        "optimismFND": {"weight": 7.4, "category": "project", "assets": ["OP"]},
        "celestia": {"weight": 7.4, "category": "project", "assets": ["TIA"]},
        "suinetwork": {"weight": 7.2, "category": "project", "assets": ["SUI"]},
        "aptos": {"weight": 7.0, "category": "project", "assets": ["APT"]},
        "TONblockchain": {"weight": 7.4, "category": "project", "assets": ["TON"]},
    },
    "institutional_vc": {
        "a16zcrypto": {"weight": 9.2, "category": "vc_fund", "assets": ["all"]},
        "cdixon": {"weight": 8.4, "category": "vc", "assets": ["all"]},
        "paradigm": {"weight": 9.0, "category": "vc_fund", "assets": ["all"]},
        "PanteraCapital": {"weight": 8.8, "category": "fund", "assets": ["all"]},
        "dan_pantera": {"weight": 8.0, "category": "fund_ceo", "assets": ["all"]},
        "novogratz": {"weight": 8.6, "category": "fund_manager", "assets": ["BTC", "all"]},
        "galaxyhq": {"weight": 8.4, "category": "fund", "assets": ["all"]},
        "GrayscaleInvest": {"weight": 8.6, "category": "institutional", "assets": ["BTC", "ETH"]},
        "BlackRock": {"weight": 9.6, "category": "institutional", "assets": ["BTC", "ETH"]},
        "larryfink": {"weight": 9.4, "category": "institutional_ceo", "assets": ["BTC", "all"]},
        "Fidelity": {"weight": 8.6, "category": "institutional", "assets": ["BTC", "ETH"]},
        "VanEck_US": {"weight": 8.0, "category": "institutional", "assets": ["BTC", "ETH"]},
        "ARKInvest": {"weight": 8.2, "category": "institutional", "assets": ["BTC", "all"]},
        "CathieDWood": {"weight": 8.2, "category": "fund_manager", "assets": ["BTC", "all"]},
        "framework": {"weight": 7.6, "category": "vc_fund", "assets": ["all"]},
        "DelphiDigital": {"weight": 8.0, "category": "research_fund", "assets": ["all"]},
        "multicoincap": {"weight": 8.0, "category": "fund", "assets": ["all"]},
        "KyleSamani": {"weight": 7.8, "category": "fund", "assets": ["all"]},
        "VentureCoinist": {"weight": 6.8, "category": "vc", "assets": ["all"]},
        "DigitalAssetCap": {"weight": 7.2, "category": "fund", "assets": ["all"]},
        "MorganCreekCap": {"weight": 7.4, "category": "fund", "assets": ["BTC", "all"]},
        "saylor": {"weight": 9.2, "category": "corporate_btc", "assets": ["BTC"]},
        "MicroStrategy": {"weight": 9.0, "category": "corporate_btc", "assets": ["BTC"]},
        "Strategy": {"weight": 8.6, "category": "corporate_btc", "assets": ["BTC"]},
    },
    "onchain_analysts": {
        "glassnode": {"weight": 9.0, "category": "data_provider", "assets": ["all"]},
        "_checkmatey_": {"weight": 8.0, "category": "onchain_analyst", "assets": ["BTC"]},
        "lookonchain": {"weight": 9.3, "category": "onchain_tracker", "assets": ["all"]},
        "ki_young_ju": {"weight": 8.6, "category": "onchain_analyst", "assets": ["BTC"]},
        "cryptoquant_com": {"weight": 8.4, "category": "data_provider", "assets": ["all"]},
        "Willy_Woo": {"weight": 8.6, "category": "onchain_analyst", "assets": ["BTC"]},
        "WhaleAlert": {"weight": 9.0, "category": "whale_tracker", "assets": ["all"]},
        "whale_alert": {"weight": 8.8, "category": "whale_tracker", "assets": ["all"]},
        "ArkhamIntel": {"weight": 8.8, "category": "intelligence", "assets": ["all"]},
        "nansen_ai": {"weight": 8.6, "category": "intelligence", "assets": ["all"]},
        "spotonchain": {"weight": 8.3, "category": "onchain_tracker", "assets": ["all"]},
        "EmberCN": {"weight": 8.2, "category": "whale_tracker", "assets": ["all"]},
        "santimentfeed": {"weight": 8.0, "category": "data_provider", "assets": ["all"]},
        "intotheblock": {"weight": 7.8, "category": "data_provider", "assets": ["all"]},
        "MessariCrypto": {"weight": 8.2, "category": "research", "assets": ["all"]},
        "DefiLlama": {"weight": 8.4, "category": "data_provider", "assets": ["all"]},
        "tokenterminal": {"weight": 7.6, "category": "data_provider", "assets": ["all"]},
        "coinmetrics": {"weight": 7.8, "category": "data_provider", "assets": ["all"]},
        "thetokenunlocks": {"weight": 7.6, "category": "unlock_tracker", "assets": ["all"]},
        "DefiantNews": {"weight": 7.2, "category": "news", "assets": ["all"]},
    },
    "traders_analysts": {
        "CryptoCred": {"weight": 7.6, "category": "trader", "assets": ["all"]},
        "pentosh1": {"weight": 7.8, "category": "trader", "assets": ["all"]},
        "RektCapital": {"weight": 7.8, "category": "analyst", "assets": ["BTC", "ETH"]},
        "CryptoCapo_": {"weight": 7.2, "category": "trader", "assets": ["all"]},
        "IncomeSharks": {"weight": 7.2, "category": "trader", "assets": ["all"]},
        "TheCryptoDog": {"weight": 7.2, "category": "trader", "assets": ["all"]},
        "CryptoMichNL": {"weight": 7.2, "category": "trader", "assets": ["all"]},
        "rovercrc": {"weight": 6.8, "category": "trader", "assets": ["all"]},
        "AltcoinSherpa": {"weight": 7.0, "category": "trader", "assets": ["all"]},
        "CryptoKaleo": {"weight": 7.2, "category": "trader", "assets": ["all"]},
        "inversebrah": {"weight": 6.8, "category": "trader", "assets": ["all"]},
        "CryptoGodJohn": {"weight": 6.6, "category": "trader", "assets": ["all"]},
        "smartcontracter": {"weight": 7.4, "category": "analyst", "assets": ["all"]},
        "ColdBloodShill": {"weight": 6.6, "category": "trader", "assets": ["all"]},
        "DaanCrypto": {"weight": 7.0, "category": "trader", "assets": ["all"]},
        "ali_charts": {"weight": 7.0, "category": "analyst", "assets": ["all"]},
        "CrypNuevo": {"weight": 7.2, "category": "analyst", "assets": ["all"]},
        "52kskew": {"weight": 7.2, "category": "derivatives", "assets": ["all"]},
        "CredibleCrypto": {"weight": 7.0, "category": "trader", "assets": ["BTC", "XRP"]},
        "TraderSZ": {"weight": 6.8, "category": "trader", "assets": ["all"]},
        "ThinkingUSD": {"weight": 7.0, "category": "trader", "assets": ["all"]},
        "GiganticRebirth": {"weight": 7.2, "category": "trader", "assets": ["all"]},
        "HsakaTrades": {"weight": 7.6, "category": "trader", "assets": ["all"]},
        "loomdart": {"weight": 7.0, "category": "trader", "assets": ["all"]},
        "AltcoinGordon": {"weight": 6.6, "category": "influencer", "assets": ["all"]},
    },
    "macro_crypto": {
        "RaoulGMI": {"weight": 8.6, "category": "macro", "assets": ["BTC", "all"]},
        "LynAldenContact": {"weight": 8.8, "category": "macro", "assets": ["BTC", "all"]},
        "arthurhayes": {"weight": 8.6, "category": "macro_trader", "assets": ["BTC", "ETH"]},
        "PeterSchiff": {"weight": 6.8, "category": "macro_bear", "assets": ["BTC"]},
        "100trillionUSD": {"weight": 7.6, "category": "analyst", "assets": ["BTC"]},
        "woonomic": {"weight": 8.2, "category": "onchain_analyst", "assets": ["BTC"]},
        "TheLastBearSta1": {"weight": 7.2, "category": "macro", "assets": ["all"]},
        "fejau_inc": {"weight": 7.8, "category": "macro", "assets": ["all"]},
        "CryptoHayes": {"weight": 7.4, "category": "macro_trader", "assets": ["all"]},
        "QwQiao": {"weight": 7.4, "category": "macro", "assets": ["all"]},
    },
    "news_media": {
        "CoinDesk": {"weight": 8.2, "category": "news", "assets": ["all"]},
        "Cointelegraph": {"weight": 7.6, "category": "news", "assets": ["all"]},
        "TheBlock__": {"weight": 8.5, "category": "news", "assets": ["all"]},
        "WuBlockchain": {"weight": 8.4, "category": "news", "assets": ["all"]},
        "DBCryptoWorld": {"weight": 7.0, "category": "news", "assets": ["all"]},
        "BitcoinMagazine": {"weight": 7.6, "category": "news", "assets": ["BTC"]},
        "decryptmedia": {"weight": 7.4, "category": "news", "assets": ["all"]},
        "DL_News_": {"weight": 7.6, "category": "news", "assets": ["all"]},
        "tier10k": {"weight": 8.0, "category": "news_aggregator", "assets": ["all"]},
        "FirstSquawk": {"weight": 8.2, "category": "news_wire", "assets": ["all"]},
        "unfolded__": {"weight": 7.4, "category": "news", "assets": ["all"]},
        "CoinGecko": {"weight": 7.4, "category": "data_provider", "assets": ["all"]},
        "CoinMarketCap": {"weight": 7.6, "category": "data_provider", "assets": ["all"]},
        "watcherguru": {"weight": 8.0, "category": "news_aggregator", "assets": ["all"]},
        "CryptosR_Us": {"weight": 6.6, "category": "news", "assets": ["all"]},
        "BWEnews": {"weight": 7.8, "category": "news_wire", "assets": ["all"]},
    },
    # +20 KRİTİK EK HESAP (2026 — on-chain akış, likidasyon, üst-düzey trader/analiz)
    "critical_signals_2026": {
        "lookonchain": {"weight": 9.0, "category": "onchain_flow", "assets": ["all"]},
        "spotonchain": {"weight": 8.6, "category": "onchain_flow", "assets": ["all"]},
        "EmberCN": {"weight": 8.4, "category": "onchain_flow", "assets": ["all"]},
        "ArkhamIntel": {"weight": 8.5, "category": "onchain_intel", "assets": ["all"]},
        "nansen_ai": {"weight": 8.4, "category": "onchain_intel", "assets": ["all"]},
        "santimentfeed": {"weight": 8.0, "category": "onchain_data", "assets": ["all"]},
        "glassnode": {"weight": 8.6, "category": "onchain_data", "assets": ["BTC", "ETH"]},
        "intotheblock": {"weight": 7.8, "category": "onchain_data", "assets": ["all"]},
        "Pentosh1": {"weight": 8.6, "category": "top_trader", "assets": ["all"]},
        "CryptoCred": {"weight": 8.4, "category": "top_trader", "assets": ["all"]},
        "HsakaTrades": {"weight": 8.5, "category": "top_trader", "assets": ["all"]},
        "GiganticRebirth": {"weight": 8.3, "category": "top_trader", "assets": ["all"]},
        "CredibleCrypto": {"weight": 8.0, "category": "top_trader", "assets": ["BTC", "all"]},
        "rektcapital": {"weight": 8.0, "category": "ta_analyst", "assets": ["BTC"]},
        "CryptoKaleo": {"weight": 7.8, "category": "top_trader", "assets": ["all"]},
        "IncomeSharks": {"weight": 7.6, "category": "ta_analyst", "assets": ["all"]},
        "AltcoinSherpa": {"weight": 7.6, "category": "ta_analyst", "assets": ["all"]},
        "WClementeIII": {"weight": 8.2, "category": "macro_onchain", "assets": ["BTC", "all"]},
        "RaoulGMI": {"weight": 8.4, "category": "macro", "assets": ["all"]},
        "TedTalksMacro": {"weight": 7.8, "category": "macro", "assets": ["all"]},
    },
}

# ===========================================================================
# BLOK 2 — SİYASİ / MAKRO (borsaları etkileyen)
# ===========================================================================
POLITICAL_MACRO_ACCOUNTS: Dict[str, Dict[str, Dict]] = {
    "us_government": {
        "POTUS": {"weight": 9.8, "category": "head_of_state", "assets": ["all"]},
        "realDonaldTrump": {"weight": 9.8, "category": "head_of_state", "assets": ["all", "BTC"]},
        "WhiteHouse": {"weight": 9.4, "category": "government", "assets": ["all"]},
        "VP": {"weight": 8.6, "category": "government", "assets": ["all"]},
        "SecYellen": {"weight": 9.2, "category": "treasury", "assets": ["all"]},
        "USTreasury": {"weight": 9.0, "category": "treasury", "assets": ["all"]},
        "SpeakerJohnson": {"weight": 7.8, "category": "congress", "assets": ["all"]},
        "SenLummis": {"weight": 8.2, "category": "congress_crypto", "assets": ["BTC", "all"]},
        "RepTomEmmer": {"weight": 7.6, "category": "congress_crypto", "assets": ["all"]},
        "SenWarren": {"weight": 8.0, "category": "congress_crypto_bear", "assets": ["all"]},
        "GOPMajorityWhip": {"weight": 7.0, "category": "congress", "assets": ["all"]},
    },
    "regulators": {
        "SECGov": {"weight": 9.8, "category": "regulator", "assets": ["all"]},
        "GaryGensler": {"weight": 9.4, "category": "regulator", "assets": ["all"]},
        "CFTC": {"weight": 9.4, "category": "regulator", "assets": ["all"]},
        "USCBO": {"weight": 7.0, "category": "fiscal", "assets": ["all"]},
        "TheJusticeDept": {"weight": 8.6, "category": "doj", "assets": ["all"]},
        "FBI": {"weight": 7.6, "category": "enforcement", "assets": ["all"]},
        "USOCC": {"weight": 7.4, "category": "banking_regulator", "assets": ["all"]},
        "FDICgov": {"weight": 7.6, "category": "banking_regulator", "assets": ["all"]},
        "IRSnews": {"weight": 7.4, "category": "tax", "assets": ["all"]},
        "ECB": {"weight": 9.0, "category": "central_bank", "assets": ["all"]},
        "bankofengland": {"weight": 8.2, "category": "central_bank", "assets": ["all"]},
        "bankofjapan": {"weight": 8.2, "category": "central_bank", "assets": ["all"]},
    },
    "central_banks_fed": {
        "federalreserve": {"weight": 9.8, "category": "central_bank", "assets": ["all"]},
        "NewYorkFed": {"weight": 8.6, "category": "central_bank", "assets": ["all"]},
        "stlouisfed": {"weight": 7.6, "category": "central_bank", "assets": ["all"]},
        "GlobalMktObserv": {"weight": 7.4, "category": "macro_observer", "assets": ["all"]},
        "NickTimiraos": {"weight": 9.0, "category": "fed_whisperer", "assets": ["all"]},
        "federalreserveBoard": {"weight": 8.2, "category": "central_bank", "assets": ["all"]},
        "Lagarde": {"weight": 8.8, "category": "central_bank_head", "assets": ["all"]},
    },
    "world_leaders_finance": {
        "elonmusk": {"weight": 9.6, "category": "ceo_influencer", "assets": ["DOGE", "BTC", "all"]},
        "Tesla": {"weight": 8.0, "category": "corporate", "assets": ["BTC", "DOGE"]},
        "nayibbukele": {"weight": 8.6, "category": "head_of_state_btc", "assets": ["BTC"]},
        "RBReich": {"weight": 6.8, "category": "economist_policy", "assets": ["all"]},
        "IMFNews": {"weight": 8.4, "category": "international_org", "assets": ["all"]},
        "WorldBank": {"weight": 7.8, "category": "international_org", "assets": ["all"]},
        "wef": {"weight": 7.6, "category": "international_org", "assets": ["all"]},
        "POTUSPress": {"weight": 7.4, "category": "government", "assets": ["all"]},
        "EU_Commission": {"weight": 8.0, "category": "government_eu", "assets": ["all"]},
        "10DowningStreet": {"weight": 7.6, "category": "government_uk", "assets": ["all"]},
    },
    "macro_economists": {
        "Nouriel": {"weight": 7.6, "category": "economist_bear", "assets": ["all"]},
        "elerianm": {"weight": 8.2, "category": "economist", "assets": ["all"]},
        "biancoresearch": {"weight": 8.2, "category": "macro_research", "assets": ["all"]},
        "zerohedge": {"weight": 7.8, "category": "macro_media", "assets": ["all"]},
        "DeItaone": {"weight": 8.6, "category": "news_wire", "assets": ["all"]},
        "LizAnnSonders": {"weight": 7.8, "category": "economist", "assets": ["all"]},
        "GregJDaco": {"weight": 7.4, "category": "economist", "assets": ["all"]},
        "jposhaughnessy": {"weight": 7.0, "category": "investor", "assets": ["all"]},
        "WarrenBuffett": {"weight": 8.4, "category": "investor", "assets": ["all"]},
        "RayDalio": {"weight": 8.4, "category": "macro_investor", "assets": ["BTC", "all"]},
        "stlouisfed_econ": {"weight": 6.8, "category": "economist", "assets": ["all"]},
        "KobeissiLetter": {"weight": 8.4, "category": "macro_media", "assets": ["all"]},
        "Schuldensuehner": {"weight": 7.4, "category": "macro_media", "assets": ["all"]},
        "MacroAlf": {"weight": 7.8, "category": "macro_research", "assets": ["all"]},
        "michaelnaudt": {"weight": 6.8, "category": "macro", "assets": ["all"]},
    },
}


def _merge(blocks: Dict[str, Dict[str, Dict]], block_name: str, out: Dict[str, Dict]):
    for group, accounts in blocks.items():
        for handle, data in accounts.items():
            out[handle] = {**data, "group": group, "block": block_name}


def all_accounts() -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    _merge(CRYPTO_ACCOUNTS, "crypto", out)
    _merge(POLITICAL_MACRO_ACCOUNTS, "political_macro", out)
    return out


# Geriye dönük uyumluluk (eski kod CRITICAL_ACCOUNTS bekliyordu)
CRITICAL_ACCOUNTS = {**CRYPTO_ACCOUNTS, **POLITICAL_MACRO_ACCOUNTS}


def account_stats() -> Dict[str, int]:
    acc = all_accounts()
    crypto = sum(1 for a in acc.values() if a["block"] == "crypto")
    pol = sum(1 for a in acc.values() if a["block"] == "political_macro")
    return {"total": len(acc), "crypto_exchange": crypto, "political_macro": pol}
