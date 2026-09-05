"""
Sağlayıcı kataloğu — TEK DOĞRULUK KAYNAĞI (single source of truth).

Tüm borsa / piyasa-verisi / on-chain / bildirim / LLM sağlayıcıları BURADA
tanımlanır. Bir sağlayıcı (özellikle ücretsiz-API borsa aracı sitesi) eklemek
artık tek yerde, tek blok düzenleme ile yapılır; sistem otomatik olarak:

  • Kimlik bilgileri panelini (UI)         -> cred_schema()
  • .env / ortam anahtar listesini          -> env_keys()
  • Canlı borsa veri katmanını (ccxt)       -> exchange_priority() / default_exchanges()
  • UI'daki "ücretsiz anahtar al →" linkini -> signup_url alanı

… hepsini bu katalogdan türetir. Başka hiçbir dosyayı elle düzenlemek gerekmez.

YENİ SAĞLAYICI EKLEME (3 adım, hata yapması zor):
  1. Aşağıdaki PROVIDERS listesine bir Provider(...) bloğu ekle.
  2. Borsa ise ccxt_id ver (ccxt'de desteklenen id) — canlı veriye OTOMATİK katılır.
     Pasaparola gerektiren borsalarda fields'a Field("<ID>_PASSWORD", ...) ekle.
  3. signup_url'a ücretsiz anahtarın alındığı sayfayı yaz — panelde tıklanır link olur.

Doğrulama için:  python -m agi_trader.providers   (katalogu yazdırır + tutarlılık kontrolü)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Field:
    """Panelde bir giriş kutusuna karşılık gelen tek ortam anahtarı."""
    key: str                 # ortam değişkeni adı, ör. "BINANCE_API_KEY"
    label: str               # kutucuk etiketi
    secret: bool = True      # True ise maskeli (password) gösterilir


@dataclass(frozen=True)
class Provider:
    id: str                          # benzersiz slug, ör. "binance"
    name: str                        # görünen ad
    group: str                       # UI başlığı (kartların gruplanması)
    category: str                    # exchange|market|onchain|notify|llm|social
    fields: List[Field]
    free: bool = True                # ücretsiz katmanı/anahtarı var mı?
    signup_url: str = ""             # ücretsiz anahtarın alındığı sayfa (UI'da link)
    docs_url: str = ""               # API dokümanı
    note: str = ""                   # kısa açıklama (örn. "pasaparola gerekir")
    # --- yalnız borsalar (category == "exchange") ---
    ccxt_id: Optional[str] = None    # ccxt borsa id; verilirse canlı veriye katılır
    priority: float = 0.7            # çoklu-borsa veri birleştirme ağırlığı (0..1)
    keyless_public: bool = True      # public OHLCV anahtarsız çalışır mı?


# ════════════════════════════════════════════════════════════════════════════
#  KATALOG — ücretsiz-API odaklı. Yeni sağlayıcıyı buraya ekle, gerisi otomatik.
# ════════════════════════════════════════════════════════════════════════════
_EX = "🏦 Borsa (public veri anahtarsız · ücretsiz API ile hesap/özel veri)"
_MK = "📊 Piyasa Verisi (ücretsiz API)"
_OC = "🐋 On-chain (ücretsiz API)"
_OCP = "🐋 On-chain (premium)"
_NT = "🔔 Bildirim (işlem adayında uyarı)"
_LM = "🤖 LLM (derin tweet/haber analizi)"
_SC = "🐦 Sosyal Sinyal"
_MC = "📉 Makro & Haber (ücretsiz API)"
_DR = "📈 Türev & Likidasyon (çoğu ücretli)"
_ST = "🏛️ Hisse / ETF Verisi"


def _ex(id, name, ccxt_id, priority, signup, *, passphrase=False, note="", free=True):
    fs = [Field(f"{ccxt_id.upper()}_API_KEY", f"{name} API Key"),
          Field(f"{ccxt_id.upper()}_SECRET", f"{name} Secret")]
    if passphrase:
        fs.append(Field(f"{ccxt_id.upper()}_PASSWORD", f"{name} Passphrase"))
    return Provider(id=id, name=name, group=_EX, category="exchange", fields=fs,
                    free=free, signup_url=signup, ccxt_id=ccxt_id, priority=priority,
                    note=note or ("pasaparola gerekir" if passphrase else ""))


PROVIDERS: List[Provider] = [
    # ───────────────────────── Borsalar (ccxt — canlı veriye otomatik katılır) ──
    _ex("binance", "Binance", "binance", 1.00, "https://www.binance.com/en/my/settings/api-management"),
    _ex("bybit",   "Bybit",   "bybit",   0.95, "https://www.bybit.com/app/user/api-management"),
    _ex("okx",     "OKX",     "okx",     0.95, "https://www.okx.com/account/my-api", passphrase=True),
    _ex("kraken",  "Kraken",  "kraken",  0.85, "https://www.kraken.com/u/security/api"),
    _ex("coinbase","Coinbase","coinbase",0.85, "https://www.coinbase.com/settings/api"),
    _ex("kucoin",  "KuCoin",  "kucoin",  0.80, "https://www.kucoin.com/account/api", passphrase=True),
    _ex("gateio",  "Gate.io", "gateio",  0.75, "https://www.gate.io/myaccount/api_key_manage"),
    _ex("bitget",  "Bitget",  "bitget",  0.75, "https://www.bitget.com/account/newapi", passphrase=True),
    _ex("mexc",    "MEXC",    "mexc",    0.70, "https://www.mexc.com/user/openapi"),
    _ex("htx",     "HTX (Huobi)", "htx", 0.70, "https://www.htx.com/en-us/apikey/"),

    # ───────────────────────── Piyasa verisi (ücretsiz API anahtarı) ────────────
    Provider("coingecko", "CoinGecko", _MK, "market",
             [Field("COINGECKO_API_KEY", "CoinGecko API Key (ücretsiz Demo)")],
             signup_url="https://www.coingecko.com/en/developers/dashboard",
             note="ücretsiz Demo planı: ~30 çağrı/dk"),
    Provider("coinmarketcap", "CoinMarketCap", _MK, "market",
             [Field("COINMARKETCAP_API_KEY", "CoinMarketCap API Key (ücretsiz Basic)")],
             signup_url="https://pro.coinmarketcap.com/account",
             note="ücretsiz Basic planı: 10k çağrı/ay"),
    Provider("cryptocompare", "CryptoCompare", _MK, "market",
             [Field("CRYPTOCOMPARE_API_KEY", "CryptoCompare API Key (ücretsiz)")],
             signup_url="https://www.cryptocompare.com/cryptopian/api-keys",
             note="ücretsiz: ~100k çağrı/ay"),

    # ───────────────────────── On-chain (ücretsiz API anahtarı) ─────────────────
    Provider("etherscan", "Etherscan", _OC, "onchain",
             [Field("ETHERSCAN_API_KEY", "Etherscan API Key (ücretsiz)")],
             signup_url="https://etherscan.io/myapikey",
             note="ETH/ERC-20 büyük transfer + borsa cüzdan etiketi"),
    Provider("bscscan", "BscScan", _OC, "onchain",
             [Field("BSCSCAN_API_KEY", "BscScan API Key (ücretsiz)")],
             signup_url="https://bscscan.com/myapikey",
             note="BNB Chain zincir-üstü veriler"),
    Provider("blockchaincom", "Blockchain.com", _OC, "onchain",
             [Field("BLOCKCHAINCOM_API_KEY", "Blockchain.com API Key (ücretsiz)")],
             signup_url="https://www.blockchain.com/explorer/api",
             note="BTC zincir-üstü büyük transferler"),

    # ───────────────────────── On-chain (premium) ──────────────────────────────
    Provider("whale_alert", "Whale Alert", _OCP, "onchain",
             [Field("WHALE_ALERT_API_KEY", "Whale Alert API Key")],
             free=False, signup_url="https://whale-alert.io/",
             note="$29.95/ay — entity-etiketli balina transferleri"),

    # ───────────────────────── Sosyal sinyal ───────────────────────────────────
    Provider("twitter", "Twitter / X", _SC, "social",
             [Field("TWITTER_BEARER_TOKEN", "Twitter/X Bearer Token")],
             free=False, signup_url="https://developer.twitter.com/en/portal/dashboard",
             note="canlı sosyal ısı — ücretli API"),
    Provider("cryptopanic", "CryptoPanic", _SC, "social",
             [Field("CRYPTOPANIC_API_KEY", "CryptoPanic API Key (ücretsiz)")],
             signup_url="https://cryptopanic.com/developers/api/",
             note="ücretsiz kripto haber/sentiment akışı"),
    Provider("lunarcrush", "LunarCrush", _SC, "social",
             [Field("LUNARCRUSH_API_KEY", "LunarCrush API Key (ücretsiz)")],
             signup_url="https://lunarcrush.com/developers/api",
             note="sosyal medya metrikleri — ücretsiz 1000/gün"),

    # ───────────────────────── Makro & Haber (ücretsiz API) ─────────────────────
    Provider("fred", "FRED (St. Louis Fed)", _MC, "macro",
             [Field("FRED_API_KEY", "FRED API Key (ücretsiz)")],
             signup_url="https://fredaccount.stlouisfed.org/apikeys",
             note="FED faizi, CPI, işsizlik, GSYİH — makro katmanını CANLI yapar"),
    Provider("newsapi", "NewsAPI", _MC, "macro",
             [Field("NEWSAPI_API_KEY", "NewsAPI Key (ücretsiz)")],
             signup_url="https://newsapi.org/register",
             note="80K+ kaynaktan haber başlığı — ücretsiz 100/gün"),

    # ───────────────────────── Bildirim ────────────────────────────────────────
    Provider("telegram", "Telegram", _NT, "notify",
             [Field("TELEGRAM_BOT_TOKEN", "Telegram Bot Token"),
              Field("TELEGRAM_CHAT_ID", "Telegram Chat ID", secret=False)],
             signup_url="https://t.me/BotFather",
             note="@BotFather ile ücretsiz bot oluştur"),
    Provider("discord", "Discord", _NT, "notify",
             [Field("DISCORD_WEBHOOK_URL", "Discord Webhook URL")],
             signup_url="https://support.discord.com/hc/en-us/articles/228383668",
             note="kanal ayarlarından ücretsiz webhook"),
    Provider("tradingview", "TradingView Webhook", _NT, "notify",
             [Field("TRADINGVIEW_WEBHOOK_SECRET", "TradingView Webhook Secret")],
             signup_url="https://www.tradingview.com/support/solutions/43000529348-webhooks/",
             note="strateji alarmından POST /api/webhook/tradingview — gizli anahtar doğrulaması"),

    # ───────────────────────── LLM ─────────────────────────────────────────────
    Provider("anthropic", "Anthropic (Claude)", _LM, "llm",
             [Field("ANTHROPIC_API_KEY", "Anthropic API Key")],
             free=False, signup_url="https://console.anthropic.com/settings/keys"),
    Provider("deepseek", "DeepSeek", _LM, "llm",
             [Field("DEEPSEEK_API_KEY", "DeepSeek API Key")],
             free=False, signup_url="https://platform.deepseek.com/api_keys",
             note="düşük maliyetli LLM"),
    Provider("openai", "OpenAI", _LM, "llm",
             [Field("OPENAI_API_KEY", "OpenAI API Key")],
             free=False, signup_url="https://platform.openai.com/api-keys"),
    Provider("gemini_ai", "Google Gemini", _LM, "llm",
             [Field("GEMINI_API_KEY", "Gemini API Key")],
             free=True, signup_url="https://aistudio.google.com/app/apikey",
             note="ücretsiz katman: sınırlı istek/dk"),
    Provider("openrouter", "OpenRouter", _LM, "llm",
             [Field("OPENROUTER_API_KEY", "OpenRouter API Key")],
             free=False, signup_url="https://openrouter.ai/keys",
             note="tek anahtarla çok model"),

    # ═══════════════ FAZ 7 genişletmesi — BYOK kataloğu ═══════════════════════
    # Kural: ücretsiz olan varsayılan çalışır; ücretli olan yalnız kullanıcı
    # kendi anahtarını girerse devreye girer. `free` alanı UI'da rozet olur.

    # ───────────────────────── Türev / likidasyon ─────────────────────────────
    Provider("coinglass", "CoinGlass", _DR, "derivatives",
             [Field("COINGLASS_API_KEY", "CoinGlass API Key")],
             free=False, signup_url="https://www.coinglass.com/pricing",
             note="likidasyon haritası + OI geçmişi (~29-79 $/ay) — Binance public "
                  "uçlarının veremediği tarihsel likidasyon verisi"),
    Provider("laevitas", "Laevitas", _DR, "derivatives",
             [Field("LAEVITAS_API_KEY", "Laevitas API Key")],
             free=False, signup_url="https://www.laevitas.ch/",
             note="opsiyon yüzeyi, gamma maruziyeti"),
    Provider("amberdata", "Amberdata", _DR, "derivatives",
             [Field("AMBERDATA_API_KEY", "Amberdata API Key")],
             free=False, signup_url="https://www.amberdata.io/",
             note="kurumsal tick/order-book geçmişi"),

    # ───────────────────────── On-chain (ek) ──────────────────────────────────
    Provider("glassnode", "Glassnode", _OCP, "onchain",
             [Field("GLASSNODE_API_KEY", "Glassnode API Key")],
             free=False, signup_url="https://studio.glassnode.com/settings/api",
             note="borsa netflow, SOPR, HODL dalgaları"),
    Provider("cryptoquant", "CryptoQuant", _OCP, "onchain",
             [Field("CRYPTOQUANT_API_KEY", "CryptoQuant API Key")],
             free=False, signup_url="https://cryptoquant.com/pricing",
             note="borsa rezervi, madenci akışı"),
    Provider("nansen", "Nansen", _OCP, "onchain",
             [Field("NANSEN_API_KEY", "Nansen API Key")],
             free=False, signup_url="https://www.nansen.ai/",
             note="akıllı para etiketleri"),
    Provider("arkham", "Arkham", _OC, "onchain",
             [Field("ARKHAM_API_KEY", "Arkham API Key (ücretsiz katman)")],
             signup_url="https://platform.arkhamintelligence.com/",
             note="varlık/kuruluş etiketleme"),
    Provider("dune", "Dune Analytics", _OC, "onchain",
             [Field("DUNE_API_KEY", "Dune API Key")],
             signup_url="https://dune.com/settings/api",
             note="özel SQL sorguları; ücretsiz katman sınırlı"),
    Provider("bitquery", "Bitquery", _OC, "onchain",
             [Field("BITQUERY_API_KEY", "Bitquery API Key (ücretsiz)")],
             signup_url="https://account.bitquery.io/user/api_v2/access_tokens",
             note="GraphQL çok-zincir veri"),
    Provider("moralis", "Moralis", _OC, "onchain",
             [Field("MORALIS_API_KEY", "Moralis API Key (ücretsiz)")],
             signup_url="https://admin.moralis.io/settings",
             note="cüzdan/token API'si"),
    Provider("blockchair", "Blockchair", _OC, "onchain",
             [Field("BLOCKCHAIR_API_KEY", "Blockchair API Key (opsiyonel)")],
             signup_url="https://blockchair.com/api",
             note="anahtarsız da çalışır; anahtar limitleri yükseltir"),
    Provider("covalent", "Covalent", _OC, "onchain",
             [Field("COVALENT_API_KEY", "Covalent API Key (ücretsiz)")],
             signup_url="https://www.covalenthq.com/platform/",
             note="çok-zincir bakiye/işlem"),

    # ───────────────────────── Piyasa verisi (ek) ─────────────────────────────
    Provider("messari", "Messari", _MK, "market",
             [Field("MESSARI_API_KEY", "Messari API Key (ücretsiz)")],
             signup_url="https://messari.io/api",
             note="varlık temelleri, arz metrikleri"),
    Provider("coinpaprika", "CoinPaprika", _MK, "market",
             [Field("COINPAPRIKA_API_KEY", "CoinPaprika API Key (opsiyonel)")],
             signup_url="https://coinpaprika.com/api/",
             note="anahtarsız ücretsiz katman mevcut"),

    # ───────────────────────── Hisse / ETF ────────────────────────────────────
    Provider("polygon", "Polygon.io", _ST, "stocks",
             [Field("POLYGON_API_KEY", "Polygon.io API Key")],
             signup_url="https://polygon.io/dashboard/api-keys",
             note="ücretsiz katman 5 istek/dk; ETF/hisse tick verisi"),
    Provider("tiingo", "Tiingo", _ST, "stocks",
             [Field("TIINGO_API_KEY", "Tiingo API Key (ücretsiz)")],
             signup_url="https://www.tiingo.com/account/api/token",
             note="ETF/hisse günlük geçmiş"),
    Provider("iexcloud", "IEX Cloud", _ST, "stocks",
             [Field("IEX_API_KEY", "IEX Cloud API Key")],
             free=False, signup_url="https://iexcloud.io/console/tokens"),

    # ───────────────────────── Makro / haber (ek) ─────────────────────────────
    Provider("alphavantage", "Alpha Vantage", _MC, "macro",
             [Field("ALPHAVANTAGE_API_KEY", "Alpha Vantage API Key (ücretsiz)")],
             signup_url="https://www.alphavantage.co/support/#api-key",
             note="ücretsiz 25 istek/gün"),
    Provider("finnhub", "Finnhub", _MC, "macro",
             [Field("FINNHUB_API_KEY", "Finnhub API Key (ücretsiz)")],
             signup_url="https://finnhub.io/dashboard",
             note="ücretsiz 60 istek/dk — makro takvim + haber"),
    Provider("tradingeconomics", "Trading Economics", _MC, "macro",
             [Field("TRADINGECONOMICS_API_KEY", "Trading Economics API Key")],
             free=False, signup_url="https://tradingeconomics.com/api/",
             note="resmî makro takvim (beklenti/gerçekleşen)"),
    Provider("marketaux", "Marketaux", _MC, "macro",
             [Field("MARKETAUX_API_KEY", "Marketaux API Key (ücretsiz)")],
             signup_url="https://www.marketaux.com/account/dashboard",
             note="varlık etiketli finans haberi"),

    # ───────────────────────── Sosyal (ek) ────────────────────────────────────
    Provider("reddit", "Reddit", _SC, "social",
             [Field("REDDIT_CLIENT_ID", "Reddit Client ID"),
              Field("REDDIT_CLIENT_SECRET", "Reddit Client Secret")],
             signup_url="https://www.reddit.com/prefs/apps",
             note="ücretsiz; r/cryptocurrency duygu akışı"),
    Provider("santiment", "Santiment", _SC, "social",
             [Field("SANTIMENT_API_KEY", "Santiment API Key")],
             signup_url="https://app.santiment.net/account",
             note="ücretsiz katman sınırlı; sosyal hacim/duygu"),
]


# ════════════════════════════════════════════════════════════════════════════
#  Türetilmiş görünümler — diğer modüller BUNLARI kullanır (katalogu kopyalamaz)
# ════════════════════════════════════════════════════════════════════════════
def cred_schema() -> List[dict]:
    """Panel için düz alan listesi (server/app.py CRED_SCHEMA yerine)."""
    out: List[dict] = []
    for p in PROVIDERS:
        for f in p.fields:
            out.append({
                "key": f.key,
                "label": f.label,
                "group": p.group,
                "category": p.category,
                # `provider` GERİYE DÖNÜK olarak görünen addır (eski panel bunu
                # kullanıyordu). Kasa kayıtları KARARLI slug ile yapılır:
                "provider_id": p.id,
                "provider_name": p.name,
                "provider": p.name,
                "free": p.free,
                "signup_url": p.signup_url,
                "docs_url": p.docs_url,
                "note": p.note,
                "secret": f.secret,
                "ccxt_id": p.ccxt_id,
            })
    return out


def env_keys() -> List[str]:
    """Tüm ortam anahtarları (config.py ENV_KEYS yerine)."""
    keys: List[str] = []
    for p in PROVIDERS:
        for f in p.fields:
            keys.append(f.key)
    return keys


def valid_keys() -> set:
    return set(env_keys())


def exchanges() -> List[Provider]:
    return [p for p in PROVIDERS if p.category == "exchange" and p.ccxt_id]


def exchange_priority() -> Dict[str, float]:
    """ccxt_id -> birleştirme önceliği (exchange_manager EXCHANGE_PRIORITY yerine)."""
    return {p.ccxt_id: p.priority for p in exchanges()}


def default_exchanges() -> List[str]:
    """Public veri için varsayılan olarak denenecek ccxt id'leri (öncelik sırası)."""
    return [p.ccxt_id for p in sorted(exchanges(), key=lambda p: -p.priority)]


def exchange_credentials(config) -> Dict[str, Dict[str, str]]:
    """ccxt_id -> {apiKey, secret, password?} (yalnız dolu olanlar).
    exchange_manager bu yardımcıyı kullanarak pasaparolayı da otomatik geçirir."""
    creds: Dict[str, Dict[str, str]] = {}
    for p in exchanges():
        c = p.ccxt_id.upper()
        key = config.secret(f"{c}_API_KEY")
        sec = config.secret(f"{c}_SECRET")
        if key and sec:
            d = {"apiKey": key, "secret": sec}
            pw = config.secret(f"{c}_PASSWORD")
            if pw:
                d["password"] = pw
            creds[p.ccxt_id] = d
    return creds


def env_example() -> str:
    """.env.example içeriğini katalogdan üret (her zaman güncel kalsın)."""
    lines = [
        "# API anahtarları — bu dosyayı .env olarak kopyalayıp doldurun.",
        "# .env dosyasını ASLA git'e commit etmeyin. Anahtarlar yalnız ortamdan okunur.",
        "# Public OHLCV verisi için anahtar GEREKMEZ; anahtarlar yalnız hesap-özel",
        "# veri ve (bilinçli aktive edilirse) canlı emir içindir.",
        "# Bu dosya agi_trader/providers.py kataloğundan üretilir.",
        "",
    ]
    last_group = None
    for p in PROVIDERS:
        if p.group != last_group:
            tag = "ücretsiz" if p.free else "paralı/ücretli"
            lines.append(f"\n# --- {p.group} ---")
            last_group = p.group
        sfx = f"  ({p.note})" if p.note else ""
        link = f"  -> {p.signup_url}" if p.signup_url else ""
        lines.append(f"# {p.name}{sfx}{link}")
        for f in p.fields:
            lines.append(f"{f.key}=")
    return "\n".join(lines) + "\n"


def _self_check() -> List[str]:
    """Katalog tutarlılığını doğrula — yinelenen anahtar / id, ccxt eksikliği."""
    problems: List[str] = []
    seen_keys: Dict[str, str] = {}
    seen_ids = set()
    for p in PROVIDERS:
        if p.id in seen_ids:
            problems.append(f"yinelenen provider id: {p.id}")
        seen_ids.add(p.id)
        if p.category == "exchange" and not p.ccxt_id:
            problems.append(f"borsa '{p.id}' ccxt_id eksik")
        if not p.fields:
            problems.append(f"provider '{p.id}' alan(field) içermiyor")
        for f in p.fields:
            if f.key in seen_keys:
                problems.append(f"yinelenen anahtar '{f.key}' ({p.id} ve {seen_keys[f.key]})")
            seen_keys[f.key] = p.id
    return problems


if __name__ == "__main__":  # python -m agi_trader.providers
    import sys as _sys
    # Windows konsolunda (cp1254 vb.) Unicode rozet/emoji için UTF-8 çıktı
    try:
        _sys.stdout.reconfigure(encoding="utf-8")
        _sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(f"Toplam sağlayıcı: {len(PROVIDERS)}  ·  Toplam anahtar: {len(env_keys())}")
    by_cat: Dict[str, int] = {}
    for p in PROVIDERS:
        by_cat[p.category] = by_cat.get(p.category, 0) + 1
    print("Kategoriler:", ", ".join(f"{k}={v}" for k, v in by_cat.items()))
    print("Borsalar (ccxt, öncelik sırası):", ", ".join(default_exchanges()))
    probs = _self_check()
    if probs:
        print("\n⚠️  TUTARLILIK SORUNLARI:")
        for x in probs:
            print("  -", x)
        raise SystemExit(1)
    print("\n✅ Katalog tutarlı — yinelenen anahtar/id yok, tüm borsalar ccxt_id taşıyor.")
