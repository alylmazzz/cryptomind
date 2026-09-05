"""
Sosyal/haber olay toplayıcı (FAZ 6) — olay çalışmasının yakıtı.

NEDEN GEREKLİ: X API Basic (200 $/ay) **geçmişe dönük arama VERMEZ** — yalnız
son ~7 gün ve sınırlı kota. Dolayısıyla "hangi hesap fiyat hareket ettiriyor"
sorusu ancak İLERİYE DÖNÜK toplanan veriyle cevaplanabilir. Bu, piyasa
özelliklerinde `data/recorder.py` ile yapılanın sosyal karşılığıdır: kayıt
bugün başlamazsa altı ay sonra da başlanamaz.

KAYNAK KATMANLARI (aşağı doğru zarif düşüş)
  1. X API (BYOK)      — kullanıcı kendi anahtarını kasaya girerse gerçek akış
  2. CryptoPanic       — ücretsiz katman; haber + topluluk oyu
  3. RSS               — anahtarsız; CoinDesk/Cointelegraph başlıkları
Anahtar yoksa sistem 2-3'le çalışır ve bunu açıkça bildirir; sessizce boş
dönmez.

Çıktı: `runs/social/YYYY-MM.parquet` (yoksa .csv.gz)
  ts · source · handle · text · sentiment · assets · url
"""
from __future__ import annotations

import gzip
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Varlık çıkarımı için basit sözlük (analysis tarafıyla tutarlı)
ASSET_PATTERNS = {
    "BTC": r"\b(btc|bitcoin)\b", "ETH": r"\b(eth|ethereum)\b",
    "SOL": r"\b(sol|solana)\b", "DOGE": r"\b(doge|dogecoin)\b",
    "AVAX": r"\b(avax|avalanche)\b", "XRP": r"\b(xrp|ripple)\b",
    "BNB": r"\b(bnb|binance coin)\b", "ADA": r"\b(ada|cardano)\b",
}

RSS_FEEDS = [
    ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("cointelegraph", "https://cointelegraph.com/rss"),
    ("bitcoinmagazine", "https://bitcoinmagazine.com/feed"),
]


def _out_dir() -> Path:
    p = Path(__file__).resolve().parents[2] / "runs" / "social"
    p.mkdir(parents=True, exist_ok=True)
    return p


def extract_assets(text: str) -> List[str]:
    t = (text or "").lower()
    return [a for a, pat in ASSET_PATTERNS.items() if re.search(pat, t)]


def score_text(text: str, ti=None) -> float:
    """Duygu skoru −1..+1. TwitterIntelligence varsa onun sözlüğü/modeli kullanılır."""
    if ti is not None:
        try:
            return float(ti._score_text(text))
        except Exception:
            pass
    t = (text or "").lower()
    pos = sum(w in t for w in ("bullish", "buy", "long", "surge", "rally", "breakout",
                               "pump", "moon", "ath", "adoption", "approve"))
    neg = sum(w in t for w in ("bearish", "sell", "short", "crash", "dump", "hack",
                               "ban", "lawsuit", "exploit", "liquidation", "reject"))
    if pos == neg:
        return 0.0
    return float(max(-1.0, min(1.0, (pos - neg) / max(1, pos + neg))))


# ===========================================================================
# Kaynaklar
# ===========================================================================
def from_x(handles: List[str], config, ti=None, per_handle: int = 5) -> List[Dict]:
    """X API (BYOK). Anahtar yoksa boş liste + neden döner."""
    rows: List[Dict] = []
    if ti is None or not getattr(ti, "live", lambda: False)():
        return rows
    for h in handles:
        try:
            texts = ti.fetch_recent(h, max_results=per_handle) or []
        except Exception:
            continue
        for t in texts:
            rows.append({"ts": pd.Timestamp.now("UTC"), "source": "x", "handle": h,
                         "text": t, "sentiment": score_text(t, ti),
                         "assets": ",".join(extract_assets(t)), "url": ""})
    return rows


def from_cryptopanic(api_key: Optional[str], limit: int = 50) -> List[Dict]:
    """CryptoPanic ücretsiz katmanı. Anahtarsız da sınırlı çalışır."""
    import requests
    try:
        params = {"public": "true", "kind": "news"}
        if api_key:
            params["auth_token"] = api_key
        r = requests.get("https://cryptopanic.com/api/v1/posts/", params=params, timeout=12)
        if r.status_code != 200:
            return []
        items = (r.json() or {}).get("results", [])[:limit]
    except Exception:
        return []
    rows = []
    for it in items:
        title = it.get("title") or ""
        src = ((it.get("source") or {}).get("title")) or "cryptopanic"
        ts = it.get("published_at") or it.get("created_at")
        rows.append({"ts": pd.to_datetime(ts, utc=True, errors="coerce") or pd.Timestamp.now("UTC"),
                     "source": "cryptopanic", "handle": src, "text": title,
                     "sentiment": score_text(title),
                     "assets": ",".join(extract_assets(title)),
                     "url": it.get("url", "")})
    return rows


def from_rss(limit_per_feed: int = 20) -> List[Dict]:
    """Anahtarsız RSS başlıkları — her zaman çalışan taban katman."""
    import requests
    rows: List[Dict] = []
    for name, url in RSS_FEEDS:
        try:
            r = requests.get(url, timeout=12,
                             headers={"User-Agent": "Mozilla/5.0 (CryptoMind)"})
            if r.status_code != 200:
                continue
            titles = re.findall(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>",
                                r.text, re.S)[1:limit_per_feed + 1]
            dates = re.findall(r"<pubDate>(.*?)</pubDate>", r.text, re.S)[:limit_per_feed]
        except Exception:
            continue
        for i, t in enumerate(titles):
            t = re.sub(r"<[^>]+>", "", t).strip()
            if not t:
                continue
            ts = pd.Timestamp.now("UTC")
            if i < len(dates):
                try:
                    ts = pd.to_datetime(dates[i], utc=True, errors="coerce") or ts
                except Exception:
                    pass
            rows.append({"ts": ts, "source": "rss", "handle": name, "text": t,
                         "sentiment": score_text(t),
                         "assets": ",".join(extract_assets(t)), "url": url})
    return rows


# ===========================================================================
# Depolama + döngü
# ===========================================================================
def append(rows: List[Dict]) -> Optional[Path]:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    if df.empty:
        return None
    base = _out_dir() / df["ts"].iloc[0].strftime("%Y-%m")
    try:
        import pyarrow  # noqa: F401
        path = base.with_suffix(".parquet")
        if path.exists():
            old = pd.read_parquet(path)
            df = pd.concat([old, df], ignore_index=True)
            # aynı metni iki kez sayma (RSS/CryptoPanic tekrarları)
            df = df.drop_duplicates(subset=["source", "handle", "text"], keep="first")
        df.to_parquet(path, index=False)
        return path
    except Exception:
        path = base.with_suffix(".csv.gz")
        header = not path.exists()
        with gzip.open(path, "at", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False, header=header)
        return path


def load_events(month: Optional[str] = None) -> pd.DataFrame:
    d = _out_dir()
    files = sorted(list(d.glob("*.parquet")) + list(d.glob("*.csv.gz")))
    if month:
        files = [p for p in files if p.name.startswith(month)]
    frames = []
    for p in files:
        try:
            frames.append(pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["ts", "source", "handle", "text",
                                     "sentiment", "assets", "url"])
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    return out.dropna(subset=["ts"]).sort_values("ts")


def collect_once(config=None, handles: Optional[List[str]] = None) -> Dict:
    """Tüm kaynaklardan bir tur topla ve kaydet."""
    ti = None
    x_live = False
    if config is not None:
        try:
            from .twitter_intelligence import TwitterIntelligence
            ti = TwitterIntelligence(config)
            x_live = bool(ti.live())
        except Exception:
            ti = None

    if handles is None:
        try:
            from .accounts import CRITICAL_ACCOUNTS
            handles = [h for cat in CRITICAL_ACCOUNTS.values() for h in cat][:40]
        except Exception:
            handles = []

    rows: List[Dict] = []
    if x_live:
        rows += from_x(handles, config, ti)
    cp_key = config.secret("CRYPTOPANIC_API_KEY") if config else None
    rows += from_cryptopanic(cp_key)
    rows += from_rss()

    path = append(rows)
    return {"collected": len(rows), "x_live": x_live,
            "sources": sorted({r["source"] for r in rows}),
            "file": (path.name if path else None),
            "note": ("X akışı AÇIK (BYOK anahtarı bulundu)" if x_live else
                     "X akışı KAPALI — anahtar yok; ücretsiz haber kaynakları "
                     "kullanılıyor. Kendi X anahtarınızı /#hesap sayfasından girebilirsiniz.")}


def run(config=None, interval: int = 1800, once: bool = False) -> None:
    print(f"[social] her {interval}s → {_out_dir()}", flush=True)
    while True:
        t0 = time.time()
        try:
            r = collect_once(config)
            print(f"[social] {pd.Timestamp.now("UTC"):%Y-%m-%d %H:%M:%S} "
                  f"{r['collected']} kayıt · kaynak {r['sources']} · "
                  f"X={'açık' if r['x_live'] else 'kapalı'}", flush=True)
        except Exception as e:
            print(f"[social] hata: {type(e).__name__}: {e}", flush=True)
        if once:
            return
        time.sleep(max(60.0, interval - (time.time() - t0)))
