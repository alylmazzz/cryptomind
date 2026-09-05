#!/usr/bin/env python3
"""FAZ6 — trend-takip PAPER daemon'ı (kalıcı durum).
  python trend_daemon.py            # tek günlük adım (cron için)
  python trend_daemon.py --loop     # sürekli döngü (24s aralık)
Durum: runs/trend_state.json. Canlı emir YOK.
"""
from __future__ import annotations
import sys, time, argparse, json
from pathlib import Path
import pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, str(Path(__file__).parent))
from agi_trader.config import load_config
from agi_trader.auto.trend_engine import TrendTrader
from agi_trader.monitor import risk_report
from agi_trader.notify.notifier import Notifier

NOTIF_LOG = Path(__file__).parent / "runs" / "notifications.log"


def notify(cfg, title, body):
    """Bildirim: Telegram (anahtar varsa) + her zaman runs/notifications.log."""
    line = f"{title} | {body}"
    try:
        with open(NOTIF_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    try:
        n = Notifier(cfg)
        if n.enabled:
            n.send_text(title, body)
    except Exception:
        pass
    print("  🔔 " + line, flush=True)

CRYPTO = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "AVAX/USDT"]
# Kripto-dışı diversifikasyon — VERİ-TEMELLİ SEÇİLMİŞ evren (select_universe.py).
# HYG/FXB/FXF/FXE forward-selection ile eklendi: Calmar 1.59→1.80, DD 13.3→10.6%.
NONCRYPTO = ["GLD", "SPY", "QQQ", "TLT", "UUP", "USO", "SLV", "DBC",
             "HYG", "FXB", "FXF", "FXE"]
STATE = Path(__file__).parent / "runs" / "trend_state.json"

# --- RISK RAYLARI ---------------------------------------------------------
# Ayni sinyal, farkli risk istahi. Hepsi PAPER; hepsi ayni gun ayni veriyle
# calisir (veri BIR KEZ cekilir), boylece aralarindaki fark yalniz KALDIRACTAN
# gelir - karsilastirma bilimsel olarak temizdir.
#   base       15% hedef vol / maks 2,5x  -> olculmus: yillik ~%40, DD %2,2
#   aggressive 45% hedef vol / maks 6x    -> ~%159, DD %6,6   (48 gunluk supurme)
#   extreme    75% hedef vol / maks 10x   -> ~%349, DD %11,0
#   max       225% hedef vol / maks 16x   -> %1/GUN hedefi (15,9x); DD ~%35, cokusta ~%49
#             ve TASFIYE gercek bir olasilik - ayrintisi trend_engine.RISK_TRACKS'te
# Kaldirac Sharpe'i ARTIRMAZ, yalnizca olcegi buyutur; bu yuzden her rayda
# dususe bagli kaldirac kisicisi (dd_soft/dd_hard/dd_kill) devrededir.
TRACKS = ["base", "aggressive", "extreme", "max"]


def state_path(track: str) -> Path:
    """base geriye donuk uyum icin eski dosya adini korur."""
    return STATE if track == "base" else STATE.parent / f"trend_state_{track}.json"


PRICE_CACHE = Path(__file__).parent / "runs" / "price_cache"


def _cache_write(t, df):
    try:
        PRICE_CACHE.mkdir(parents=True, exist_ok=True)
        df.tail(520).to_csv(PRICE_CACHE / f"{t.replace('/', '_')}.csv")
    except Exception:
        pass


def _cache_read(t, max_age_days=10):
    """Son başarılı çekimin kopyası. Kaynak düşerse portföy körleşmesin diye."""
    try:
        f = PRICE_CACHE / f"{t.replace('/', '_')}.csv"
        if not f.exists():
            return None
        age = (time.time() - f.stat().st_mtime) / 86400.0
        if age > max_age_days:
            return None
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        d = d.dropna(subset=["close"])
        return d if len(d) else None
    except Exception:
        return None


def _fetch_chart_api(t, rng="2y"):
    """Yedek kaynak: Yahoo'nun genel chart uç noktası, DOĞRUDAN (kütüphanesiz).

    Neden: sunucuda `yfinance` 429 alıyordu ama aynı hosta tarayıcı User-Agent'ıyla yapılan düz
    istek 200 dönüyor — sorun kaynağın kendisi değil, kütüphanenin istek/crumb akışı. Bu yüzden
    yedek katman kütüphaneyi atlayıp uç noktayı doğrudan çağırır."""
    import urllib.request
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
           f"?range={rng}&interval=1d")
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            j = json.loads(r.read().decode("utf-8", "ignore"))
        res = ((j.get("chart") or {}).get("result") or [None])[0]
        if not res:
            return None
        ts = res.get("timestamp") or []
        q = ((res.get("indicators") or {}).get("quote") or [{}])[0]
        adj = ((res.get("indicators") or {}).get("adjclose") or [{}])
        closes = (adj[0].get("adjclose") if adj and adj[0].get("adjclose") else q.get("close")) or []
        if not ts or not closes:
            return None
        d = pd.DataFrame({
            "open": q.get("open") or closes, "high": q.get("high") or closes,
            "low": q.get("low") or closes, "close": closes,
            "volume": q.get("volume") or [0.0] * len(ts),
        }, index=pd.to_datetime(pd.Series(ts), unit="s"))
        d = d.astype(float).dropna(subset=["close"])
        return d.tail(520) if len(d) else None
    except Exception:
        return None


def fetch_live_daily(pairs):
    """Kripto (ccxt) + kripto-dışı (yfinance → Stooq → önbellek) günlük OHLCV.

    ÜÇ KATMAN, çünkü tek kaynak düşünce (Yahoo 429) portföy 12 varlığı kaybedip özsermayeyi
    NaN yapıyordu. Sıra: canlı birincil → canlı yedek → son başarılı çekimin kopyası."""
    data = {}
    crypto = [p for p in pairs if "/" in p]
    noncrypto = [p for p in pairs if "/" not in p]
    if crypto:
        import ccxt
        ex = ccxt.binance({"enableRateLimit": True})
        since = ex.milliseconds() - 320*24*3600*1000
        for p in crypto:
            o = ex.fetch_ohlcv(p, "1d", since=since, limit=400)
            df = pd.DataFrame(o, columns=["ts","open","high","low","close","volume"])
            df.index = pd.to_datetime(df["ts"], unit="ms")
            data[p] = df[["open","high","low","close","volume"]].astype(float)
    if noncrypto:
        try:
            import yfinance as yf
        except Exception as e:
            print(f"  ⚠️ yfinance yok ({type(e).__name__}) — {len(noncrypto)} kripto-dışı varlık ATLANDI", flush=True)
            return data
        failed, via = [], {"yahoo": 0, "chart_api": 0, "önbellek": 0}
        for t in noncrypto:
            d = None
            try:
                y = yf.download(t, period="500d", interval="1d", progress=False, auto_adjust=True)
                if y is not None and len(y):
                    y.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in y.columns]
                    y = y[["open", "high", "low", "close", "volume"]].astype(float).dropna(subset=["close"])
                    if len(y):
                        d = y; via["yahoo"] += 1
            except Exception:
                pass
            if d is None:
                d = _fetch_chart_api(t)
                if d is not None:
                    via["chart_api"] += 1
            if d is None:
                d = _cache_read(t)
                if d is not None:
                    via["önbellek"] += 1
            if d is None:
                failed.append(t); continue
            data[t] = d
            if via["önbellek"] == 0 or t not in (failed or []):
                _cache_write(t, d)
        print(f"  kripto-dışı veri: yfinance {via['yahoo']} · chart-api {via['chart_api']} · önbellek {via['önbellek']}"
              + (f" · ALINAMAYAN {len(failed)}: {', '.join(failed)}" if failed else ""), flush=True)
        # SESSİZ YUTMA YASAK: veri gelmezse portföy o varlığı NaN'la işaretliyor ve equity kalıcı
        # bozuluyordu (2026-09-03/04 canlı arızası: 12 varlık düştü → 45 günlük kayıt okunamaz oldu).
        if failed:
            print(f"  ⚠️ VERİ ALINAMADI ({len(failed)}/{len(noncrypto)}): {', '.join(failed)}", flush=True)
    return data


def one_step(tt, cfg=None, data=None, notify_on=True, force=False):
    prev = {p for p, v in tt.weights.items() if abs(v) > 1e-4}
    if data is None:
        data = fetch_live_daily(tt.pairs)
    if not data:
        print("  HATA: hicbir varlik icin veri yok - adim atlandi", flush=True)
        return
    date = str(list(data.values())[0].index[-1])[:10]
    # AYNI GÜN İKİ KEZ ADIM ATMA. `step()` gün başına bir kez çağrılmak üzere yazıldı:
    # ikinci çağrı gün içi fiyat farkını "günlük getiri" sanıp işaretler, üstüne bir kez
    # daha yeniden dengeleme maliyeti düşer ve geçmişe aynı tarihten iki kayıt girer.
    # Cron'un iki kez tetiklenmesi ya da elle bir koşum, 49 günlük gerçek kaydı bozardı.
    if not force and tt.last_rebalance == date:
        print(f"[{date}] <{tt.track}> ATLANDI — bu gün zaten dengelendi "
              f"(equity {tt.equity:.2f}). Zorlamak için --force.", flush=True)
        return
    ev = tt.step(data, date_str=date)
    sigs = tt.signals(data)
    if ev.get("missing_prices"):
        print(f"    ⚠️ fiyatı gelmeyen varlık: {', '.join(ev['missing_prices'])} "
              f"(son bilinen kapanış korundu; işaretleme dışı bırakıldı)", flush=True)
    if getattr(tt, "recovered_from_nan", False):
        print("    🔧 bozuk (NaN) durum son sağlam güne kurtarıldı", flush=True)
    inmkt = [p for p in tt.pairs if sigs[p]["in_market"]]
    ddm = tt.dd_multiplier()
    print(f"[{date}] <{tt.track}> equity={ev['equity']:.2f} ({ev['return_pct']:+.2f}%) "
          f"yatırılan=%{ev['invested_pct']} dusus=%{tt.drawdown_pct()} "
          f"kaldirac_kisici={ddm:.2f}{' KILITLI' if tt.dd_locked else ''} "
          f"pozisyonda={inmkt or 'YOK (nakit)'}", flush=True)
    for p in tt.pairs:
        s = sigs[p]
        print(f"    {p:9s} {'🟢' if s['in_market'] else '⚪'} w=%{ev['targets'].get(p,0)*100:4.1f}  {s.get('reason','')}", flush=True)
    tt.save_state(str(state_path(tt.track)))

    # ---- BİLDİRİMLER ----  (yalnız seçilen rayda; üç ray üç kat bildirim demek olurdu)
    if cfg is not None and notify_on:
        now = set(inmkt)
        entered = now - prev; exited = prev - now
        if entered:
            notify(cfg, "🟢 YENİ POZİSYON", f"{date}: {', '.join(sorted(entered))} → LONG (yatırılan %{ev['invested_pct']})")
        if exited:
            notify(cfg, "⚪ POZİSYON KAPANDI", f"{date}: {', '.join(sorted(exited))} → NAKİT")
        if ev.get("missing_prices"):
            notify(cfg, "🟠 VERİ EKSİK", f"{date}: fiyatı alınamayan {len(ev['missing_prices'])} varlık "
                                         f"({', '.join(ev['missing_prices'][:6])}) — portföy işaretlemesi eksik")
        try:
            r = risk_report(json.loads(state_path(tt.track).read_text(encoding="utf-8")))
            if r.get("health") == "red":
                notify(cfg, "🔴 RİSK ALARMI", f"{date}: " + " ".join(a for a in r.get("alerts", []) if "🔴" in a))
            elif r.get("health") == "yellow":
                notify(cfg, "🟡 RİSK UYARISI", f"{date}: " + " ".join(a for a in r.get("alerts", []) if "🟡" in a))
        except Exception:
            pass
        # Yuksek kaldiracli raylarda dusus kisicisi devreye girdiginde SESSIZ KALMA:
        # kaldiracin kendiliginden yariya dusmesi, sistemin "iyi gidiyor" gorunurken
        # riski azalttigi andir; operator bunu gormeli.
        if tt.track != "base" and ddm < 1.0:
            notify(cfg, "🟠 KALDIRAÇ KISILDI",
                   f"{date} <{tt.track}>: düşüş %{tt.drawdown_pct()} → kaldıraç çarpanı {ddm:.2f}"
                   + (" (KİLİTLİ — nakit)" if tt.dd_locked else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=86400)
    ap.add_argument("--universe", choices=["diversified", "crypto"], default="diversified",
                    help="diversified = kripto+altın/endeks/tahvil (Sharpe 1.33); crypto = sadece 5 kripto (1.07)")
    ap.add_argument("--tracks", default=",".join(TRACKS),
                    help="risk rayları (virgülle): base,aggressive,extreme")
    ap.add_argument("--notify-track", default="base", help="bildirimleri hangi ray göndersin")
    ap.add_argument("--force", action="store_true",
                    help="aynı gün zaten dengelenmiş olsa bile adım at (kaydı bozabilir)")
    args = ap.parse_args()
    c = load_config()
    pairs = CRYPTO + NONCRYPTO if args.universe == "diversified" else CRYPTO
    tracks = [t.strip() for t in str(args.tracks).split(",") if t.strip()]
    print(f"Evren: {args.universe} ({len(pairs)} varlık) · raylar: {', '.join(tracks)}")
    traders = []
    for t in tracks:
        tt = TrendTrader(c, pairs=pairs, initial=10000, track=t)
        if tt.load_state(str(state_path(t))):
            print(f"  <{t}> durum yüklendi: equity {tt.equity:.2f}, tepe {tt.peak_equity:.2f}, son {tt.last_rebalance}")
        else:
            print(f"  <{t}> yeni paper portföy (10000 USDT) · hedef vol %{tt.target_vol_a*100:.0f} · maks kaldıraç {tt.max_lev}x")
        traders.append(tt)

    def _cycle():
        # Veri BİR KEZ çekilir ve bütün raylara aynı gün/aynı fiyatla uygulanır:
        # raylar arası fark yalnız kaldıraçtan gelsin (ve veri kaynağı 3× yüklenmesin).
        data = fetch_live_daily(pairs)
        for tt in traders:
            try:
                one_step(tt, c, data=data, notify_on=(tt.track == args.notify_track),
                         force=args.force)
            except Exception as e:
                print(f"  <{tt.track}> adım hatası: {type(e).__name__}: {e}", flush=True)

    if args.loop:
        print(f"Sürekli döngü (her {args.interval}s). Ctrl+C ile dur.")
        while True:
            try:
                _cycle()
            except Exception as e:
                print(f"adım hatası: {type(e).__name__}: {e}", flush=True)
            time.sleep(args.interval)
    else:
        _cycle()

if __name__ == "__main__":
    main()
