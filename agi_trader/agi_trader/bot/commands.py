"""
Bot Komut İşleyici — Telegram / Discord (ve dashboard) için metin komutları.

Komutlar (TR + EN):
  /yardim /help            → komut listesi
  /durum /status           → ortam + portföy özeti
  /portfoy /portfolio      → otonom motor kazanım özeti
  /analiz <SEMBOL>         → tek parite hızlı sinyal (yön/güven/eşik/maks-min)
  /sektor /sectors         → sektör rotasyonu özeti
  /alarmlar /alarms        → son alarmlar
  /pozisyonlar /positions  → açık pozisyonlar
"""
from __future__ import annotations

from typing import Optional


def _fmt(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    return f"{n:,.2f}" if abs(n) < 1000 else f"{n:,.0f}"


def process_command(orch, auto, text: str) -> str:
    parts = (text or "").strip().split()
    if not parts:
        return "Komut bekleniyor. /yardim yazın."
    cmd = parts[0].lower().lstrip("/")
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("yardim", "help", "start"):
        return ("🤖 AGI Trader komutları:\n"
                "/durum — ortam + portföy\n/portfoy — kazanım özeti\n"
                "/analiz <SEMBOL> — hızlı sinyal\n/sektor — sektör rotasyonu\n"
                "/pozisyonlar — açık pozisyonlar\n/alarmlar — son alarmlar")

    if cmd in ("durum", "status"):
        env = orch.describe_environment()
        st = auto.status()
        pf = st["portfolio"]
        fg = env.get("fear_greed") or {}
        return (f"📊 DURUM\nVeri: {env['data_source'][:40]}\n"
                f"Motor: {'● çalışıyor' if st['running'] else '○ durdu'} ({st['mode']})\n"
                f"Özsermaye: ${_fmt(pf['equity'])} (%{pf['return_pct']})\n"
                f"Açık poz: {pf['open_positions']} · Kazanma %{pf['win_rate']}\n"
                f"Korku/Açgözlülük: {fg.get('value','—')} {fg.get('label','')}")

    if cmd in ("portfoy", "portfolio", "portföy"):
        pf = auto.status()["portfolio"]
        return (f"💼 PORTFÖY\nÖzsermaye: ${_fmt(pf['equity'])} (başlangıç ${_fmt(pf['initial'])})\n"
                f"Getiri: %{pf['return_pct']} · Gerçekleşen ${_fmt(pf['realized_pnl'])}\n"
                f"Açık PnL: ${_fmt(pf['unrealized_pnl'])}\n"
                f"İşlem: {pf['closed_trades']} · Kazanma %{pf['win_rate']} · PF {pf['profit_factor']}\n"
                f"Max DD: %{pf['max_drawdown_pct']}")

    if cmd in ("analiz", "analyze", "analiz"):
        if not arg:
            return "Kullanım: /analiz BTC/USDT"
        sym = arg.upper()
        if "/" not in sym and not sym.endswith(".IS"):
            sym = sym + "/USDT"
        try:
            sig = orch.analyze_symbol(sym)
        except Exception as e:
            return f"❌ {sym} analiz edilemedi: {type(e).__name__}"
        fc = sig.forecast or {}
        cls = {"kesin_al": "KESİN AL", "zayif_al": "ZAYIF AL", "notr": "NÖTR",
               "zayif_sat": "ZAYIF SAT", "kesin_sat": "KESİN SAT", "acil_cikis": "ACİL ÇIKIŞ"}.get(sig.signal_class, sig.signal_class)
        ico = "🟢" if sig.direction.value == "LONG" else "🔴" if sig.direction.value == "SHORT" else "⚪"
        return (f"{ico} {sym} [{sig.timeframe}] — {cls}\n"
                f"Yön: {sig.direction.value} · güven %{sig.confidence*100:.0f}\n"
                f"Şu an: {_fmt(sig.entry)}\n"
                f"🟢 AL eşiği: {_fmt(fc.get('buy_threshold'))} · 🔴 SAT eşiği: {_fmt(fc.get('sell_threshold'))}\n"
                f"▲ MAKS: {_fmt(fc.get('expected_high'))} · ▼ MİN: {_fmt(fc.get('expected_low'))}\n"
                f"Al %{sig.buy_pressure_pct} / Sat %{sig.sell_pressure_pct} · "
                f"{'✅ işlem adayı' if sig.actionable else 'izle'}")

    if cmd in ("sektor", "sektör", "sectors"):
        from ..macro import compute_sector_rotation
        r = compute_sector_rotation(orch, "1d")
        top = r["sectors"][:5]
        lines = "\n".join(f"{'▲' if s['mom7']>=0 else '▼'} {s['sector']}: %{s['mom7']}"
                          + (" 🔥" if s["accelerating"] else "") for s in top)
        return f"🔄 SEKTÖR ROTASYONU\n{r['note']}\n\n{lines}"

    if cmd in ("pozisyonlar", "positions", "pozisyon"):
        ps = auto.full_state()["positions"]
        if not ps:
            return "💼 Açık pozisyon yok."
        lines = "\n".join(f"{p['symbol']} {p['direction']} · giriş {_fmt(p['entry'])} · "
                          f"PnL {p['unrealized']:+.2f} (%{p['pnl_pct']})" for p in ps)
        return f"💼 AÇIK POZİSYONLAR\n{lines}"

    if cmd in ("alarmlar", "alarms", "alarm"):
        al = auto.orch.alarms.recent(6) if hasattr(auto.orch, "alarms") else []
        if not al:
            return "🔔 Henüz alarm yok."
        lines = "\n".join(f"{a['kind']}: {a['message'][:70]}" for a in al)
        return f"🔔 SON ALARMLAR\n{lines}"

    return f"Bilinmeyen komut: {cmd}. /yardim yazın."
