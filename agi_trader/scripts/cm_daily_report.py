#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CryptoMind GÜNLÜK RAPOR — anlık sonuç: P&L, çıkışlar, tepe yakalama, komisyon, kaçırılanlar, kapı isabeti,
kalibrasyon (Brier), Monte Carlo (drawdown/iflas olasılığı), challenger, kaynak, defter zinciri.

  python scripts/cm_daily_report.py                         # yerel: http://127.0.0.1:8210
  python scripts/cm_daily_report.py --url https://mindcorplab.com/cryptomind
  python scripts/cm_daily_report.py --md runs/reports        # Markdown'a da yaz (cron için)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
def _utf8_stdout():
    if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "").lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def get(url: str):
    with urllib.request.urlopen(url, timeout=40) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8210")
    ap.add_argument("--md", default="", help="Markdown çıktı klasörü")
    a = ap.parse_args()
    from agi_trader.learn import calibration as CAL
    base = a.url.rstrip("/")
    sim = get(f"{base}/api/simulator"); feed = get(f"{base}/api/simulator/feed"); missed = get(f"{base}/api/simulator/missed?limit=30")
    st = sim["stats"]; trades = sim.get("trades") or []
    day = time.strftime("%Y-%m-%d", time.gmtime())
    today = [t for t in trades if time.strftime("%Y-%m-%d", time.gmtime(t["closed_ts"])) == day]
    L = []
    L.append(f"# CryptoMind günlük rapor — {day} (UTC)")
    L.append(f"- Özsermaye **{st['equity']} $** · toplam net {st['net_pnl']:+.2f} $ · brüt {st['gross_pnl']:+.2f} · komisyon {st['fees_paid']:.2f} (brütün %{st.get('fee_share_of_gross_pct')})")
    L.append(f"- Kapanan {st['closed_trades']} (bugün {len(today)}) · kazanma %{st['win_rate']} · kâr faktörü {st['profit_factor']} · ort. tepe yakalama {st.get('avg_peak_capture')} · maker payı %{st.get('maker_share_pct')}")
    L.append(f"- Çıkış sebepleri: {st.get('exit_reasons')} · tutma: { {k: v['n'] for k, v in st.get('hold_buckets', {}).items()} }")
    L.append(f"- Sleeve bazında: " + " · ".join(f"{k} n{v['n']} net {v['net']:+.2f} ({v['wins']}/{v['n']})" for k, v in (st.get('by_sleeve') or {}).items()))
    pm = sim.get("portfolio_mode") or {}; rs = sim.get("resource") or {}
    L.append(f"- Portföy modu {pm.get('mode')} ({', '.join((pm.get('reasons') or [])[:2]) or '—'}) · kaynak {rs.get('state')} RSS {rs.get('rss_mb')} MB · döngü {rs.get('cycle_sec')} sn · açık {len(sim.get('positions') or [])}")
    if today:
        best = max(today, key=lambda t: t["net_pnl"]); worst = min(today, key=lambda t: t["net_pnl"])
        L.append(f"- En iyi: {best['symbol']} {best['sleeve']} {best['reason']} {best['net_pnl']:+.2f} $ · En kötü: {worst['symbol']} {worst['sleeve']} {worst['reason']} {worst['net_pnl']:+.2f} $")
    L.append("")
    L.append("## Kaçırılanlar / kapı isabeti")
    L.append(f"- kaçırılan kazanç {missed['n_missed']} (net toplam %{missed['missed_net_pct_sum']}) · doğru kaçınma {missed['n_avoided']} · gölgede {missed['n_open']}")
    for g in (missed.get("gates") or [])[:8]:
        L.append(f"  - {g['gate_tr']}: n {g['n']} · kaçırılan {g['missed']} · kaçınılan {g['avoided']} · {g['verdict']}")
    wp, sp = missed.get("winner_profile") or {}, missed.get("stop_profile") or {}
    if wp.get("n"):
        L.append(f"- Kazanan profili: n {wp['n']} · medyan {wp.get('median_minutes')} dk · MAE/stop {wp.get('median_mae_over_stop')} · {wp.get('lesson')}")
    if sp.get("n"):
        L.append(f"- Stop profili: n {sp['n']} · MFE/hedef {sp.get('median_mfe_over_target')} · uyaranlar {sp.get('common_warnings')} · {sp.get('lesson')}")
    if missed.get("proposals"):
        L.append(f"- Challenger önerileri: {missed['proposals']}")
    L.append("")
    L.append("## Kalibrasyon ve Monte Carlo")
    cal = CAL.reliability_table(trades)
    L.append(f"- Brier {cal.get('brier')} (referans {cal.get('brier_reference')}, beceri {cal.get('skill')}) → **{cal.get('verdict')}** · n {cal.get('n')}")
    for b in cal.get("bins") or []:
        L.append(f"  - p {b['bin']}: n {b['n']} · tahmin {b['p_mean']} · gerçekleşen {b['realized']} (fark {b['gap']:+.2f})")
    mc = CAL.monte_carlo(trades, capital=float(sim["config"]["capital_usdt"]), daily_loss_limit_pct=float(sim["config"]["daily_loss_limit_pct"]) * 100)
    if mc.get("paths"):
        L.append(f"- Monte Carlo ({mc['paths']} yol, {mc['horizon_trades']} işlem ufku): son özsermaye P5 {mc['final_p5']} · P50 {mc['final_p50']} · P95 {mc['final_p95']} · "
                 f"maks DD P50 %{mc['max_dd_p50_pct']} / P95 %{mc['max_dd_p95_pct']} · günlük limit aşma olasılığı {mc['p_breach_daily_limit']} · iflas (DD≥%30) {mc['p_ruin']}")
    else:
        L.append(f"- Monte Carlo: {mc.get('note')}")
    ch = sim.get("challenger") or {}
    L.append(f"- Challenger: {ch.get('params') or '—'} · gölge {ch.get('n_challenger_only')} · terfi {ch.get('promote')} · red {ch.get('reject')}")
    led = sim.get("ledger") or {}
    L.append(f"- Defter zinciri: {led.get('ok', '—')} ({led.get('n', 0)} kayıt) · uyarı kanalı: {'açık' if (sim.get('alerts') or {}).get('configured') else 'yapılandırılmadı'}")
    text = "\n".join(L)
    print(text)
    if a.md:
        out = Path(a.md); out.mkdir(parents=True, exist_ok=True)
        (out / f"{day}.md").write_text(text + "\n", encoding="utf-8")
        print(f"\nyazıldı: {out / (day + '.md')}")
    return 0


if __name__ == "__main__":
    _utf8_stdout()
    sys.exit(main())
