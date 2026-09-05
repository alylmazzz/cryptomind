"""Açıklanabilir sinyal raporu biçimlendirici (Explainable AI çıktısı)."""
from __future__ import annotations

from typing import List

from .core.models import Direction, TradeSignal


def format_signal(sig: TradeSignal) -> str:
    L = []
    head = f"  {sig.symbol}  [{sig.timeframe}]  "
    L.append("═" * 70)
    L.append(head)
    L.append("═" * 70)

    badge = "✅ İŞLEM ADAYI" if sig.actionable else "⛔ İŞLEM YOK (eşik altı)"
    L.append(f"Yön           : {sig.direction.value}   ({sig.bias.value})   {badge}")
    L.append(f"Güven (conf.) : {sig.confidence*100:.1f} %   | Başarı olasılığı: {sig.success_probability*100:.0f} %")
    L.append(f"Alış / Satış  : 🟢 %{sig.buy_pressure_pct:.1f}  vs  🔴 %{sig.sell_pressure_pct:.1f}   → {sig.pressure_label}")
    L.append(f"Fiyat / Entry : {sig.entry:.6f}")

    # Sonraki periyot maks/min tahmini
    if sig.forecast:
        fc = sig.forecast
        L.append("- " * 35)
        L.append(f"SONRAKİ {sig.timeframe} TAHMİNİ (yukarı olasılığı %{fc['prob_up']*100:.0f}):")
        L.append(f"  Beklenen MAKS : {fc['expected_high']:.6f}  ({fc['up_move_pct']:+.2f}%)")
        L.append(f"  Beklenen MİN  : {fc['expected_low']:.6f}  ({fc['down_move_pct']:+.2f}%)")
        L.append(f"  Beklenen kapanış: {fc['expected_close']:.6f}")
        L.append(f"  %68 aralık: [{fc['band68'][0]:.4f} – {fc['band68'][1]:.4f}]  "
                 f"| %95 aralık: [{fc['band95'][0]:.4f} – {fc['band95'][1]:.4f}]")

    if sig.direction != Direction.FLAT:
        L.append(f"Stop-Loss     : {sig.stop_loss:.6f}   (iptal: {sig.invalidation:.6f})")
        tp = sig.take_profits
        L.append(f"Take-Profit   : TP1 {tp[0]:.6f} | TP2 {tp[1]:.6f} | TP3 {tp[2]:.6f}")
        L.append(f"Risk / Reward : {sig.risk_reward:.2f}   | Beklenen +{sig.expected_return_pct:.2f}% / -{sig.expected_loss_pct:.2f}%")
        if sig.risk:
            r = sig.risk
            L.append(f"Pozisyon      : {r.recommended_position_size:.2f} USDT (%{r.position_pct:.2f}) | Kelly {r.kelly_fraction:.3f}")
            L.append(f"Risk metrik   : VaR95 {r.value_at_risk_95:.2f} | CVaR {r.conditional_var:.2f} | beklenen Sharpe {r.expected_sharpe:.2f}")
            L.append(f"Monte Carlo   : kazanç {r.mc_win_probability:.0f}% | ort. getiri {r.mc_expected_return:+.1f}% | ort. maxDD {r.mc_max_drawdown:.1f}%")

    L.append("-" * 70)
    L.append("Gerekçe (özet):")
    for r in sig.reasons:
        L.append(f"  • {r}")

    L.append("-" * 70)
    L.append("Katman kırılımı (ağırlıklı katkıya göre):")
    for b in sig.layer_breakdown:
        if b.get("layer", "").startswith("_"):
            continue
        L.append(f"  [{b['layer']:<16}] {b['bias']:<5} skor {b['score']:+.2f} "
                 f"× güven {b['confidence']:.2f} × ağ. {b['weight']:.2f} = katkı {b['contribution']:+.3f}")
        for rr in (b.get("reasons") or [])[:2]:
            L.append(f"        - {rr}")

    # extremes
    ext = next((b["detail"] for b in sig.layer_breakdown if b.get("layer") == "_extremes"), None)
    if ext:
        L.append("-" * 70)
        L.append(f"Maks/Min konumu: ATH'den {ext['pct_from_high']:.1f}% | dipten +{ext['pct_from_low']:.1f}% "
                 f"| range konumu {ext['range_position']*100:.0f}%")

    L.append("-" * 70)
    L.append(f"Alternatif senaryo: {sig.alternative_scenario}")

    ex = next((b["detail"] for b in sig.layer_breakdown if b.get("layer") == "_execution"), None)
    if ex:
        L.append(f"Execution      : {ex.get('action')} — {ex.get('reason', '')}".rstrip(" —"))
    L.append("")
    return "\n".join(L)


def format_all(signals: List[TradeSignal], env: dict) -> str:
    out = []
    out.append("┌" + "─" * 68 + "┐")
    out.append("│  AGI TRADER — Çok Katmanlı Açıklanabilir Karar Motoru             │")
    out.append("└" + "─" * 68 + "┘")
    out.append(f"Veri kaynağı : {env['data_source']}")
    out.append(f"On-chain     : {'CANLI whale/funding/OI' if env.get('onchain_live') else 'proxy (anahtar yok)'} "
               f"| Derin öğrenme: {'AÇIK (Transformer+LSTM+RL)' if env.get('deep_learning') else 'kapalı'}")
    out.append(f"Twitter      : {'CANLI' if env['twitter_live'] else 'kapalı (anahtar yok)'} "
               f"| izlenen hesap: {env['tracked_accounts']} "
               f"({env.get('accounts_crypto', 0)} kripto + {env.get('accounts_political', 0)} siyasi/makro)")
    es = env["execution"]
    out.append(f"Execution    : mod={es['mode']} | canlı={es['is_live']} | kill-switch={es['killed']}")
    out.append("")
    for sig in signals:
        out.append(format_signal(sig))

    actionable = [s for s in signals if s.actionable]
    out.append("═" * 70)
    out.append(f"ÖZET: {len(signals)} parite analiz edildi, {len(actionable)} işlem adayı "
               f"(güven ≥ %{int(signals[0].confidence*0+90) if signals else 90} eşiği).")
    if actionable:
        for s in actionable:
            out.append(f"   → {s.symbol}: {s.direction.value} @ {s.entry:.4f} "
                       f"(güven %{s.confidence*100:.0f}, R/R {s.risk_reward:.2f})")
    return "\n".join(out)
