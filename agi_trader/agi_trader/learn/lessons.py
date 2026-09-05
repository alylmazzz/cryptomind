"""
DERS MOTORU — her işlemden sonra "nasıl kazanılır, nasıl kaybedilmez" sorusunu
VERİYLE cevaplar ve sonraki işlemi değiştirir.

Dört bellek:
  1. ROL GÜVENİLİRLİĞİ  — her rolün oyu gerçekleşen yönle uyuştu mu? (Beta(1,1) öncülü)
  2. ÖZELLİK İSTATİSTİĞİ — rejim, tutma kovası, çıkış sebebi, tetikleyici, emir tipi,
                           saat dilimi × sonuç
  3. GÖLGE TAKİP          — VETO edilen adaylar ve STOP olan işlemler ufuk boyunca izlenir:
                           "gözardı ettiğimiz şey kazanacak mıydı?" (karşı-olgusal)
  4. DERSLER              — kanıt eşiği geçilince (n ≥ N, oran sınırı) yazılan kural +
                           sınırlar içinde uygulanan parametre değişikliği

İlkeler (bu projede pahalıya öğrenildi):
  • Kanıtsız ders YAZILMAZ; her dersin n, oran ve etkisi yazılır.
  • Parametreler yalnız SINIR içinde ve soğuma süresiyle değişir (aşırı uyum freni).
  • Kill-switch eşiklerine dokunulmaz — güvenlik öğrenmeyle gevşemez.
  • Günlük hem makine (jsonl) hem insan (Markdown) için yazılır.
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

# parametre sınırları: (alt, üst, adım)
BOUNDS: Dict[str, tuple] = {
    "min_hold_sec": (0, 3600, 300),
    "min_gross_to_cost": (1.5, 5.0, 0.5),
    "rr": (1.0, 3.0, 0.2),
    "stop_sigma_mult": (0.75, 3.0, 0.25),
    "giveback": (0.3, 0.8, 0.1),
    "theta": (0.1, 0.6, 0.05),
    "counter_trend_mult": (0.1, 1.0, 0.15),
    "chase_taker_ratio": (1.5, 6.0, 0.5),
}
MIN_N = 8
COOLDOWN_SEC = 24 * 3600
SHADOW_MAX = 300


def _beta_mean(a: float, b: float) -> float:
    return (a + 1.0) / (a + b + 2.0)


class LessonEngine:
    def __init__(self, path: Path, journal_md: Optional[Path] = None,
                 journal_jsonl: Optional[Path] = None):
        self.path = Path(path)
        self.md = journal_md
        self.jsonl = journal_jsonl
        self.role_stats: Dict[str, Dict[str, float]] = {}     # role -> {agree, disagree, n}
        self.symbol_stats: Dict[str, Dict] = {}
        self.feature_stats: Dict[str, Dict[str, Dict]] = {}    # dim -> bucket -> {n, wins, net}
        self.lessons: List[Dict] = []
        self.shadows: Deque[Dict] = deque(maxlen=SHADOW_MAX)
        self.veto_stats: Dict[str, Dict[str, int]] = {}        # gate -> {blocked, would_win, would_lose, timeout}
        self.overrides: Dict[str, float] = {}
        self.maker: Dict[str, int] = {"attempts": 0, "filled": 0, "chased": 0}
        self.n_trades = 0
        self._last_applied: Dict[str, float] = {}
        self.load()

    # ---------------------------------------------------------------- kalıcılık
    def load(self) -> None:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.role_stats = d.get("role_stats", {})
        self.symbol_stats = d.get("symbol_stats", {})
        self.feature_stats = d.get("feature_stats", {})
        self.lessons = d.get("lessons", [])
        self.shadows = deque(d.get("shadows", []), maxlen=SHADOW_MAX)
        self.veto_stats = d.get("veto_stats", {})
        self.overrides = d.get("overrides", {})
        self.maker = d.get("maker", self.maker)
        self.n_trades = int(d.get("n_trades", 0))
        self._last_applied = d.get("last_applied", {})

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            d = {"role_stats": self.role_stats, "symbol_stats": self.symbol_stats,
                 "feature_stats": self.feature_stats, "lessons": self.lessons[-200:],
                 "shadows": list(self.shadows), "veto_stats": self.veto_stats,
                 "overrides": self.overrides, "maker": self.maker,
                 "n_trades": self.n_trades, "last_applied": self._last_applied,
                 "saved_ts": time.time()}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, ensure_ascii=False, default=str), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:
            pass

    # ---------------------------------------------------------------- sorgular
    def reliability(self, role: str) -> Optional[float]:
        s = self.role_stats.get(role)
        if not s or s.get("n", 0) < 3:
            return None
        return _beta_mean(s.get("agree", 0.0), s.get("disagree", 0.0))

    def learned(self) -> Dict:
        """Komiteye verilen özet: rol güvenilirliği + maker dolum oranı + geçersiz kılmalar."""
        rel = {r: self.reliability(r) for r in self.role_stats}
        att = self.maker.get("attempts", 0)
        p_fill = (self.maker.get("filled", 0) / att) if att >= 5 else None
        out = {"reliability": {k: v for k, v in rel.items() if v is not None},
               "overrides": dict(self.overrides)}
        if p_fill is not None:
            out["p_maker_fill"] = round(p_fill, 3)
        return out

    def p_win(self, prior: float = 0.5, prior_n: float = 10.0) -> float:
        """Komitenin kendi kazanma olasılığı — Beta(öncül) ile büzülmüş."""
        w = sum(s.get("wins", 0) for s in self.symbol_stats.values())
        n = sum(s.get("n", 0) for s in self.symbol_stats.values())
        return float((w + prior * prior_n) / (n + prior_n))

    def paused_reason(self, symbol: str, now: Optional[float] = None) -> Optional[str]:
        now = time.time() if now is None else now
        s = self.symbol_stats.get(symbol) or {}
        until = float(s.get("paused_until") or 0.0)
        if until > now:
            return f"{s.get('pause_reason', 'ders')} ({(until - now) / 3600:.1f} sa kaldı)"
        return None

    # ---------------------------------------------------------------- işlem sonrası
    def on_trade_closed(self, trade: Dict, verdict: Optional[Dict], now: Optional[float] = None,
                        current: Optional[Dict] = None) -> List[Dict]:
        """Kapanan işlem + onu açan komite kararı → istatistik + dersler.
        `current`: etkin parametreler (ders bir parametreyi değiştirirken tabanı bilsin)."""
        now = time.time() if now is None else now
        self.n_trades += 1
        won = bool(trade.get("win"))
        net = float(trade.get("net_pnl", 0.0))
        d = str(trade.get("direction", "LONG"))
        prof_sign = (1 if d == "LONG" else -1) * (1 if won else -1)

        # 1) rol güvenilirliği
        for v in (verdict or {}).get("votes") or []:
            if not v.get("data_ok", True) or v.get("role") in ("maliyet_yurutme", "risk_yonetimi", "denetci"):
                continue
            sc = float(v.get("score") or 0.0)
            if abs(sc) < 0.05:
                continue
            w = abs(sc) * float(v.get("confidence") or 0.0)
            st = self.role_stats.setdefault(v["role"], {"agree": 0.0, "disagree": 0.0, "n": 0})
            if (sc > 0) == (prof_sign > 0):
                st["agree"] += w
            else:
                st["disagree"] += w
            st["n"] += 1

        # 2) özellik istatistikleri
        sym = trade.get("symbol", "?")
        ss = self.symbol_stats.setdefault(sym, {"n": 0, "wins": 0, "net": 0.0, "streak_loss": 0})
        ss["n"] += 1
        ss["wins"] += int(won)
        ss["net"] = round(ss["net"] + net, 4)
        ss["streak_loss"] = 0 if won else ss["streak_loss"] + 1
        regime = next((n.split(" (")[0] for v in (verdict or {}).get("votes") or []
                       if v.get("role") == "rejim_oynaklik" for n in v.get("notes") or []
                       if any(k in n for k in ("TREND", "RANGE", "VOLATİL"))), "—")
        hour_block = f"{int(time.strftime('%H', time.gmtime(float(trade.get('opened_ts', now))))) // 4 * 4:02d}-UTC"
        self._feat("rejim", regime, won, net)
        self._feat("tutma", str(trade.get("hold_bucket", "?")), won, net)
        self._feat("cikis", str(trade.get("reason", "?")), won, net)
        self._feat("tetikleyici", str((verdict or {}).get("trigger") or trade.get("trigger") or "?"), won, net)
        self._feat("emir", str(trade.get("order_type", "?")), won, net)
        self._feat("saat", hour_block, won, net)
        self._feat("sablon", str((verdict or {}).get("template") or "?"), won, net)
        fees = float(trade.get("fees", 0.0))
        gross = float(trade.get("gross_pnl", 0.0))
        self._feat("komisyon", "toplam", won, net, extra={"fees": fees, "gross_abs": abs(gross)})

        # STOP → gölge: "stop çok mu dardı?" (hedef sonradan vurulur mu?)
        if str(trade.get("reason")) == "STOP" and trade.get("target"):
            self.shadows.append({
                "kind": "post_stop", "symbol": sym, "direction": d, "ts": now,
                "entry": float(trade.get("entry")), "target": float(trade["target"]),
                "stop": float(trade.get("exit", trade.get("entry"))) * (0.97 if d == "LONG" else 1.03),
                "expires": now + float(trade.get("horizon_sec", 3600)), "outcome": None, "gate": "STOP"})

        new = self._derive(now, current)
        self._journal_trade(trade, verdict, new, now)
        self.save()
        return new

    def _feat(self, dim: str, bucket: str, won: bool, net: float, extra: Optional[Dict] = None):
        b = self.feature_stats.setdefault(dim, {}).setdefault(bucket, {"n": 0, "wins": 0, "net": 0.0})
        b["n"] += 1
        b["wins"] += int(won)
        b["net"] = round(b["net"] + net, 4)
        for k, v in (extra or {}).items():
            b[k] = round(b.get(k, 0.0) + float(v), 4)

    # ---------------------------------------------------------------- gölgeler
    def on_candidate_vetoed(self, verdict: Dict, horizon_sec: float, now: Optional[float] = None):
        """Veto edilen aday: ufuk boyunca hedef/stop'a ilk hangisi değecek?"""
        now = time.time() if now is None else now
        plan = verdict.get("plan")
        if not plan or not verdict.get("direction"):
            return
        gates = [str(x).split(" ")[0].split(":")[0] for x in verdict.get("vetoes") or []] or ["?"]
        # aynı parite için açık gölge varsa tekrar ekleme (spam)
        if any(s["symbol"] == verdict["symbol"] and s["outcome"] is None and s["kind"] == "veto"
               for s in self.shadows):
            return
        self.shadows.append({
            "kind": "veto", "symbol": verdict["symbol"], "direction": verdict["direction"],
            "ts": now, "entry": float(plan["entry"]), "target": float(plan["target"]),
            "stop": float(plan["stop"]), "expires": now + float(horizon_sec),
            "outcome": None, "gates": gates, "gate": gates[0]})
        for g in gates:
            self.veto_stats.setdefault(g, {"blocked": 0, "would_win": 0, "would_lose": 0, "timeout": 0})["blocked"] += 1

    def on_maker_attempt(self, filled: bool, chased: bool = False):
        self.maker["attempts"] = self.maker.get("attempts", 0) + 1
        if filled:
            self.maker["filled"] = self.maker.get("filled", 0) + 1
        if chased:
            self.maker["chased"] = self.maker.get("chased", 0) + 1

    def update_shadows(self, bars: Dict[str, "object"], now: Optional[float] = None) -> List[Dict]:
        """bars: symbol -> DataFrame(high, low, close). Son barla ilk-geçiş kontrolü."""
        now = time.time() if now is None else now
        resolved = []
        for s in self.shadows:
            if s["outcome"] is not None:
                continue
            df = bars.get(s["symbol"])
            if df is not None and len(df):
                hi = float(df["high"].iloc[-1]); lo = float(df["low"].iloc[-1])
                sign = 1 if s["direction"] == "LONG" else -1
                hit_t = hi >= s["target"] if sign > 0 else lo <= s["target"]
                hit_s = lo <= s["stop"] if sign > 0 else hi >= s["stop"]
                if hit_t and hit_s:
                    s["outcome"] = "AMBIGUOUS"
                elif hit_t:
                    s["outcome"] = "TARGET"
                elif hit_s:
                    s["outcome"] = "STOP"
            if s["outcome"] is None and now >= s["expires"]:
                s["outcome"] = "TIMEOUT"
            if s["outcome"] is not None:
                s["resolved_ts"] = now
                resolved.append(s)
                if s["kind"] == "veto":
                    for g in s.get("gates") or [s.get("gate", "?")]:
                        vs = self.veto_stats.setdefault(g, {"blocked": 0, "would_win": 0, "would_lose": 0, "timeout": 0})
                        vs["would_win" if s["outcome"] == "TARGET" else
                           "would_lose" if s["outcome"] == "STOP" else "timeout"] += 1
        if resolved:
            self._derive(now)
            self.save()
        return resolved

    # ---------------------------------------------------------------- dersler
    def _rate(self, dim: str, bucket: str):
        b = (self.feature_stats.get(dim) or {}).get(bucket)
        if not b or b["n"] < MIN_N:
            return None
        return b["n"], b["wins"] / b["n"], b["net"]

    def _apply(self, param: str, delta: float, now: float, title: str, evidence: Dict) -> Optional[Dict]:
        lo, hi, _ = BOUNDS.get(param, (None, None, None))
        if lo is None:
            return None
        last = self._last_applied.get(param, 0.0)
        if now - last < COOLDOWN_SEC:
            return None
        cur = self.overrides.get(param)
        base = evidence.pop("_current", None)
        cur = float(cur if cur is not None else (base if base is not None else (lo + hi) / 2))
        new = float(min(hi, max(lo, cur + delta)))
        if abs(new - cur) < 1e-9:
            return None
        self.overrides[param] = round(new, 4)
        self._last_applied[param] = now
        les = {"ts": now, "title": title, "evidence": evidence,
               "action": {"param": param, "from": round(cur, 4), "to": round(new, 4)},
               "applied": True}
        self.lessons.append(les)
        return les

    def _note(self, now: float, title: str, evidence: Dict, key: str) -> Optional[Dict]:
        """Parametre değiştirmeyen ders — aynı anahtar 24 saatte bir yazılır."""
        if now - self._last_applied.get("note:" + key, 0.0) < COOLDOWN_SEC:
            return None
        self._last_applied["note:" + key] = now
        les = {"ts": now, "title": title, "evidence": evidence, "action": None, "applied": False}
        self.lessons.append(les)
        return les

    def derive(self, now: Optional[float] = None, current: Optional[Dict] = None) -> List[Dict]:
        """Döngü içinden çağrılır: gölge sonuçları vb. birikince ders çıkar."""
        return self._derive(time.time() if now is None else now, current)

    def _derive(self, now: float, current: Optional[Dict] = None) -> List[Dict]:
        current = current or {}
        new: List[Dict] = []

        def add(x):
            if x:
                new.append(x)

        # 1) erken çıkış zarar bölgesi (videonun ölçümü)
        r = self._rate("tutma", "0-15 dk")
        if r and r[1] < 0.35:
            add(self._apply("min_hold_sec", +300, now,
                            f"0-15 dk'lık işlemler zarar bölgesi: {r[0]} işlemde kazanma %{r[1]*100:.0f}, net {r[2]:+.2f} $ → asgari tutma +5 dk",
                            {"n": r[0], "win_rate": round(r[1], 3), "net": r[2], "_current": current.get("min_hold_sec", 900)}))
        # 2) komisyon payı
        k = (self.feature_stats.get("komisyon") or {}).get("toplam")
        if k and k["n"] >= MIN_N and k.get("gross_abs", 0) > 0:
            share = k["fees"] / k["gross_abs"]
            if share > 0.4:
                add(self._apply("min_gross_to_cost", +0.5, now,
                                f"Komisyon brüt hareketin %{share*100:.0f}'ini yiyor ({k['n']} işlem) → brüt/maliyet eşiği +0,5",
                                {"n": k["n"], "fee_share": round(share, 3), "fees": k["fees"], "_current": current.get("min_gross_to_cost", 2.0)}))
        # 3) trende karşı
        for reg in ("TREND AŞAĞI", "TREND YUKARI"):
            r = self._rate("rejim", reg)
            if r and r[1] < 0.40 and reg == "TREND AŞAĞI":
                add(self._apply("counter_trend_mult", -0.15, now,
                                f"Düşen trendde LONG: {r[0]} işlemde kazanma %{r[1]*100:.0f} → trende karşı boyut çarpanı düşürüldü",
                                {"n": r[0], "win_rate": round(r[1], 3), "net": r[2], "_current": current.get("counter_trend_mult", 0.5)}))
        # 4) zaman-stop kârı kesiyor → hedef yakınlaştır
        r = self._rate("cikis", "TIME_STOP")
        tp = (self.feature_stats.get("cikis") or {}).get("TP", {"n": 0})
        if r and r[2] > 0 and r[0] >= 2 * max(1, tp["n"]):
            add(self._apply("rr", -0.2, now,
                            f"Zaman-stop {r[0]} kez kârda kapandı, hedef {tp['n']} kez vuruldu → hedef yakınlaştırıldı (R −0,2)",
                            {"time_stop_n": r[0], "tp_n": tp["n"], "net": r[2], "_current": current.get("rr", 1.6)}))
        # 5) parite serisi → duraklat
        for sym, s in self.symbol_stats.items():
            if s.get("streak_loss", 0) >= 4 and float(s.get("paused_until") or 0) < now:
                s["paused_until"] = now + 24 * 3600
                s["pause_reason"] = f"{s['streak_loss']} ardışık zarar"
                s["streak_loss"] = 0
                add(self._note(now, f"{sym}: {s['pause_reason']} → 24 saat duraklatıldı",
                               {"symbol": sym, "n": s["n"], "net": s["net"]}, f"pause:{sym}"))
        # 6) rol isabeti
        for role, st in self.role_stats.items():
            rel = self.reliability(role)
            if rel is not None and st.get("n", 0) >= 15:
                if rel < 0.4:
                    add(self._note(now, f"{role} rolü isabetsiz: güvenilirlik {rel:.2f} ({st['n']} oy) → ağırlığı otomatik düştü",
                                   {"role": role, "reliability": round(rel, 3), "n": st["n"]}, f"role_low:{role}"))
                elif rel > 0.65:
                    add(self._note(now, f"{role} rolü isabetli: güvenilirlik {rel:.2f} ({st['n']} oy) → ağırlığı otomatik yükseldi",
                                   {"role": role, "reliability": round(rel, 3), "n": st["n"]}, f"role_high:{role}"))
        # 7) veto karşı-olgusal
        for gate, vs in self.veto_stats.items():
            done = vs["would_win"] + vs["would_lose"]
            if done < MIN_N:
                continue
            wr = vs["would_win"] / done
            if wr >= 0.65:
                if gate.startswith("OY"):
                    add(self._apply("theta", -0.05, now,
                                    f"Oy eşiği {done} adayı engelledi, %{wr*100:.0f}'i hedefe ulaşacaktı → eşik gevşetildi",
                                    {"gate": gate, **vs, "would_win_rate": round(wr, 3), "_current": current.get("theta", 0.25)}))
                elif gate.startswith("KOMİSYON"):
                    add(self._apply("min_gross_to_cost", -0.5, now,
                                    f"Komisyon kapısı {done} adayı engelledi, %{wr*100:.0f}'i hedefe ulaşacaktı → eşik gevşetildi",
                                    {"gate": gate, **vs, "would_win_rate": round(wr, 3), "_current": current.get("min_gross_to_cost", 2.0)}))
                else:
                    add(self._note(now, f"{gate} vetosu {done} adayı engelledi, %{wr*100:.0f}'i hedefe ulaşacaktı — gözardı edilen fırsat",
                                   {"gate": gate, **vs, "would_win_rate": round(wr, 3)}, f"veto_loose:{gate}"))
            elif wr <= 0.35:
                add(self._note(now, f"{gate} vetosu haklı: engellenen {done} adayın %{(1-wr)*100:.0f}'i stop olacaktı",
                               {"gate": gate, **vs, "would_win_rate": round(wr, 3)}, f"veto_right:{gate}"))
        # 8) stop çok dar? (post_stop gölgeleri)
        ps = [s for s in self.shadows if s["kind"] == "post_stop" and s["outcome"]]
        if len(ps) >= MIN_N:
            back = sum(1 for s in ps if s["outcome"] == "TARGET") / len(ps)
            if back >= 0.5:
                add(self._apply("stop_sigma_mult", +0.25, now,
                                f"Stop olan {len(ps)} işlemin %{back*100:.0f}'i sonradan hedefe ulaştı → stop genişletildi",
                                {"n": len(ps), "target_after_stop_rate": round(back, 3), "_current": current.get("stop_sigma_mult", 1.5)}))
        # 9) maker dolum
        att = self.maker.get("attempts", 0)
        if att >= 10:
            fr = self.maker.get("filled", 0) / att
            if fr < 0.3:
                add(self._apply("chase_taker_ratio", -0.5, now,
                                f"Maker emirlerin yalnız %{fr*100:.0f}'i doldu ({att} deneme) → taker'a geçiş eşiği düşürüldü",
                                {"attempts": att, "fill_rate": round(fr, 3), "_current": current.get("chase_taker_ratio", 3.0)}))
        # 10) saat dilimi
        for blk, b in (self.feature_stats.get("saat") or {}).items():
            if b["n"] >= 10 and b["wins"] / b["n"] < 0.30:
                add(self._note(now, f"{blk} saat diliminde {b['n']} işlemde kazanma %{b['wins']/b['n']*100:.0f} — bu dilimde dikkat",
                               {"block": blk, "n": b["n"], "net": b["net"]}, f"hour:{blk}"))
        for les in new:
            self._journal_lesson(les)
        return new

    # ---------------------------------------------------------------- günlük
    def _journal_trade(self, trade: Dict, verdict: Optional[Dict], lessons: List[Dict], now: float):
        rec = {"ts": now, "type": "trade", "trade": trade,
               "roles": [{"role": v["role"], "score": v.get("score"), "veto": v.get("veto")}
                         for v in (verdict or {}).get("votes") or []],
               "lessons": [l["title"] for l in lessons]}
        self._append_jsonl(rec)
        if self.md:
            emoji = "🟢" if trade.get("win") else "🔴"
            lines = [f"\n### {emoji} İşlem #{self.n_trades} · {trade.get('symbol')} {trade.get('direction')} · "
                     f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(now))}",
                     f"- Sonuç: **net {float(trade.get('net_pnl', 0)):+.2f} $** (brüt {float(trade.get('gross_pnl', 0)):+.2f}, "
                     f"komisyon {float(trade.get('fees', 0)):.2f}) · sebep {trade.get('reason')} · "
                     f"tutma {trade.get('hold_bucket')} · emir {trade.get('order_type', '?')}"]
            prof_sign = (1 if trade.get("direction") == "LONG" else -1) * (1 if trade.get("win") else -1)
            for v in (verdict or {}).get("votes") or []:
                if v.get("role") in ("maliyet_yurutme", "risk_yonetimi", "denetci") or not v.get("data_ok", True):
                    continue
                sc = float(v.get("score") or 0)
                mark = "✓" if (abs(sc) >= 0.05 and (sc > 0) == (prof_sign > 0)) else ("✗" if abs(sc) >= 0.05 else "·")
                lines.append(f"  - {mark} {v.get('title', v['role'])}: oy {sc:+.2f} — {(v.get('notes') or ['—'])[0]}")
            if trade.get("win"):
                lines.append("- **Nasıl kazanıldı:** " + self._how(trade, verdict, True))
            else:
                lines.append("- **Nasıl kaybedilmezdi:** " + self._how(trade, verdict, False))
            for l in lessons:
                lines.append(f"- 📘 DERS: {l['title']}" + (f" → `{l['action']['param']}` {l['action']['from']} → {l['action']['to']}" if l.get("action") else ""))
            self._append_md("\n".join(lines))

    def _how(self, trade: Dict, verdict: Optional[Dict], won: bool) -> str:
        reason = str(trade.get("reason"))
        bucket = str(trade.get("hold_bucket"))
        fees = float(trade.get("fees", 0)); gross = float(trade.get("gross_pnl", 0))
        parts = []
        if won:
            if reason == "TP":
                parts.append("hedef yapısal seviyede tutuldu, plan sonuna kadar taşındı")
            elif reason == "GIVEBACK":
                parts.append(f"tepe kârın %{trade.get('peak_pnl_pct', 0):.2f} olduğu yerden geri-verme kuralı kârı kilitledi")
            elif reason == "TIME_STOP":
                parts.append("zaman-stop kârda çıkardı; hedef uzak olabilir")
            if fees > 0 and gross > 0 and fees / gross > 0.3:
                parts.append(f"ama komisyon brütün %{fees/gross*100:.0f}'i — maker dolumu bunu düşürür")
        else:
            if reason == "STOP":
                parts.append("stop vuruldu — post-stop gölge takibi 'stop çok mu dardı' sorusunu ölçecek")
            elif reason == "TIME_STOP":
                parts.append("hareket gelmedi; tetikleyici erken ya da rejim uyumsuzdu")
            elif reason == "HALT":
                parts.append("kill-switch kapattı — günlük limit koruması çalıştı")
            if bucket == "0-15 dk":
                parts.append("0-15 dk kovası (videoda zarar bölgesi)")
            wrong = [v.get("role") for v in (verdict or {}).get("votes") or []
                     if float(v.get("score") or 0) * (1 if trade.get("direction") == "LONG" else -1) > 0.3]
            if wrong:
                parts.append("yanlış yönde ısrarcı roller: " + ", ".join(wrong[:3]))
        return "; ".join(parts) or "—"

    def _journal_lesson(self, les: Dict):
        self._append_jsonl({"ts": les["ts"], "type": "lesson", **les})

    def _append_jsonl(self, rec: Dict):
        if not self.jsonl:
            return
        try:
            self.jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(self.jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def _append_md(self, text: str):
        if not self.md:
            return
        try:
            self.md.parent.mkdir(parents=True, exist_ok=True)
            new = not self.md.exists()
            with open(self.md, "a", encoding="utf-8") as f:
                if new:
                    f.write("# CryptoMind 1.000 $ Simülatör — İşlem Günlüğü ve Dersler\n\n"
                            "Her işlem kapanınca yazılır: roller haklı mıydı, nasıl kazanıldı / nasıl "
                            "kaybedilmezdi, hangi ders çıktı ve hangi parametre değişti.\n")
                f.write(text + "\n")
        except Exception:
            pass

    # ---------------------------------------------------------------- panel
    def status(self) -> Dict:
        roles = []
        for r, st in self.role_stats.items():
            roles.append({"role": r, "n": st.get("n", 0), "reliability": self.reliability(r),
                          "agree": round(st.get("agree", 0.0), 2), "disagree": round(st.get("disagree", 0.0), 2)})
        open_sh = [s for s in self.shadows if s["outcome"] is None]
        done_sh = [s for s in self.shadows if s["outcome"] is not None][-30:]
        return {"n_trades": self.n_trades, "p_win": round(self.p_win(), 3),
                "roles": roles, "lessons": self.lessons[-30:][::-1],
                "overrides": self.overrides, "veto_stats": self.veto_stats,
                "feature_stats": self.feature_stats, "symbol_stats": self.symbol_stats,
                "maker": self.maker,
                "shadows_open": len(open_sh), "shadows_recent": done_sh[::-1]}
