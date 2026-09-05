"""
VETO İNCELEMESİ — veto alan aday tek bir kapının kararına bırakılmaz; bütünsel kanıt tablosu çıkarılır:

  • FORMASYON VARLIKLARI   ağır bağlam: grafik/harmonik/mum konsensüsü; hafif katman: yapısal bayraklar
                           (likidite süpürme, başarısız kırılma, yapı kırılımı+retest, sıkışma, Donchian)
  • İNDİKATÖRLER           ağır bağlam: 300 göstergeli aile (al/sat/nötr sayısı); hafif katman: 8 göstergelik
                           yerel konsensüs (EMA20/50, RSI, VWAP, Bollinger-z, ADX, EMA9×21, OBI, hacim)
  • STRATEJİ UYUMU         tetikleyen sleeve bu rejimde izinli mi, şablon tutarlı mı, trend skoru
  • HABER                  skor, kaç haber, boğa/ayı sayısı, ağır risk, doğrulanmış hareket

KARAR KURALI (dürüst): yalnız YUMUŞAK vetolar (OY, GÜVEN — oy toplamının veri yokluğundan sıfıra çakılması)
bütünsel kanıtla aşılabilir ve boyut ×0,6 olur. SERT vetolar (komisyon/EV, denetçi, risk, kill-switch,
sağlık, haber riski, max chase, tier-3) ASLA aşılmaz. Aşma şartları: indikatör al ≥ 2×sat ve ≥ 4 al,
formasyon karşı yönde değil, haber ≥ −0,2 ve ağır risk yok, strateji uyumu var.
"""
from __future__ import annotations

from typing import Dict, List, Optional

SOFT_GATES = {"OY", "GÜVEN"}


def _norm_gate(v: str) -> str:
    from ..learn.missed import normalize_gate
    return normalize_gate(v)


def light_indicator_consensus(f: Dict, direction: str = "LONG") -> Dict:
    """Hafif katman için 8 göstergelik al/sat/nötr sayımı (ağır aile yoksa)."""
    rows: List[Dict] = []

    def add(name, al, sat):
        rows.append({"name": name, "vote": "AL" if al else "SAT" if sat else "NÖTR"})

    tu = f.get("trend_up")
    add("EMA20>EMA50", tu is True, tu is False)
    rsi = f.get("rsi")
    add("RSI(14)", rsi is not None and rsi < 40, rsi is not None and rsi > 60)
    dv = f.get("dist_vwap_pct")
    add("VWAP", dv is not None and dv > 0, dv is not None and dv < 0)
    z = f.get("z")
    add("Bollinger-z", z is not None and z < -1.0, z is not None and z > 1.0)
    adx = f.get("adx")
    add("ADX(14)", adx is not None and adx >= 20 and tu is True, adx is not None and adx >= 20 and tu is False)
    add("EMA9×21", bool(f.get("ema_cross_up")), False)
    obi = f.get("obi")
    add("Defter dengesi", obi is not None and obi >= 0.6, obi is not None and obi <= 0.4)
    vr = f.get("vol_ratio")
    add("Hacim", vr is not None and vr >= 1.3 and bool(f.get("bar_up")), vr is not None and vr >= 1.3 and not f.get("bar_up"))
    cvd = f.get("cvd_ratio")
    if cvd is not None:
        add("CVD (taker akışı)", cvd >= 0.2, cvd <= -0.2)
    al = sum(1 for r in rows if r["vote"] == "AL"); sat = sum(1 for r in rows if r["vote"] == "SAT")
    return {"source": f"hafif ({len(rows)} gösterge)", "al": al, "sat": sat, "notr": len(rows) - al - sat, "total": len(rows),
            "rows": rows, "bias": "YUKARI" if al > sat else "AŞAĞI" if sat > al else "NÖTR"}


def formations_summary(slow: Optional[Dict], f: Dict) -> Dict:
    pats = (slow or {}).get("patterns") or {}
    cons = pats.get("consensus") or {}
    harm = ((slow or {}).get("harmonics") or {}).get("patterns") or []
    cand = ((slow or {}).get("candles") or {}).get("summary") or {}
    structural = [name for name, key in (("likidite süpürme", "swept_low"), ("başarısız kırılma", "failed_breakdown"),
                                          ("yapı kırılımı+retest", "bos_retest_up"), ("Donchian kırılımı", "donchian_break"),
                                          ("sıkışma sonrası", "bb_prev_pctile")) if
                  (f.get(key) is True) or (key == "bb_prev_pctile" and f.get(key) is not None and f[key] <= 20)]
    out = {"heavy": bool(cons), "bias": cons.get("bias"), "score": cons.get("score"), "n": cons.get("n"),
           "long": cons.get("long"), "short": cons.get("short"), "harmonics": len(harm), "candles": cand.get("n"),
           "structural": structural}
    out["present"] = bool(cons.get("n")) or bool(structural) or bool(harm)
    out["against"] = (cons.get("bias") == "AŞAĞI") if cons else False
    return out


def indicators_summary(slow: Optional[Dict], f: Dict) -> Dict:
    ind = (slow or {}).get("indicators") or {}
    fam = ind.get("family") or {}
    if ind.get("available") and fam:
        al, sat, notr = int(fam.get("al") or 0), int(fam.get("sat") or 0), int(fam.get("notr") or 0)
        return {"source": "ağır (300 gösterge/144 aile)", "al": al, "sat": sat, "notr": notr, "total": al + sat + notr,
                "bias": ind.get("bias"), "net": ind.get("net"), "rows": []}
    return light_indicator_consensus(f)


def news_summary(news: Optional[Dict]) -> Dict:
    n = news or {}
    sc = n.get("score")
    return {"available": bool(n.get("data_ok")), "score": sc, "n": n.get("n_items"), "bull": n.get("bull"), "bear": n.get("bear"),
            "severe_risk": bool(n.get("severe_risk")), "confirmed": bool(n.get("confirmed")), "event": n.get("top_event"),
            "label": ("ağır risk" if n.get("severe_risk") else "negatif" if (sc is not None and sc <= -0.2) else
                      "pozitif" if (sc is not None and sc >= 0.2) else "nötr/yok")}


def strategy_fit(trigger: Optional[str], allowed: List[str], template: str, regime: Optional[str], f: Dict) -> Dict:
    ok = bool(trigger) and trigger in (allowed or [])
    return {"trigger": trigger, "allowed": ok, "regime": regime, "template": template,
            "trend_score": f.get("trend_score"), "fit": ok}


def review(direction: str, vetoes: List[str], slow: Optional[Dict], f: Dict, news: Optional[Dict],
           trigger: Optional[str], allowed: List[str], template: str, regime: Optional[str],
           vote_score: float = 0.0) -> Dict:
    """vote_score: yön-hizalı ağırlıklı rol oyu. Roller NET KARŞI oy verdiyse (< 0) yumuşak veto aşılmaz —
    inceleme yalnız veri yokluğundan sıfıra çakılan oyu kurtarır, karşı oyu bastırmaz."""
    gates = [_norm_gate(v) for v in vetoes]
    hard = [g for g in gates if g not in SOFT_GATES]
    fo = formations_summary(slow, f)
    ind = indicators_summary(slow, f)
    nw = news_summary(news)
    fit = strategy_fit(trigger, allowed, template, regime, f)
    s = 1 if direction == "LONG" else -1
    al, sat = (ind["al"], ind["sat"]) if s > 0 else (ind["sat"], ind["al"])
    checks = {
        "indikator": al >= 4 and al >= 2 * sat,
        "formasyon": not (fo["against"] if s > 0 else (fo.get("bias") == "YUKARI")),
        "haber": (not nw["severe_risk"]) and (nw["score"] is None or (s * float(nw["score"])) >= -0.2),
        "strateji": fit["fit"],
        "oy_karsi_degil": float(vote_score) >= 0.0,
    }
    overridable = bool(vetoes) and not hard
    decision = "AÇ" if overridable and all(checks.values()) else "VETO"
    why = []
    if hard:
        why.append("sert veto aşılmaz: " + ", ".join(sorted(set(hard))))
    for k, ok in checks.items():
        if not ok:
            why.append({"indikator": f"indikatör yetersiz ({al} al / {sat} sat)", "formasyon": "formasyon karşı yönde",
                        "haber": f"haber {nw['label']}", "strateji": "strateji uyumu yok",
                        "oy_karsi_degil": f"roller net karşı oy verdi ({float(vote_score):+.2f})"}[k])
    summary = (f"formasyon: {'var' if fo['present'] else 'yok'}" + (f" ({fo['bias']}, n={fo['n']})" if fo["heavy"] else
               (" [" + ", ".join(fo["structural"]) + "]" if fo["structural"] else "")) +
               f" · indikatör: {ind['al']} al / {ind['sat']} sat / {ind['notr']} nötr ({ind['source']})" +
               f" · haber: {nw['label']}" + (f" ({nw['score']:+.2f}, {nw['n']} haber)" if nw["score"] is not None else "") +
               f" · strateji: {'uyumlu' if fit['fit'] else 'uyumsuz'} ({trigger or '—'} @ {regime or 'rejim yok'})" +
               f" → {decision}" + (" ×0,6" if decision == "AÇ" else (" — " + "; ".join(why) if why else "")))
    return {"decision": decision, "size_mult": 0.6 if decision == "AÇ" else 0.0, "gates": gates, "soft_only": overridable,
            "checks": checks, "formations": fo, "indicators": ind, "news": nw, "strategy": fit, "why": why, "summary_tr": summary}
