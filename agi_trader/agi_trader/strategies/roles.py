"""
KOMİTE — CryptoMind'ın bütün verilerini işleyen 12 uzman rol.

Her rol tek bir şeyden sorumludur ve ÜÇ şey döndürür: yön puanı (−1..+1),
kendi güveni (0..1) ve gerekirse VETO. Roller birbirinin verisine el sürmez;
komite (committee.py) oyları ölçülmüş güvenilirlikle ağırlıklandırır.

Tasarım ilkeleri (bu projede ölçülmüş derslerden):
  • "Veri yok" ile "etki yok" ayrı: verisi olmayan rol OY VERMEZ, 'VERİ YOK' der.
  • Yön öngörüsü ÖLÇÜLÜP çürütülmüş aileler (harmonik, mum, çift tepe) yön oyuna
    girmez; yalnız seviye/bağlam sağlar. Gösterge tablosu (ölçülmüş takip getirisi
    negatif) düşük taban ağırlıkla başlar — ders motoru isabete göre günceller.
  • Maliyet ve risk rolleri VETO yetkilidir; hiçbir puan onları ezemez.
  • Denetçi rolü değişmezleri (stop doğru tarafta, hedef maliyeti aşar, boyut
    tavan altında, veri taze) kontrol eder; ihlal = giriş yok.

Bu modül ağ erişimi yapmaz, RNG kullanmaz; saf fonksiyonlardır.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# Taban ağırlıklar — CryptoMind'ın ÖLÇÜLMÜŞ kanıtına göre başlangıç.
# (gösterge tablosu 4h'de r=−0,16 → düşük; formasyon yönü ölçülmedi → orta;
#  mover yalnız HAREKET olasılığı → güvene katkı; sosyal ölçülmüş hesap yoksa 0)
ROLE_BASE_WEIGHT: Dict[str, float] = {
    "piyasa_yapisi": 1.00,
    "formasyon": 0.60,
    "gosterge_konsensusu": 0.30,
    "rejim_oynaklik": 0.80,
    "nitelendirme_istatistik": 0.90,
    "hareket_avcisi": 0.40,
    "makro_olay": 0.50,
    "haber_sosyal": 0.45,
    "sleeve_sinyali": 0.70,       # tetikleyen stratejinin kendi kanaati (hafif katmanda roller susunca oy sıfıra çakılmasın)
    "orkestrator_konsensusu": 0.90,
}
VETO_ROLES = ("maliyet_yurutme", "risk_yonetimi", "denetci")
ROLE_TITLES: Dict[str, str] = {
    "piyasa_yapisi": "Piyasa Yapısı & Maks/Min Analisti",
    "formasyon": "Formasyon Uzmanı (grafik/harmonik/mum)",
    "gosterge_konsensusu": "Gösterge Konsensüsü (300 gösterge/144 aile)",
    "rejim_oynaklik": "Rejim & Oynaklık Uzmanı (HMM)",
    "nitelendirme_istatistik": "Nitelendirme & İstatistik (parite×ufuk)",
    "hareket_avcisi": "Hareket Avcısı (%1 oynama olasılığı)",
    "sleeve_sinyali": "Strateji Sinyali (tetikleyen sleeve'in kanaati)",
    "makro_olay": "Makro & Olay Takvimi",
    "haber_sosyal": "Haber & Sosyal Tarayıcı (RSS/Reddit/StockTwits/Binance)",
    "orkestrator_konsensusu": "Orkestratör Konsensüsü (11 katman, 4h)",
    "maliyet_yurutme": "Maliyet & Yürütme (komisyon/spread/derinlik)",
    "risk_yonetimi": "Risk Yöneticisi (sermaye/korelasyon/limit)",
    "denetci": "Denetçi (değişmezler)",
}


@dataclass
class RoleVote:
    role: str
    score: float = 0.0           # −1 (kesin SHORT) … +1 (kesin LONG); 0 = yön yok
    confidence: float = 0.0      # 0..1 — rolün kendi kararına güveni
    veto: Optional[str] = None
    size_mult: float = 1.0
    data_ok: bool = True
    notes: List[str] = field(default_factory=list)
    levels: Dict = field(default_factory=dict)   # yalnız piyasa yapısı doldurur

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["title"] = ROLE_TITLES.get(self.role, self.role)
        d["score"] = round(self.score, 3)
        d["confidence"] = round(self.confidence, 3)
        d["size_mult"] = round(self.size_mult, 3)
        return d


def _num(x, default=None):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


# ═══════════════════════════════════════════════════════════════════════════
# 1) PİYASA YAPISI — seviyeler, trend çizgileri, maks/min, swing'ler
# ═══════════════════════════════════════════════════════════════════════════
def role_market_structure(price: float, atr_pct: float, chart: Optional[Dict],
                          signal: Optional[Dict]) -> RoleVote:
    v = RoleVote("piyasa_yapisi")
    if not chart or price <= 0:
        v.data_ok = False
        v.notes.append("grafik bağlamı yok")
        return v
    lv = chart.get("levels") or {}
    tl = chart.get("trendlines") or {}
    ex = chart.get("extremes") or {}
    fc = (signal or {}).get("forecast") or {}
    band = max(0.15, atr_pct * 0.5)          # "yakın" = yarım ATR ya da %0,15

    supports: List[float] = []
    resists: List[float] = []
    for h in (tl.get("horizontals") or []):
        p = _num(h.get("price"))
        if p is None:
            continue
        (supports if p < price else resists).append(p)
    for k in ("buy_threshold",):
        p = _num(fc.get(k))
        if p and p < price:
            supports.append(p)
    for k in ("sell_threshold",):
        p = _num(fc.get(k))
        if p and p > price:
            resists.append(p)
    lo20, hi20 = _num(ex.get("recent_low_20")), _num(ex.get("recent_high_20"))
    if lo20 and lo20 < price:
        supports.append(lo20)
    if hi20 and hi20 > price:
        resists.append(hi20)
    el, eh = _num(lv.get("expected_low")), _num(lv.get("expected_high"))
    if el and el < price:
        supports.append(el)
    if eh and eh > price:
        resists.append(eh)
    for s in (chart.get("smc") or {}).get("swings") or []:
        p = _num(s.get("y"))
        if p is None:
            continue
        if s.get("kind", "").lower().startswith(("l", "d")) and p < price:   # low/dip
            supports.append(p)
        elif p > price:
            resists.append(p)

    sup = max(supports) if supports else None
    res = min(resists) if resists else None
    d_sup = (price - sup) / price * 100.0 if sup else None
    d_res = (res - price) / price * 100.0 if res else None
    rp = _num(ex.get("range_position"))
    v.levels = {"support": sup, "resistance": res, "dist_support_pct": d_sup,
                "dist_resistance_pct": d_res, "range_position": rp,
                "n_supports": len(supports), "n_resists": len(resists)}

    score = 0.0
    conf = 0.35
    if d_sup is not None and d_sup <= band:
        score += 0.6
        conf += 0.25
        v.notes.append(f"desteğe yakın (%{d_sup:.2f} üstünde)")
    if d_res is not None and d_res <= band:
        score -= 0.6
        conf += 0.25
        v.notes.append(f"dirence yakın (%{d_res:.2f} altında)")
    if d_sup is not None and d_res is not None:
        room_up, room_dn = d_res, d_sup
        if room_up > 0 and room_dn > 0:
            asym = (room_up - room_dn) / (room_up + room_dn)   # +: yukarı boşluk fazla
            score += 0.4 * asym
            v.notes.append(f"yukarı boşluk %{room_up:.2f} · aşağı %{room_dn:.2f}")
    if rp is not None:
        if rp <= 0.15:
            score += 0.2
            v.notes.append("aralığın dibinde (range_position ≤ 0,15)")
        elif rp >= 0.85:
            score -= 0.2
            v.notes.append("aralığın tepesinde (range_position ≥ 0,85)")
    if tl.get("channel"):
        conf += 0.1
        v.notes.append("kanal tespit edildi")
    v.score = _clip(score)
    v.confidence = _clip(conf, 0.0, 1.0)
    return v


# ═══════════════════════════════════════════════════════════════════════════
# 2) FORMASYON — grafik formasyonları oy verir; harmonik/mum yalnız bağlam
# ═══════════════════════════════════════════════════════════════════════════
def role_formations(patterns: Optional[Dict], harmonics: Optional[Dict],
                    candles: Optional[Dict]) -> RoleVote:
    v = RoleVote("formasyon")
    if not patterns:
        v.data_ok = False
        v.notes.append("formasyon verisi yok")
        return v
    cons = patterns.get("consensus") or {}
    rec = patterns.get("recommendation") or {}
    score = _num(cons.get("score"), 0.0) or 0.0
    n = int(cons.get("n") or 0)
    conf = min(1.0, 0.2 + 0.15 * n) if n else 0.0
    if rec.get("available"):
        d = rec.get("direction")
        rc = _num(rec.get("confidence"), 0.0) or 0.0
        if d == "LONG":
            score = 0.5 * score + 0.5 * rc
        elif d == "SHORT":
            score = 0.5 * score - 0.5 * rc
        conf = max(conf, rc)
        v.notes.append(f"öneri {d} · hedef %{_num(rec.get('target_pct'), 0):.2f} · "
                       f"R/R {_num(rec.get('rr'), 0):.2f}")
        if rec.get("verdict"):
            v.notes.append(str(rec["verdict"])[:90])
    else:
        v.notes.append(rec.get("reason", "işlenebilir formasyon yok")[:90])
    if rec.get("excluded_no_edge"):
        v.notes.append("kanıtsız aileler dışlandı: " + ", ".join(rec["excluded_no_edge"][:3]))
    # harmonik: şekil gerçek, yön ölçülmedi → yalnız not
    hn = len((harmonics or {}).get("patterns") or [])
    if hn:
        v.notes.append(f"{hn} harmonik (yön oyu YOK — ölçülmedi)")
    cs = (candles or {}).get("summary") or {}
    if cs.get("n"):
        v.notes.append(f"mum: {cs.get('bias', '—')} (yön oyu YOK — ölçüldü, kanıt yok)")
    v.score = _clip(score)
    v.confidence = _clip(conf, 0.0, 1.0)
    return v


# ═══════════════════════════════════════════════════════════════════════════
# 3) GÖSTERGE KONSENSÜSÜ — aile oyları (ölçülmüş: takip getirisi negatif → düşük ağırlık)
# ═══════════════════════════════════════════════════════════════════════════
def role_indicator_consensus(board: Optional[Dict]) -> RoleVote:
    v = RoleVote("gosterge_konsensusu")
    if not board or not board.get("available"):
        v.data_ok = False
        v.notes.append("gösterge tablosu yok")
        return v
    fam = board.get("family") or {}
    net = _num(board.get("net"), 0.0) or 0.0
    tot = int(fam.get("total") or 0)
    v.score = _clip(net)
    v.confidence = _clip(0.3 + 0.5 * abs(net), 0.0, 0.8) if tot else 0.0
    v.notes.append(f"aile oyu AL {fam.get('al', 0)} · SAT {fam.get('sat', 0)} · "
                   f"NÖTR {fam.get('notr', 0)} → {board.get('bias', '—')}")
    ev = board.get("evidence") or {}
    if isinstance(ev, dict) and ev.get("verdict"):
        v.notes.append("ölçüm: " + str(ev["verdict"])[:80])
    return v


# ═══════════════════════════════════════════════════════════════════════════
# 4) REJİM & OYNAKLIK — strateji şablonu + boyut çarpanı
# ═══════════════════════════════════════════════════════════════════════════
def role_regime(regime: Optional[Dict], volatility: str, direction_hint: str,
                counter_trend_mult: float = 0.5) -> RoleVote:
    v = RoleVote("rejim_oynaklik")
    if not regime or not regime.get("label"):
        v.data_ok = False
        v.notes.append("rejim yok")
        return v
    label = str(regime.get("label"))
    m = _num(regime.get("multiplier"), 0.6) or 0.6
    conf = _num(regime.get("confidence"), 0.5) or 0.5
    score = 0.0
    if label == "TREND YUKARI":
        score = 0.5
    elif label == "TREND AŞAĞI":
        score = -0.5
    v.notes.append(f"{label} ({regime.get('method', '?')}, güven {conf:.2f})")
    if direction_hint == "LONG" and label == "TREND AŞAĞI":
        m *= counter_trend_mult
        v.notes.append(f"trende karşı LONG → ×{counter_trend_mult}")
    elif direction_hint == "SHORT" and label == "TREND YUKARI":
        m *= counter_trend_mult
        v.notes.append(f"trende karşı SHORT → ×{counter_trend_mult}")
    if volatility == "extreme":
        m *= 0.6
        v.notes.append("aşırı oynaklık ×0,6")
    elif volatility == "high":
        m *= 0.8
    v.template = "pullback" if label.startswith("TREND") else "mean_reversion"  # type: ignore[attr-defined]
    v.score = _clip(score)
    v.confidence = _clip(conf, 0.0, 1.0)
    v.size_mult = float(max(0.0, min(1.1, m)))
    v.notes.append(f"şablon: {'geri çekilme' if label.startswith('TREND') else 'ortalamaya dönüş'}")
    return v


# ═══════════════════════════════════════════════════════════════════════════
# 5) NİTELENDİRME & İSTATİSTİK — parite×ufuk hücresi, model olasılığı
# ═══════════════════════════════════════════════════════════════════════════
QUAL_MULT = {"HIGH_CONFIDENCE": 1.0, "QUALIFIED": 1.0, "UNVERIFIED": 0.6,
             "RESEARCH_ONLY": 0.6, "NO_DATA": 0.6, "NO_EDGE": 0.5, "DEGRADED": 0.0}


def role_qualification(cell: Optional[Dict], direction_hint: str,
                       strict: bool = False) -> RoleVote:
    v = RoleVote("nitelendirme_istatistik")
    if not cell:
        v.data_ok = False
        v.notes.append("bu parite/ufuk için hücre yok")
        v.size_mult = 0.6
        return v
    st = str(cell.get("status") or "NO_DATA").upper()
    p = _num(cell.get("p_model_live"))
    base = _num(cell.get("base_rate")) or _num(cell.get("p_base"))
    m = QUAL_MULT.get(st, 0.6)
    if st == "DEGRADED":
        v.veto = "NİTELENDİRME hücresi BOZULDU"
    elif strict and st not in ("QUALIFIED", "HIGH_CONFIDENCE"):
        v.veto = f"NİTELENDİRME {st} (katı mod)"
    score = 0.0
    conf = 0.3
    if p is not None:
        ref = base if base else 0.5
        lift = (p - ref) / max(1e-6, ref)
        score = _clip(lift) * (1.0 if direction_hint == "LONG" else -1.0)
        conf = min(1.0, 0.4 + abs(lift))
        v.notes.append(f"p_model %{p*100:.1f} (taban %{ref*100:.1f}, lift {lift:+.2f})")
    v.notes.append(f"hücre {st} → ×{m}")
    conv = cell.get("convergence") or cell.get("convergence_verdict")
    if isinstance(conv, dict):
        conv = conv.get("verdict")
    if conv:
        v.notes.append(f"yakınsama {conv}")
        if str(conv).upper() == "UNSTABLE":
            conf *= 0.7
    v.score = score
    v.confidence = _clip(conf, 0.0, 1.0)
    v.size_mult = m
    return v


# ═══════════════════════════════════════════════════════════════════════════
# 6) HAREKET AVCISI — %1 oynama olasılığı (yön değil, hareket)
# ═══════════════════════════════════════════════════════════════════════════
def role_mover(pick: Optional[Dict]) -> RoleVote:
    v = RoleVote("hareket_avcisi")
    if not pick:
        v.data_ok = False
        v.notes.append("mover verisi yok")
        return v
    p = _num(pick.get("probability") or pick.get("prob") or pick.get("p"))
    base = _num(pick.get("base_rate") or pick.get("base"))
    lift = _num(pick.get("lift"))
    if p is None:
        v.data_ok = False
        return v
    v.score = 0.0                          # yön oyu YOK — yalnız güven
    v.confidence = _clip(p, 0.0, 1.0)
    v.notes.append(f"bugün ≥%1 oynama olasılığı %{p*100:.0f}"
                   + (f" (taban %{base*100:.0f})" if base else "")
                   + (f" · lift {lift:.2f}×" if lift else ""))
    if pick.get("model_failed") or str(pick.get("status", "")).upper().startswith("MODEL GE"):
        v.confidence *= 0.5
        v.notes.append("mover modeli doğrulamayı GEÇEMEDİ → yarı ağırlık")
    return v


# ═══════════════════════════════════════════════════════════════════════════
# 7) MALİYET & YÜRÜTME — VETO yetkili
# ═══════════════════════════════════════════════════════════════════════════
def role_cost_execution(target_gross_pct: float, cost_pct_taker: float, cost_pct_maker: float,
                        spread_bps: float, bid_depth: float, ask_depth: float,
                        notional: float, min_gross_to_cost: float,
                        p_maker_fill: float = 0.5, max_spread_bps: float = 15.0) -> RoleVote:
    v = RoleVote("maliyet_yurutme")
    # beklenen maliyet: maker dolarsa maker, dolmazsa taker
    exp_cost = p_maker_fill * cost_pct_maker + (1.0 - p_maker_fill) * cost_pct_taker
    ratio = target_gross_pct / exp_cost if exp_cost > 0 else float("inf")
    v.notes.append(f"brüt hedef %{target_gross_pct:.3f} · beklenen maliyet %{exp_cost:.3f} "
                   f"(maker %{cost_pct_maker:.3f} / taker %{cost_pct_taker:.3f}) · oran {ratio:.2f}")
    if ratio < min_gross_to_cost:
        v.veto = f"KOMİSYON brüt/maliyet {ratio:.2f} < {min_gross_to_cost}"
    if spread_bps > max_spread_bps:
        v.veto = (v.veto or "") + f" | SPREAD {spread_bps:.1f} bps > {max_spread_bps}"
    depth = min(bid_depth, ask_depth) if (bid_depth and ask_depth) else 0.0
    if depth and notional > depth * 0.1:
        v.veto = (v.veto or "") + f" | DERİNLİK emir {notional:.0f}$ > bandın %10'u ({depth:.0f}$)"
    elif not depth:
        v.notes.append("derinlik ölçülmedi (varsayım)")
    v.confidence = 1.0
    v.size_mult = 1.0 if ratio >= 2 * min_gross_to_cost else 0.75
    v.order_type = "maker" if spread_bps <= max_spread_bps else "taker"  # type: ignore[attr-defined]
    v.notes.append(f"emir tipi: {'MAKER (limit, en iyi teklif)' if spread_bps <= max_spread_bps else 'TAKER'}")
    if v.veto:
        v.veto = v.veto.strip(" |")
    return v


# ═══════════════════════════════════════════════════════════════════════════
# 8) RİSK YÖNETİCİSİ — VETO yetkili
# ═══════════════════════════════════════════════════════════════════════════
def role_risk(symbol: str, direction: str, open_positions: Dict[str, Dict],
              corr: Optional[Dict], max_open: int, exposure_room: float,
              notional: float, paused_reason: Optional[str], p_win: float,
              rr: float, halted: bool, corr_limit: float = 0.7) -> RoleVote:
    v = RoleVote("risk_yonetimi")
    v.confidence = 1.0
    if halted:
        v.veto = "KILL-SWITCH aktif"
        return v
    if paused_reason:
        v.veto = f"PARİTE DURAKLATILDI: {paused_reason}"
        return v
    if len(open_positions) >= max_open:
        v.veto = f"AÇIK POZİSYON tavanı ({max_open})"
        return v
    if exposure_room < notional * 0.5:
        v.veto = f"MARUZİYET odası {exposure_room:.0f}$ < emir {notional:.0f}$"
        return v
    # korelasyon: açık pozisyonla yüksek korelasyon → boyut ×0,5, iki tane varsa veto
    hits = []
    if corr and corr.get("symbols") and corr.get("matrix"):
        syms = corr["symbols"]
        M = corr["matrix"]
        if symbol in syms:
            i = syms.index(symbol)
            for s, pos in open_positions.items():
                if s in syms and pos.get("direction") == direction:
                    j = syms.index(s)
                    c = _num(M[i][j], 0.0) or 0.0
                    if abs(c) >= corr_limit:
                        hits.append((s, c))
    if len(hits) >= 2:
        v.veto = "KORELASYON: aynı yönde 2+ yüksek korelasyonlu açık pozisyon"
        return v
    if hits:
        v.size_mult *= 0.5
        v.notes.append(f"{hits[0][0]} ile korelasyon {hits[0][1]:.2f} → ×0,5")
    # Kelly tavanı (¼ Kelly): f* = p − (1−p)/b
    b = max(0.1, rr)
    f_star = p_win - (1.0 - p_win) / b
    if f_star <= 0:
        v.size_mult *= 0.5
        v.notes.append(f"Kelly ≤ 0 (p={p_win:.2f}, R/R={rr:.2f}) → ×0,5")
    else:
        v.notes.append(f"¼-Kelly %{f_star * 25:.1f} (p={p_win:.2f}, R/R={rr:.2f})")
    return v


# ═══════════════════════════════════════════════════════════════════════════
# 9) MAKRO & OLAY — takvim yakınlığı, korku/açgözlülük
# ═══════════════════════════════════════════════════════════════════════════
def role_macro(events: Optional[List[Dict]], signal: Optional[Dict]) -> RoleVote:
    v = RoleVote("makro_olay")
    if events is None and not signal:
        v.data_ok = False
        return v
    score, conf = 0.0, 0.3
    soon = []
    for e in events or []:
        d = _num(e.get("in_days"))
        if d is None:
            continue
        imp = str(e.get("impact", "")).lower()
        if d <= 0.1 and "yüksek" in imp:            # ~2,4 saat
            v.veto = f"OLAY: {e.get('name')} {d*24:.1f} saat içinde"
        elif d <= 0.5 and "yüksek" in imp:
            v.size_mult *= 0.5
            soon.append(f"{e.get('name')} ({d*24:.0f} sa)")
    if soon:
        v.notes.append("yakın yüksek-etkili olay → ×0,5: " + ", ".join(soon[:2]))
    for l in (signal or {}).get("layer_breakdown") or []:
        if l.get("layer") in ("fear_greed", "macro", "news"):
            c = _num(l.get("contribution"), 0.0) or 0.0
            score += c
            conf = max(conf, min(1.0, abs(c) * 4))
            v.notes.append(f"{l['layer']} katkı {c:+.3f}")
    v.score = _clip(score * 3)
    v.confidence = _clip(conf, 0.0, 1.0)
    return v


# ═══════════════════════════════════════════════════════════════════════════
# 10) SOSYAL & DUYARLILIK — yalnız ölçülmüş hesaplar
# ═══════════════════════════════════════════════════════════════════════════
def role_social(social: Optional[Dict], signal: Optional[Dict]) -> RoleVote:
    v = RoleVote("sosyal_duyarlilik")
    n_meas = int((social or {}).get("n_measured") or 0)
    score = 0.0
    for l in (signal or {}).get("layer_breakdown") or []:
        if l.get("layer") in ("sentiment",):
            score += _num(l.get("contribution"), 0.0) or 0.0
    if n_meas == 0 and abs(score) < 1e-6:
        v.data_ok = False
        v.notes.append("ölçülmüş sosyal hesap yok → oy yok")
        return v
    v.score = _clip(score * 4)
    v.confidence = _clip(0.2 + 0.05 * n_meas, 0.0, 0.6)
    v.notes.append(f"{n_meas} ölçülmüş hesap · duyarlılık katkısı {score:+.3f}")
    return v


# ═══════════════════════════════════════════════════════════════════════════
# 11) ORKESTRATÖR KONSENSÜSÜ — 11 katmanlı 4h karar motoru
# ═══════════════════════════════════════════════════════════════════════════
def role_orchestrator(signal: Optional[Dict]) -> RoleVote:
    v = RoleVote("orkestrator_konsensusu")
    if not signal:
        v.data_ok = False
        v.notes.append("orkestratör anlık görüntüsü yok")
        return v
    d = str(signal.get("direction") or "FLAT").upper()
    conf = _num(signal.get("confidence"), 0.0) or 0.0
    cls = str(signal.get("signal_class") or "")
    if cls == "acil_cikis":
        v.veto = "orkestratör ACİL ÇIKIŞ"
    v.score = (1.0 if d == "LONG" else -1.0 if d == "SHORT" else 0.0) * conf
    v.confidence = conf if d != "FLAT" else 0.2
    v.notes.append(f"{d} güven %{conf*100:.0f} · {cls or '—'} · momentum {signal.get('momentum_score', '—')}")
    top = (signal.get("reasons") or [])[:2]
    v.notes.extend(str(r)[:80] for r in top)
    return v


# ═══════════════════════════════════════════════════════════════════════════
# 12) DENETÇİ — değişmezler; ihlal = giriş yok
# ═══════════════════════════════════════════════════════════════════════════
def role_auditor(direction: str, entry: float, stop: float, target: float,
                 target_gross_pct: float, cost_pct: float, notional: float,
                 max_order: float, ctx_age_sec: Optional[float], max_ctx_age_sec: float,
                 rr_min: float) -> RoleVote:
    v = RoleVote("denetci")
    v.confidence = 1.0
    bad = []
    s = 1.0 if direction == "LONG" else -1.0
    if not all(map(lambda x: x is not None and math.isfinite(x) and x > 0, (entry, stop, target))):
        bad.append("seviye NaN/≤0")
    else:
        if (entry - stop) * s <= 0:
            bad.append("stop yanlış tarafta")
        if (target - entry) * s <= 0:
            bad.append("hedef yanlış tarafta")
        rr = abs(target - entry) / max(1e-12, abs(entry - stop))
        if rr < rr_min:
            bad.append(f"R/R {rr:.2f} < {rr_min}")
        v.notes.append(f"R/R {rr:.2f}")
    if target_gross_pct <= cost_pct:
        bad.append("hedef maliyeti karşılamıyor")
    if notional > max_order + 1e-9:
        bad.append(f"boyut {notional:.0f} > tavan {max_order:.0f}")
    if ctx_age_sec is None:
        bad.append("bağlam yok (ne ağır ne hafif) — kör işlem yok")
    if ctx_age_sec is not None and ctx_age_sec > max_ctx_age_sec:
        bad.append(f"yavaş bağlam bayat ({ctx_age_sec/60:.0f} dk)")
    if bad:
        v.veto = "DENETÇİ: " + "; ".join(bad)
    else:
        v.notes.append("değişmezler temiz")
    return v


# ═══════════════════════════════════════════════════════════════════════════
# 10b) HABER & SOSYAL TARAYICI — news_scanner çıktısı; ciddi risk → VETO
# ═══════════════════════════════════════════════════════════════════════════
def role_news_social(news: Optional[Dict], social: Optional[Dict], signal: Optional[Dict]) -> RoleVote:
    v = RoleVote("haber_sosyal")
    n_meas = int((social or {}).get("n_measured") or 0)
    if not news or not news.get("data_ok"):
        base = role_social(social, signal)
        base.role = "haber_sosyal"
        if not base.data_ok:
            base.notes = ["haber/sosyal verisi yok (tarayıcı henüz koşmadı ya da eşleşme yok)"]
        return base
    score = float(news.get("score") or 0.0)
    n = int(news.get("n_items") or 0)
    conf = min(0.8, 0.25 + 0.08 * n)
    soc = news.get("social") or {}
    if soc.get("ratio") is not None and (soc.get("bull", 0) + soc.get("bear", 0)) >= 5:
        score = 0.7 * score + 0.3 * float(soc["ratio"])
        v.notes.append(f"StockTwits boğa {soc.get('bull')} / ayı {soc.get('bear')} (24 sa {soc.get('msgs_24h')})")
    if news.get("confirmed"):
        conf = min(1.0, conf + 0.15)
        v.notes.append(f"hareketlilik DOĞRULANDI: 4 sa %{news.get('move_pct_4h')} · hacim ×{news.get('vol_ratio')}")
    elif news.get("confirmed") is False:
        conf *= 0.8
        v.notes.append("haber var, hareket doğrulanmadı")
    if news.get("severe_risk"):
        v.veto = "HABER RİSKİ: " + ", ".join(k for k in (news.get("risks") or {}) )
    elif news.get("risks"):
        v.size_mult = 0.7
        v.notes.append("risk etiketi: " + ", ".join(news["risks"]))
    if news.get("catalysts"):
        v.notes.append("katalizör: " + ", ".join(news["catalysts"]))
    for hl in (news.get("headlines") or [])[:2]:
        v.notes.append(f"[{hl.get('source')}] {str(hl.get('title'))[:80]}")
    if n_meas:
        v.notes.append(f"{n_meas} ölçülmüş sosyal hesap")
    v.score = _clip(score)
    v.confidence = _clip(conf, 0.0, 1.0)
    v.notes.insert(0, f"haber skoru {score:+.2f} · {n} başlık · boğa {news.get('bull')} / ayı {news.get('bear')} (sözlük, ölçülmedi)")
    return v
