"""
YENİDEN GİRİŞ KAPISI — "aynı hareketten birkaç kez kâr al", ama körlemesine değil.

NEDEN VAR
─────────
İki ayrı kusur aynı yerden geliyordu: koşucu bir pariteden çıktıktan sonra
HİÇBİR kapı yoktu.

  • İYİ TARAF kaçırılıyordu: trend sürerken kâr merdiveninden çıkılan pozisyona
    geri girilemiyordu — hareketin ikinci ve üçüncü bacağı alınamıyordu.
  • KÖTÜ TARAF serbestti: stop yiyen bir fikre aynı döngüde yeniden girilebiliyordu.
    2026-09-06 canlı ölçümünde EARLY_ABORT + STOP kovaları 41 işlemde −9,07 $ üretti;
    aynı fikri hemen tekrarlamak bu kovayı büyütür, komisyonu iki katına çıkarır.

Bu modül TEK BAŞINA giriş AÇMAZ. Yalnız izin verir ya da vermez; aday hâlâ
komite/karar zincirinin bütün kapılarından geçmek zorundadır. Yani kapı yalnız
işlem sayısını AZALTABİLİR ya da bir çıkıştan sonra geri girişe İZİN verir.

KURALLAR (hepsi ölçülebilir, hepsi gerekçeli)
─────────────────────────────────────────────
1. MALİYET KAPISI (her durumda): beklenen sonraki salınım ≥ gidiş-dönüş maliyet ×
   `min_swing_cost_mult`. "Borsa ücretinin üstünde miktarlarda oyna" kuralının
   yeniden girişteki karşılığı budur.
2. ZARARLA çıkıldıysa aynı yöne dönüş `loss_cooldown_sec` boyunca KAPALI.
   Sebep: aynı geçersiz olmuş fikri tekrarlamak, ölçülmüş en pahalı kovadır.
3. KÂR KORUYARAK çıkıldıysa (GIVEBACK/TRAIL/LADDER/TP) `cooldown_sec` sonrası
   yeniden girişe izin verilir — hareketin sonraki bacağı bunun içindir.
4. TERS yöne dönmek için `opposite_cooldown_sec` beklenir (kamçı önlemi).
5. Aynı hareket penceresinde en çok `max_reentries` yeniden giriş.
6. Devam olasılığı ölçülüyorsa (kalibre değil — yalnız DESTEK olarak) düşükse
   yeniden girişe izin verilmez.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

# Kâr KORUYARAK kapanan çıkış sebepleri — hareket geçersizleşmedi, yalnız kâr alındı.
PROFIT_EXITS = {"GIVEBACK", "TRAIL", "TP", "LADDER_TP", "LADDER_SON", "MODEL_EXIT", "BE_LOCK"}
# Fikrin geçersizleştiğini gösteren çıkışlar.
INVALIDATING_EXITS = {"STOP", "EARLY_ABORT", "NAKİT MODU"}


@dataclass
class ReentryParams:
    cooldown_sec: int = 300               # kârla çıkıştan sonra en az bekleme
    loss_cooldown_sec: int = 1800         # zararla çıkıştan sonra aynı yöne dönüş yasağı
    opposite_cooldown_sec: int = 900      # ters yöne dönüş için bekleme (kamçı önlemi)
    max_reentries: int = 3                # aynı hareket penceresinde en çok kaç kez
    move_window_sec: int = 7200           # "aynı hareket" sayılan pencere
    # Beklenen salınım / gidiş-dönüş maliyet. Mevcut `min_gross_to_cost` (2,0) PLAN
    # hedefine bakar — 85 canlı işlemin 1'i o hedefe ulaştı, yani iyimserdir. Bu kapı
    # ÖLÇÜLMÜŞ ulaşılabilir hedefe (sleeve MFE medyanı) bakar; aynı 2,0 katsayısı burada
    # gerçekten bağlayıcıdır. Daha yükseği (2,5 / 3,0) replay ile ölçülmeden sabitlenmez.
    min_swing_cost_mult: float = 2.0
    min_cont_prob: float = 0.45           # ölçülüyorsa devam olasılığı tabanı (kalibre DEĞİL)
    enabled: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "ReentryParams":
        c = cls()
        for k, v in (d or {}).items():
            if hasattr(c, k) and v is not None:
                cur = getattr(c, k)
                try:
                    setattr(c, k, bool(v) if isinstance(cur, bool) else type(cur)(v))
                except (TypeError, ValueError):
                    pass
        return c.validated()

    def validated(self) -> "ReentryParams":
        self.cooldown_sec = int(min(7200, max(0, self.cooldown_sec)))
        self.loss_cooldown_sec = int(min(86400, max(0, self.loss_cooldown_sec)))
        self.opposite_cooldown_sec = int(min(86400, max(0, self.opposite_cooldown_sec)))
        self.max_reentries = int(min(10, max(0, self.max_reentries)))
        self.move_window_sec = int(min(86400, max(60, self.move_window_sec)))
        self.min_swing_cost_mult = float(min(10.0, max(1.0, self.min_swing_cost_mult)))
        self.min_cont_prob = float(min(0.9, max(0.0, self.min_cont_prob)))
        return self


def record_exit(state: Dict, symbol: str, direction: str, reason: str, net_pnl: float,
                peak_net_pct: float, now: float, p: ReentryParams) -> Dict:
    """Kapanışı kaydet. `state` koşucuda tutulur ve diske yazılır."""
    prev = state.get(symbol) or {}
    same_move = (now - float(prev.get("first_ts") or 0.0)) <= p.move_window_sec \
        and str(prev.get("direction") or "") == direction
    rec = {
        "symbol": symbol, "direction": direction, "reason": reason,
        "net_pnl": float(net_pnl), "peak_net_pct": float(peak_net_pct or 0.0),
        "ts": float(now),
        "first_ts": float(prev.get("first_ts") or now) if same_move else float(now),
        "count": int(prev.get("count") or 0) + 1 if same_move else 1,
        "profit_exit": reason in PROFIT_EXITS and float(net_pnl) > 0,
    }
    state[symbol] = rec
    return rec


def decide(state: Dict, symbol: str, direction: str, now: float,
           expected_swing_pct: Optional[float], cost_pct: float,
           p: ReentryParams, cont_prob: Optional[float] = None) -> Dict:
    """{'allowed': bool, 'reason': str, 'reentry_count': int, 'gate': str|None}"""
    prev = state.get(symbol)
    out = {"allowed": True, "reason": "ilk giriş", "reentry_count": 0, "gate": None}
    if not p.enabled:
        return {**out, "reason": "kapı kapalı"}

    # 1) MALİYET KAPISI — her giriş için, yeniden giriş olsun olmasın
    if cost_pct > 0 and expected_swing_pct is not None:
        need = cost_pct * p.min_swing_cost_mult
        if float(expected_swing_pct) < need:
            return {"allowed": False, "gate": "SALINIM_MALİYET", "reentry_count": 0,
                    "reason": (f"beklenen salınım %{float(expected_swing_pct):.3f} < maliyet %{cost_pct:.3f} × "
                               f"{p.min_swing_cost_mult} = %{need:.3f} — komisyon için işlem yapılmaz")}

    if not prev:
        return out
    age = now - float(prev.get("ts") or 0.0)
    same_dir = str(prev.get("direction") or "") == direction
    cnt = int(prev.get("count") or 0)
    in_window = (now - float(prev.get("first_ts") or 0.0)) <= p.move_window_sec

    # 4) TERS yön — kamçı önlemi
    if not same_dir:
        if age < p.opposite_cooldown_sec:
            return {"allowed": False, "gate": "TERS_YÖN_SOĞUMA", "reentry_count": 0,
                    "reason": (f"{symbol} {prev.get('direction')} pozisyonu {age/60:.0f} dk önce kapandı; "
                               f"ters yöne dönüş için {p.opposite_cooldown_sec/60:.0f} dk beklenir")}
        return {**out, "reason": "ters yön soğuması doldu"}

    # 2) ZARARLA çıkış — aynı fikre hemen dönme
    invalid = str(prev.get("reason")) in INVALIDATING_EXITS or float(prev.get("net_pnl") or 0.0) <= 0
    if invalid:
        if age < p.loss_cooldown_sec:
            return {"allowed": False, "gate": "ZARAR_SOĞUMA", "reentry_count": 0,
                    "reason": (f"{symbol} {prev.get('reason')} ile {age/60:.0f} dk önce kapandı "
                               f"(net {float(prev.get('net_pnl') or 0):+.3f} $); aynı yöne dönüş için "
                               f"{p.loss_cooldown_sec/60:.0f} dk beklenir")}
        return {**out, "reason": "zarar soğuması doldu"}

    # 3) KÂR KORUYARAK çıkış — hareketin sonraki bacağı
    if age < p.cooldown_sec:
        return {"allowed": False, "gate": "SOĞUMA", "reentry_count": cnt,
                "reason": f"{symbol} {age:.0f} sn önce kârla kapandı; {p.cooldown_sec} sn soğuma"}
    if in_window and cnt >= p.max_reentries:
        return {"allowed": False, "gate": "YENİDEN_GİRİŞ_TAVANI", "reentry_count": cnt,
                "reason": f"{symbol} aynı harekette {cnt} kez işlem gördü (tavan {p.max_reentries})"}
    if cont_prob is not None and float(cont_prob) < p.min_cont_prob:
        return {"allowed": False, "gate": "DEVAM_OLASILIĞI", "reentry_count": cnt,
                "reason": (f"devam olasılığı {float(cont_prob):.2f} < {p.min_cont_prob} "
                           f"(sezgisel — kalibre değil, yalnız yeniden girişte kullanılır)")}
    return {"allowed": True, "gate": None, "reentry_count": (cnt if in_window else 0),
            "reason": (f"yeniden giriş #{cnt + 1}: önceki bacak {prev.get('reason')} ile "
                       f"{float(prev.get('net_pnl') or 0):+.3f} $ kârla kapandı, hareket sürüyor")}
