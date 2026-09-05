"""KOD HİJYENİ — metin araması değil, AST düzeyinde kesin kontroller.

Bu dosya bir "linter" değil; her kural, bu projede FİİLEN yaşanmış ya da
yaşanması hâlinde sessizce yanlış sonuç üretecek bir davranışı kilitler.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

Q = Path(__file__).resolve().parents[1] / "agi_trader" / "qualification"
OPP = Path(__file__).resolve().parents[1] / "agi_trader" / "opportunity"

# Yerleşik `hash()` süreçler arası TUZLANIR: aynı girdi farklı çalıştırmada
# farklı sayı verir. Bu projede bir kez tekrarlanamaz tohum üretti.
YASAK_AD = {"hash", "eval", "exec"}
# Genel RNG durumunu kirleten / tohumsuz çağrılar
YASAK_YOL = {("np", "random", "seed"), ("random", "seed"),
             ("random", "random"), ("np", "random", "rand"),
             ("np", "random", "randn")}


def _cagrilar(dosya: Path):
    t = ast.parse(dosya.read_text(encoding="utf-8"))
    for n in ast.walk(t):
        if not isinstance(n, ast.Call):
            continue
        fn = n.func
        if isinstance(fn, ast.Name):
            yield fn.id, (fn.id,), n.lineno
        elif isinstance(fn, ast.Attribute):
            p, k = [], fn
            while isinstance(k, ast.Attribute):
                p.append(k.attr)
                k = k.value
            if isinstance(k, ast.Name):
                p.append(k.id)
            yol = tuple(reversed(p))
            yield fn.attr, yol, n.lineno


def _dosyalar():
    return sorted(Q.glob("*.py")) + sorted(OPP.glob("*.py"))


def test_yerlesik_hash_kullanilmaz():
    """`hash()` süreçler arası tuzlanır — tekrarlanabilirliği bozar."""
    kotu = [f"{f.name}:{ln} {ad}()" for f in _dosyalar()
            for ad, yol, ln in _cagrilar(f) if ad in YASAK_AD and len(yol) == 1]
    assert not kotu, f"yasak çağrı: {kotu}"


def test_tohumsuz_rng_kullanilmaz():
    """`np.random.seed` genel durumu kirletir; `default_rng(seed)` kullanılır."""
    kotu = [f"{f.name}:{ln} {'.'.join(yol)}()" for f in _dosyalar()
            for ad, yol, ln in _cagrilar(f) if yol in YASAK_YOL]
    assert not kotu, f"tohumsuz/kirletici RNG: {kotu}"


def test_butun_moduller_ayristirilabilir():
    for f in _dosyalar():
        ast.parse(f.read_text(encoding="utf-8"))


def test_yakinsama_verdikti_tek_noktada():
    """Karar iki yerde verilirse biri güncellenip diğeri unutulur.

    BU FİİLEN OLDU: `REGIME_DEPENDENT` eklendiğinde 472 hücre yanlış
    etiketlendi."""
    res = (Q / "research.py").read_text(encoding="utf-8")
    assert "verdikt = REGIME_DEPENDENT" not in res, \
        "verdikt mantığı research.py'de tekrar ediyor"
    assert "decide_verdict" in res, "tek karar noktası kullanılmıyor"


def test_yakinsama_esikleri_tek_yerde():
    res = (Q / "research.py").read_text(encoding="utf-8")
    for e in ("MAX_CI_WIDTH", "MIN_EFF_SAMPLE", "MAX_PERIOD_SPREAD",
              "MAX_REGIME_SPREAD", "MIN_SHRINK_RATIO"):
        assert f"{e} = " not in res, f"{e} iki yerde tanımlı"


def test_UNKNOWN_otopiloti_engeller():
    """Ölçülmemiş bir sağlık, sağlık değildir."""
    from agi_trader.qualification.safety import BLOCKING, UNKNOWN
    assert UNKNOWN in BLOCKING


def test_canli_cikti_json_guvenli():
    live = (Q / "live.py").read_text(encoding="utf-8")
    assert "return json_safe(" in live, \
        "canlı çıktı json_safe'ten geçmiyor — inf/NaN 500 üretir"


# ── determinizm ────────────────────────────────────────────────────────

def test_yakinsama_iki_kosuda_ayni():
    from agi_trader.qualification.convergence import assess
    rng = np.random.default_rng(4)
    lab = (rng.random(5000) < 0.3).astype(int)
    g = {"x": (300, 1000), "y": (305, 1000)}
    assert assess(lab, 1, 12, 5000.0, g, None).to_dict() == \
challenge if False else assess(lab, 1, 12, 5000.0, g, None).to_dict()


def test_kimlik_iki_kosuda_ayni():
    from agi_trader.qualification.ledger import make_id
    assert make_id("B", "4h", "LONG", "t") == make_id("B", "4h", "LONG", "t")


def test_bootstrap_iki_kosuda_ayni():
    from agi_trader.qualification.stats import block_bootstrap_ci
    rng = np.random.default_rng(4)
    lab = (rng.random(5000) < 0.3).astype(int)
    assert block_bootstrap_ci(lab, 1, 50, n_boot=200) == \
        block_bootstrap_ci(lab, 1, 50, n_boot=200)


def test_seyreltme_deterministik():
    """Kantil/otokorelasyon seyreltmesi sabit adımlı olmalı — rastgele değil."""
    from agi_trader.qualification.baserate import _subsample
    x = np.arange(200_000, dtype=float)
    a, b = _subsample(x), _subsample(x)
    assert np.array_equal(a, b) and len(a) < len(x)
