"""
STRATEGY HARVESTER — açık kaynak stratejiyi ÇALIŞTIRMADAN hipoteze dönüştürür.

Boru hattı (kod kör kopyalanmaz):
  OPEN SOURCE → LICENSE CHECK → STATIC REVIEW (AST; ağ/exec/eval/subprocess yok mu?)
  → LOGIC EXTRACT (indikatörler, giriş/çıkış koşulları, zaman dilimi, stop/ROI)
  → NORMALIZE (CryptoMind özellik ailelerine eşleme; uygulanabilirlik)
  → RESEARCH_INBOX (runs/research/inbox/*.json) — otomatik üretime GEÇMEZ.

Desteklenen biçim: Freqtrade IStrategy (populate_indicators / populate_entry_trend|buy_trend /
populate_exit_trend|sell_trend, timeframe, stoploss, minimal_roi). Diğer dosyalar "GENERIC"
olarak yalnız indikatör/koşul metni çıkarımıyla kaydedilir.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

KNOWN_LICENSES = {"MIT": "MIT", "Apache": "Apache-2.0", "GPL": "GPL", "BSD": "BSD", "MPL": "MPL-2.0", "Unlicense": "Unlicense"}
FORBIDDEN_CALLS = {"exec", "eval", "compile", "__import__"}
FORBIDDEN_MODULES = {"subprocess", "socket", "requests", "urllib", "http", "os.system", "ctypes"}
# Freqtrade/TA isim → CryptoMind özellik ailesi (paylaşımlı özellik deposu)
INDICATOR_FAMILY = {
    "EMA": "ema", "SMA": "ema", "TEMA": "ema", "DEMA": "ema", "WMA": "ema", "HMA": "ema", "MACD": "momentum",
    "RSI": "rsi", "STOCH": "rsi", "STOCHRSI": "rsi", "CCI": "rsi", "WILLR": "rsi", "MFI": "rsi",
    "ADX": "trend_strength", "DMI": "trend_strength", "PLUS_DI": "trend_strength", "MINUS_DI": "trend_strength",
    "ATR": "atr", "NATR": "atr", "BBANDS": "bollinger", "bollinger_bands": "bollinger", "KC": "keltner",
    "SAR": "trend", "ICHIMOKU": "trend", "OBV": "volume", "AD": "volume", "volume": "volume", "VWAP": "vwap", "vwap": "vwap",
    "ROC": "momentum", "MOM": "momentum", "AROON": "trend", "HT_TRENDLINE": "trend", "CDL": "candles",
    "heikinashi": "candles", "typical_price": "vwap", "fisher": "rsi", "supertrend": "trend", "donchian": "breakout",
}
AVAILABLE_FAMILIES = {"ema", "rsi", "atr", "bollinger", "keltner", "momentum", "trend", "trend_strength", "volume",
                      "vwap", "candles", "breakout"}


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:12]


def detect_license(text: str, sibling_files: Optional[Dict[str, str]] = None) -> Dict:
    """Dosya başlığındaki SPDX/lisans satırı ya da depo kökündeki LICENSE dosyası."""
    m = re.search(r"SPDX-License-Identifier:\s*([A-Za-z0-9.\-+]+)", text)
    if m:
        return {"license": m.group(1), "source": "spdx_header", "ok": True}
    for name, body in (sibling_files or {}).items():
        if name.upper().startswith("LICENSE") or name.upper().startswith("COPYING"):
            for k, v in KNOWN_LICENSES.items():
                if k.lower() in body[:2000].lower():
                    return {"license": v, "source": name, "ok": True}
            return {"license": "UNKNOWN", "source": name, "ok": False}
    head = text[:1500].lower()
    for k, v in KNOWN_LICENSES.items():
        if f"{k.lower()} license" in head or f"licensed under the {k.lower()}" in head:
            return {"license": v, "source": "header", "ok": True}
    return {"license": "NONE", "source": None, "ok": False}


def static_review(tree: ast.AST) -> Dict:
    """Ağ/exec/subprocess kullanan strateji dosyası araştırmaya bile alınmaz."""
    issues = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = n.func
            name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
            if name in FORBIDDEN_CALLS:
                issues.append(f"yasak çağrı: {name}")
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                mod = (n.module + "." if isinstance(n, ast.ImportFrom) and n.module else "") + a.name
                if any(mod.split(".")[0] == f.split(".")[0] for f in FORBIDDEN_MODULES):
                    issues.append(f"yasak modül: {mod}")
    return {"ok": not issues, "issues": issues}


def _const(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _indicators(tree: ast.AST) -> List[str]:
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = n.func
            if isinstance(fn, ast.Attribute):
                base = fn.value.id if isinstance(fn.value, ast.Name) else ""
                if base in ("ta", "talib", "qtpylib", "pta", "ftt", "technical", "pandas_ta"):
                    names.add(fn.attr)
            elif isinstance(fn, ast.Name) and fn.id.upper() in INDICATOR_FAMILY:
                names.add(fn.id)
    return sorted(names)


def _conditions(fn: ast.FunctionDef) -> List[str]:
    """dataframe.loc[(cond), 'enter_long'] = 1 kalıbındaki koşul metinleri."""
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and n.targets and isinstance(n.targets[0], ast.Subscript):
            sub = n.targets[0]
            sl = sub.slice
            if isinstance(sl, ast.Tuple) and sl.elts:
                try:
                    out.append(ast.unparse(sl.elts[0])[:400])
                except Exception:
                    pass
    return out


def parse_strategy(text: str, origin: str = "", sibling_files: Optional[Dict[str, str]] = None) -> Dict:
    rec = {"origin": origin, "sha": _sha(text), "ts": time.time(), "format": "GENERIC", "status": "RESEARCH_INBOX",
           "pipeline_stage": "LOGIC_EXTRACT"}
    lic = detect_license(text, sibling_files)
    rec["license"] = lic
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        rec.update({"status": "REJECTED", "reject_reason": f"syntax: {e}", "pipeline_stage": "STATIC_REVIEW"})
        return rec
    rev = static_review(tree)
    rec["static_review"] = rev
    if not rev["ok"]:
        rec.update({"status": "REJECTED", "reject_reason": "; ".join(rev["issues"]), "pipeline_stage": "STATIC_REVIEW"})
        return rec
    if not lic["ok"]:
        rec.update({"status": "LICENSE_UNKNOWN", "pipeline_stage": "LICENSE_CHECK"})
    classes = [c for c in ast.walk(tree) if isinstance(c, ast.ClassDef)]
    strat = next((c for c in classes if any((isinstance(b, ast.Name) and b.id == "IStrategy") or
                                            (isinstance(b, ast.Attribute) and b.attr == "IStrategy") for b in c.bases)), None)
    rec["indicators"] = _indicators(tree)
    fams = sorted({INDICATOR_FAMILY.get(i, INDICATOR_FAMILY.get(i.upper(), "unknown")) for i in rec["indicators"]})
    rec["feature_families"] = fams
    rec["implementable"] = bool(fams) and all(f in AVAILABLE_FAMILIES for f in fams)
    if strat is None:
        rec["name"] = (classes[0].name if classes else Path(origin).stem or "unknown")
        return rec
    rec["format"] = "FREQTRADE"
    rec["name"] = strat.name
    for n in strat.body:
        if isinstance(n, ast.Assign) and n.targets and isinstance(n.targets[0], ast.Name):
            k = n.targets[0].id
            if k in ("timeframe", "stoploss", "minimal_roi", "trailing_stop", "trailing_stop_positive",
                     "use_exit_signal", "can_short", "startup_candle_count"):
                rec[k] = _const(n.value)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            k = n.target.id
            if k in ("timeframe", "stoploss", "minimal_roi", "trailing_stop", "can_short"):
                rec[k] = _const(n.value)
        elif isinstance(n, ast.FunctionDef):
            if n.name in ("populate_entry_trend", "populate_buy_trend"):
                rec["entry_conditions"] = _conditions(n)
            elif n.name in ("populate_exit_trend", "populate_sell_trend"):
                rec["exit_conditions"] = _conditions(n)
    rec["hypothesis"] = _hypothesis(rec)
    return rec


def _hypothesis(rec: Dict) -> str:
    fam = ", ".join(rec.get("feature_families") or []) or "—"
    tf = rec.get("timeframe") or "?"
    n_e = len(rec.get("entry_conditions") or [])
    sl = rec.get("stoploss")
    return (f"{rec.get('name')}: {tf} zaman diliminde {fam} ailelerinden {n_e} giriş kuralı; "
            f"stop {sl if sl is not None else '?'}; ROI {rec.get('minimal_roi') or '?'}. "
            "Kaynak depo sonuçları parite/dönem bağımlıdır — CryptoMind maliyet modeliyle yeniden OOS test şart.")


class Harvester:
    def __init__(self, inbox_dir: Path):
        self.inbox = Path(inbox_dir)

    def harvest_text(self, text: str, origin: str, sibling_files: Optional[Dict[str, str]] = None) -> Dict:
        rec = parse_strategy(text, origin, sibling_files)
        try:
            self.inbox.mkdir(parents=True, exist_ok=True)
            (self.inbox / f"{rec['sha']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        except Exception:
            pass
        return rec

    def harvest_dir(self, path: Path) -> List[Dict]:
        path = Path(path)
        siblings = {p.name: p.read_text(encoding="utf-8", errors="ignore")[:4000]
                    for p in path.glob("*") if p.is_file() and p.name.upper().startswith(("LICENSE", "COPYING"))}
        out = []
        for p in sorted(path.rglob("*.py")):
            try:
                out.append(self.harvest_text(p.read_text(encoding="utf-8", errors="ignore"), str(p), siblings))
            except Exception as e:
                out.append({"origin": str(p), "status": "REJECTED", "reject_reason": f"{type(e).__name__}: {e}"})
        return out

    def harvest_urls(self, urls: List[str], timeout: float = 15.0) -> List[Dict]:
        """GitHub raw URL'leri: dosya + (varsa) depo kökündeki LICENSE. Ağ hatası = kayıt yok."""
        out = []
        for u in urls:
            try:
                text = urllib.request.urlopen(u, timeout=timeout).read().decode("utf-8", "ignore")
                sib = {}
                m = re.match(r"(https://raw\.githubusercontent\.com/[^/]+/[^/]+/[^/]+)/", u)
                if m:
                    for lic_name in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
                        try:
                            sib[lic_name] = urllib.request.urlopen(f"{m.group(1)}/{lic_name}", timeout=timeout).read().decode("utf-8", "ignore")[:4000]
                            break
                        except Exception:
                            continue
                out.append(self.harvest_text(text, u, sib))
            except Exception as e:
                out.append({"origin": u, "status": "FETCH_FAILED", "reject_reason": f"{type(e).__name__}"})
        return out

    def inbox_summary(self) -> Dict:
        rows = []
        if self.inbox.exists():
            for p in sorted(self.inbox.glob("*.json")):
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    rows.append({k: d.get(k) for k in ("name", "origin", "status", "pipeline_stage", "format", "timeframe",
                                                       "feature_families", "implementable", "license")})
                except Exception:
                    continue
        counts: Dict[str, int] = {}
        for r in rows:
            counts[str(r.get("status"))] = counts.get(str(r.get("status")), 0) + 1
        return {"n": len(rows), "counts": counts, "rows": rows[-50:]}
