"""KOŞU KİMLİĞİ — yeniden üretilebilirlik iddiasının kanıtı (§XCI, CI).

"Bu sonucu yeniden üretebiliriz" cümlesi, ancak koşunun HANGİ veriyle, HANGİ
kodla ve HANGİ ayarlarla üretildiği kayıtlıysa doğrulanabilir. Aksi hâlde bir
temenni olur.

BU MODÜL DÖRT ŞEYİ MÜHÜRLER

  dataset_hash  — girdi dosyalarının içerik parmak izi (yol + boyut + mtime +
                  satır sayısı). Tam içerik özeti 435 MB'ı okumak demek olurdu;
                  bu daha ucuz ve veri değişince DEĞİŞİR.
  code_commit   — git varsa commit; yoksa kaynak dosyaların içerik özeti.
                  "git yok" diye boş bırakmak, kod değişimini görünmez yapardı.
  config_hash   — koşuyu belirleyen bütün parametreler tek özet.
  seed          — sabit ve açık.

⚠️ `hash()` KULLANILMAZ: Python'un yerleşik hash'i süreçler arası tuzlanır ve
aynı girdi için farklı sayı üretir. Bu proje bu hatayı bir kez yaptı.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def dataset_fingerprint(paths: Sequence[Path]) -> Dict:
    """Girdi dosyalarının parmak izi.

    İçeriğin tamamı okunmaz (435 MB); yol + boyut + değişim zamanı + varsa
    satır sayısı kullanılır. Veri değişirse bu özet DEĞİŞİR; amaç budur."""
    kayit: List[Dict] = []
    for p in sorted(Path(x) for x in paths):
        if not p.exists():
            kayit.append({"path": p.name, "missing": True})
            continue
        st = p.stat()
        satir = None
        if p.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
                satir = int(pq.ParquetFile(p).metadata.num_rows)
            except Exception:
                pass
        kayit.append({"path": p.name, "bytes": st.st_size,
                      "mtime": int(st.st_mtime), "rows": satir})
    ozet = _sha(json.dumps(kayit, sort_keys=True))
    return {"hash": ozet[:16], "n_files": len(kayit), "files": kayit}


def code_fingerprint(pkg_dir: Path) -> Dict:
    """git commit; yoksa kaynak dosyaların içerik özeti.

    Git yoksa "bilinmiyor" yazıp geçmek, kod değişimini görünmez yapar —
    bu yüzden yedek olarak dosya içerikleri özetlenir."""
    commit = dirty = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(pkg_dir),
            stderr=subprocess.DEVNULL, timeout=10).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(pkg_dir),
            stderr=subprocess.DEVNULL, timeout=10).decode().strip())
    except Exception:
        pass
    h = hashlib.sha256()
    n = 0
    for p in sorted(Path(pkg_dir).rglob("*.py")):
        if "__pycache__" in str(p):
            continue
        try:
            h.update(p.read_bytes())
            n += 1
        except Exception:
            continue
    return {"git_commit": commit, "git_dirty": dirty,
            "source_hash": h.hexdigest()[:16], "n_source_files": n,
            "note": ("git yoksa kaynak içerik özeti kullanılır — kod değişimi "
                     "her hâlükârda görünür")}


@dataclass
class RunProvenance:
    run_id: str
    started_at: str
    seed: int
    dataset: Dict
    code: Dict
    config: Dict
    config_hash: str
    environment: Dict
    finished_at: Optional[str] = None
    duration_sec: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    def finish(self, t0: float) -> "RunProvenance":
        self.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.duration_sec = round(time.time() - t0, 1)
        return self


def build(seed: int, data_paths: Sequence[Path], pkg_dir: Path,
          config: Dict) -> RunProvenance:
    """Koşu kimliğini üret. `run_id` DETERMİNİSTİK değildir (zaman içerir)
    ama içindeki üç özet deterministiktir — aynı veri+kod+ayar aynı özetleri
    verir, farklı zamanda koşulsa bile."""
    ds = dataset_fingerprint(data_paths)
    cd = code_fingerprint(pkg_dir)
    ch = _sha(json.dumps(config, sort_keys=True, default=str))[:16]
    simdi = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return RunProvenance(
        run_id=f"{simdi}-{ds['hash'][:6]}-{cd['source_hash'][:6]}-{ch[:6]}",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        seed=int(seed), dataset=ds, code=cd, config=config, config_hash=ch,
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": _ver("numpy"), "pandas": _ver("pandas"),
            "scipy": _ver("scipy"),
            "cpu_count": os.cpu_count(),
        })


def _ver(mod: str) -> Optional[str]:
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def same_inputs(a: Dict, b: Dict) -> bool:
    """İki koşu AYNI girdilerle mi koşuldu? (zaman damgası hariç)"""
    return (a.get("dataset", {}).get("hash") == b.get("dataset", {}).get("hash")
            and a.get("code", {}).get("source_hash")
            == b.get("code", {}).get("source_hash")
            and a.get("config_hash") == b.get("config_hash")
            and a.get("seed") == b.get("seed"))
