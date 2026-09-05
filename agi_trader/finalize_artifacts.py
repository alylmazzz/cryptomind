#!/usr/bin/env python3
"""Tamamlanmış bir araştırma koşusuna SONRADAN eklenen artefaktları üretir.

NEDEN GEREKLİ
Koşu kimliği (`provenance.json`) ve model kayıt defteri (`model_registry.json`)
27 paritelik koşu BAŞLADIKTAN SONRA yazıldı; o yüzden koşunun kendisi bunları
üretmedi. İkisi de mevcut artefaktlardan türetilebilir:

  • provenance — veri/kod/ayar özeti zaten DETERMİNİSTİK; sonradan hesaplanır
    ve `recomputed_after_run: true` ile İŞARETLENİR. Koşu anındaki kodu değil,
    ŞU ANKİ kodu özetlediği için bunu gizlemek yanlış olur.
  • model kartları — `validation_report.json` içindeki ölçümlerden kurulur.

Bir sonraki tam koşuda ikisi de kendiliğinden üretilir; bu betik yalnız
mevcut koşuyu tamamlamak içindir.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

KOK = Path(__file__).parent
OUT = KOK / "runs" / "qualification"


def main() -> int:
    sys.path.insert(0, str(KOK))
    from agi_trader.qualification.model_registry import Registry, card_for_softmax
    from agi_trader.qualification.provenance import build as prov_build

    vr_p = OUT / "validation_report.json"
    if not vr_p.exists():
        print("validation_report.json yok — önce araştırma koşulmalı")
        return 1
    vr = json.loads(vr_p.read_text(encoding="utf-8"))
    uni = json.loads((OUT / "universe_5m.json").read_text(encoding="utf-8"))
    symbols = [x["symbol"] for x in uni.get("selected", [])]

    # ── koşu kimliği ────────────────────────────────────────────────────
    prov = prov_build(
        seed=vr.get("seed", 20260818),
        data_paths=[KOK / "runs" / "data_5m" / f"{s}_5m.parquet" for s in symbols],
        pkg_dir=KOK / "agi_trader",
        config={"symbols": symbols,
                "split": vr.get("split"),
                "base_resolution": vr.get("base_resolution"),
                "n_cells": vr.get("n_cells"),
                "n_trials": vr.get("n_trials_registry")})
    d = prov.to_dict()
    d["recomputed_after_run"] = True
    d["note"] = ("Bu kimlik koşu BİTTİKTEN SONRA hesaplandı. Veri ve ayar "
                 "özetleri koşuyu doğru temsil eder; KOD özeti ise şu anki "
                 "kaynağı yansıtır — koşu sırasında provenance/model-registry "
                 "modülleri henüz eklenmemişti. Sonraki tam koşuda kimlik "
                 "koşunun kendisi tarafından yazılır.")
    (OUT / "provenance.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"provenance.json → {prov.run_id}")

    # ── model kartları ──────────────────────────────────────────────────
    reg = Registry(OUT / "model_registry.json")
    n = 0
    for anahtar, m in (vr.get("models") or {}).items():
        if not m.get("ok"):
            continue
        hz, yon = anahtar.split("|")
        lt = m.get("locked_test") or {}
        sel = lt.get("selected") or {}
        reg.register(card_for_softmax(
            hz, yon, symbols,
            metrics={"oos_n": m.get("n_oos"),
                     "oos_brier": m.get("oos_brier"),
                     "oos_brier_base": m.get("oos_brier_base"),
                     "oos_ece": m.get("oos_ece"),
                     "calibration_slope": (m.get("oos_calibration") or {}).get("slope"),
                     "locked_test_n": lt.get("n"),
                     "locked_test_baseline": lt.get("baseline"),
                     "locked_test_tp_rate": sel.get("tp_rate"),
                     "locked_test_net_mean": sel.get("net_mean_pct"),
                     "locked_test_t": sel.get("t_stat"),
                     "positive_subperiod_frac": lt.get("positive_subperiod_frac"),
                     "psi": m.get("psi")},
            provenance={"run_id": prov.run_id,
                        "dataset_hash": prov.dataset["hash"],
                        "code_hash": prov.code["source_hash"],
                        "config_hash": prov.config_hash, "seed": prov.seed,
                        "recomputed_after_run": True},
            train_end=str((vr.get("split") or {}).get("train", "")).replace("< ", ""),
            valid_end=str((vr.get("split") or {}).get("locked_test", ""))
                        .replace("≥ ", "")))
        n += 1
    print(f"model_registry.json → {n} kart (hiçbiri APPROVED değil)")

    # ── yakınsama verdiktlerini GÜNCEL mantıkla yeniden hesapla ─────────
    # Koşu, REGIME_DEPENDENT ayrımı eklenmeden önce bitti; matrix.json'daki
    # verdiktler "rejime koşullu" durumları UNSTABLE gösteriyor. Yayılımlar
    # zaten kayıtlı olduğu için karar YENİDEN hesaplanabilir — koşuyu
    # tekrarlamaya gerek yok. Ölçüm değişmiyor, yalnız ETİKET düzeliyor.
    from agi_trader.qualification.convergence import (
        CONVERGED, CONVERGING, REGIME_DEPENDENT, UNMEASURED, UNSTABLE,
        MAX_PERIOD_SPREAD, MAX_REGIME_SPREAD)

    mx_p = OUT / "matrix.json"
    mx = json.loads(mx_p.read_text(encoding="utf-8"))
    sayac, degisen = {}, 0
    for kart in mx.get("cards", []):
        for h in kart.get("horizons", []):
            c = h.get("convergence")
            if not c:
                continue
            k = c.get("checks") or {}
            ps, rs = c.get("period_spread"), c.get("regime_spread")
            k["temporally_stable"] = (None if ps is None
                                      else bool(ps <= MAX_PERIOD_SPREAD))
            k["regime_stable"] = (None if rs is None
                                  else bool(rs <= MAX_REGIME_SPREAD))
            if k["temporally_stable"] is False:
                v = UNSTABLE
            elif k["regime_stable"] is False:
                v = REGIME_DEPENDENT
            elif (k["temporally_stable"] is None
                  and k["regime_stable"] is None):
                v = UNMEASURED
            elif k.get("sample_sufficient") and k.get("shrinks_with_n") is not False:
                v = CONVERGED
            else:
                v = CONVERGING
            if v != c.get("verdict"):
                degisen += 1
            c["verdict"], c["checks"] = v, k
            # gerekçe metnini de güncelle
            if k["regime_stable"] is False:
                c["reasons"] = [
                    (f"rejimler arası fark {rs*100:.1f} puan — ölçüm YANLIŞ "
                     f"değil, büyüklük rejime KOŞULLU; satırı rejim kırılımıyla "
                     f"okuyun") if "rejim" in x else x
                    for x in (c.get("reasons") or [])]
            sayac[v] = sayac.get(v, 0) + 1
    mx.setdefault("summary", {})["by_convergence"] = sayac
    mx_p.write_text(json.dumps(mx, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"yakınsama verdiktleri yeniden hesaplandı: {degisen} değişti → {sayac}")

    # ── PBO'yu GÜNCEL mantıkla yeniden hesapla ──────────────────────────
    # PBO yalnız hücre verisinden (dönem × rejim × stop adayı) hesaplanır —
    # model gerektirmez, bu yüzden tam koşu tekrarlanmadan düzeltilebilir.
    # Eski değerler 594/594 hücrede tam 1,0'dı ("ölçülemedi" 1,0 sayılıyordu).
    import pandas as pd
    from agi_trader.qualification.research import _pbo_from_stop_grid
    from agi_trader.qualification.targets import CostProfile

    cells_p = OUT / "cells.parquet"
    if cells_p.exists():
        cells = pd.read_parquet(cells_p)
        profiller = json.loads((OUT / "cost_profiles.json")
                               .read_text(encoding="utf-8"))
        onbellek = {}
        degisti = 0
        for kart in mx.get("cards", []):
            sym = kart["symbol"]
            pd_ = profiller.get(sym)
            if not pd_:
                continue
            prof = CostProfile(**{k: v for k, v in pd_.items()
                                  if k in CostProfile.__dataclass_fields__})
            for h in kart.get("horizons", []):
                anahtar = (sym, h["horizon"], h["direction"])
                if anahtar not in onbellek:
                    onbellek[anahtar] = _pbo_from_stop_grid(
                        cells, sym, h["horizon"], h["direction"], prof)
                yeni = onbellek[anahtar]
                if yeni != h.get("pbo"):
                    degisti += 1
                h["pbo"] = yeni
        olculen = [v for v in onbellek.values() if v is not None]
        print(f"PBO yeniden hesaplandı: {degisti} satır değişti · "
              f"{len(olculen)}/{len(onbellek)} hücre ölçülebildi"
              + (f" · aralık {min(olculen):.2f}–{max(olculen):.2f}"
                 if olculen else ""))
        mx_p.write_text(json.dumps(mx, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    # ── validation_report'a kimliği göm ─────────────────────────────────
    vr["provenance"] = d
    vr_p.write_text(json.dumps(vr, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("validation_report.json güncellendi")
    return 0


if __name__ == "__main__":
    sys.exit(main())
