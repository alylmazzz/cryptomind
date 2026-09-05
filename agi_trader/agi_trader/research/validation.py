"""
İstatistiksel doğrulama araçları — "bu edge gerçek mi?" sorusunun cevabı.

Kaynak: Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014);
López de Prado, "Advances in Financial Machine Learning" (2018) bölüm 7 ve 11.

NEDEN GEREKLİ: Yeterince çok strateji denenirse, hiçbir gerçek edge olmasa bile
en iyisi yüksek Sharpe gösterir. Bu, seçim yanlılığıdır (selection bias under
multiple testing). Bu projede tam olarak bu yaşandı: konfig Dec2025-Jun2026'ya
kalibre edilince +%61,8 göründü, örneklem dışı yıllarda her yıl zarar çıktı.

Bu modül üç ayrı savunma sunar:
  1. PSR / DSR   — gözlenen Sharpe, KAÇ DENEME yapıldığı bilgisiyle düzeltilir
  2. Purged CV   — etiket örtüşmesi ve seri korelasyon kaynaklı sızıntı temizlenir
  3. PBO         — "seçtiğim en iyi konfig örneklem dışında medyanın altına düşer mi?"

Bağımlılık: yalnız numpy (scipy YOK — sunucuda kurulu değil).
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

EULER_GAMMA = 0.5772156649015329

# Kabul kapısı eşikleri — plan tarafından sabitlendi, keyfi gevşetilmemeli.
GATE_MIN_DSR = 0.95
GATE_MAX_PBO = 0.30
GATE_MAX_CORR = 0.40
GATE_SUSPECT_SHARPE = 2.5      # üstü: önce bug varsayılır


# ===========================================================================
# Normal dağılım yardımcıları (scipy'siz)
# ===========================================================================
def norm_cdf(x: float) -> float:
    """Standart normal birikimli dağılım."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Standart normal ters CDF (Acklam rasyonel yaklaşımı, |hata| < 1.15e-9).

    scipy.stats.norm.ppf yerine kullanılır — sunucuda scipy yok."""
    if not 0.0 < p < 1.0:
        if p <= 0.0:
            return -math.inf
        if p >= 1.0:
            return math.inf
        raise ValueError("p (0,1) aralığında olmalı")

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    # bir adım Halley iyileştirmesi
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


# ===========================================================================
# Temel Sharpe istatistikleri
# ===========================================================================
def sharpe(returns: Sequence[float], periods_per_year: Optional[float] = None) -> float:
    """Getiri serisinin Sharpe'ı. periods_per_year verilirse yıllıklandırılır."""
    a = np.asarray(returns, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return 0.0
    sd = a.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    sr = float(a.mean() / sd)
    return sr * math.sqrt(periods_per_year) if periods_per_year else sr


def _moments(a: np.ndarray) -> Tuple[float, float]:
    """(çarpıklık, ham basıklık). Normal dağılımda (0, 3)."""
    m = a - a.mean()
    s = a.std(ddof=0)
    if s < 1e-12:
        return 0.0, 3.0
    return float((m ** 3).mean() / s ** 3), float((m ** 4).mean() / s ** 4)


def psr(returns: Sequence[float], sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio — gözlenen Sharpe'ın benchmark'ı gerçekten
    aşma olasılığı; çarpıklık/basıklık ve örnek sayısıyla düzeltilir.

    Yorum: 0,95 = "%95 olasılıkla gerçek Sharpe > benchmark". 0,90 altı zayıf.
    NOT: `returns` ile `sr_benchmark` AYNI periyotta olmalı (ikisi de dönem-başı
    veya ikisi de yıllık). Karıştırmak sessizce yanlış sonuç verir."""
    a = np.asarray(returns, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 5 or a.std(ddof=1) < 1e-12:
        return 0.0
    sr = float(a.mean() / a.std(ddof=1))
    skew, kurt = _moments(a)
    denom = math.sqrt(max(1e-9, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return float(norm_cdf(z))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """N bağımsız denemenin EN İYİSİNİN, gerçek edge SIFIRKEN bile beklenen
    Sharpe'ı (Bailey & López de Prado, Gumbel yaklaşımı).

    sr_std: denenen stratejilerin Sharpe'larının standart sapması.
    Bu, DSR'ın karşılaştırma eşiğidir — "şansla bu kadarı zaten çıkardı"."""
    n = max(2, int(n_trials))
    z1 = norm_ppf(1 - 1.0 / n)
    z2 = norm_ppf(1 - 1.0 / (n * math.e))
    return float(sr_std * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns: Sequence[float], n_trials: int,
                    sr_std: Optional[float] = None,
                    trial_sharpes: Optional[Sequence[float]] = None,
                    periods_per_year: float = 365.0) -> Dict[str, float]:
    """Deflated Sharpe Ratio — PSR, ama benchmark olarak 0 yerine
    "N deneme sonunda şansla beklenen en iyi Sharpe" kullanılır.

    n_trials: bu sonuca ulaşana kadar denenen strateji/konfig sayısı.
              Dürüst sayı `trial_count()` ile takip edilir; küçük göstermek
              DSR'ı sahte şekilde yükseltir.
    sr_std / trial_sharpes: denemelerin Sharpe dağılımının yayılımı,
              **YILLIK** cinsten (loglanan Sharpe'lar yıllıktır). Verilmezse
              muhafazakâr varsayılan 1.0 kullanılır.
    periods_per_year: `returns` serisinin periyodu (günlük=365, saatlik=8760).

    BİRİM UYUMU (kritik): PSR dönem-başı Sharpe ile çalışır, kullanıcı ise
    yıllık düşünür. sr0 yıllık hesaplanıp √periods_per_year'a bölünerek
    dönem-başına çevrilir. Bu dönüşüm atlanırsa DSR her zaman 0 çıkar."""
    a = np.asarray(returns, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 5:
        return {"dsr": 0.0, "sr_annual": 0.0, "sr0_annual": 0.0,
                "n_trials": int(n_trials), "n_obs": int(len(a)),
                "verdict": "yetersiz örnek"}

    if sr_std is None:
        if trial_sharpes is not None and len(trial_sharpes) > 1:
            sr_std = float(np.std(np.asarray(trial_sharpes, dtype=float), ddof=1))
        else:
            sr_std = 1.0                      # muhafazakâr: yayılım bilinmiyor
    sr_std = max(float(sr_std), 1e-6)

    ppy = max(1.0, float(periods_per_year))
    sr0_annual = expected_max_sharpe(n_trials, sr_std)
    sr0_period = sr0_annual / math.sqrt(ppy)          # ← birim uyumu
    d = psr(a, sr_benchmark=sr0_period)

    sr_period = float(a.mean() / (a.std(ddof=1) + 1e-12))
    return {"dsr": round(d, 4),
            "sr_annual": round(sr_period * math.sqrt(ppy), 4),
            "sr0_annual": round(sr0_annual, 4),
            "sr_period": round(sr_period, 6),
            "sr_std_annual": round(sr_std, 4),
            "n_trials": int(n_trials), "n_obs": int(len(a)),
            "periods_per_year": ppy,
            "verdict": "GEÇTİ" if d >= GATE_MIN_DSR else "KALDI"}


def min_backtest_length(target_sharpe: float, n_trials: int,
                        sr_std: float = 1.0) -> float:
    """Bu Sharpe'ı N deneme arasından seçerken yanılmamak için gereken
    ASGARİ backtest uzunluğu (yıl). Gözlenen veri bundan kısaysa sonuç
    istatistiksel olarak anlamsızdır.

    Formül: MinBTL ≈ (E[max SR] / SR)² yıl (yıllık Sharpe cinsinden)."""
    if target_sharpe <= 0:
        return math.inf
    sr0 = expected_max_sharpe(n_trials, sr_std)
    return float((sr0 / target_sharpe) ** 2)


# ===========================================================================
# Purged / embargolu çapraz doğrulama (AFML bölüm 7)
# ===========================================================================
def purged_kfold_splits(t1: "np.ndarray | Sequence", n_splits: int = 5,
                        embargo_pct: float = 0.01) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Etiket örtüşmesini temizleyen K-kat bölme.

    Finansal etiketler örtüşür (bir barın etiketi sonraki N barı kullanır).
    Sıradan KFold'da eğitim kümesi test etiketinin İÇİNİ görür → sızıntı →
    sahte yüksek skor. Burada:
      • purge : test aralığıyla ÖRTÜŞEN eğitim örnekleri atılır
      • embargo: testten hemen SONRAKİ %embargo_pct örnek de atılır
                 (seri korelasyon nedeniyle)

    t1: her örneğin ETİKETİNİN BİTTİĞİ indeks (int konumu) — i. örneğin
        etiketi t1[i] barında sonuçlanır. Örn. 10 barlık üçlü bariyerde
        t1[i] = i + 10.

    Döndürür: (train_idx, test_idx) çiftleri."""
    t1 = np.asarray(t1)
    n = len(t1)
    if n < n_splits * 2:
        raise ValueError(f"örnek sayısı ({n}) kat sayısı için yetersiz")

    embargo = int(n * float(embargo_pct))
    bounds = [(int(b[0]), int(b[-1]) + 1)
              for b in np.array_split(np.arange(n), n_splits)]

    for start, stop in bounds:
        test_idx = np.arange(start, stop)
        # test aralığının kapladığı zaman: [start, max(t1[test]))
        test_end = int(max(stop, t1[test_idx].max() + 1))

        train_mask = np.ones(n, dtype=bool)
        train_mask[start:stop] = False
        # PURGE: etiketi test başlangıcından sonra biten önceki örnekler
        overlap_before = (np.arange(n) < start) & (t1 >= start)
        train_mask[overlap_before] = False
        # EMBARGO: testten sonraki ilk `embargo` örnek
        emb_stop = min(n, test_end + embargo)
        train_mask[stop:emb_stop] = False

        train_idx = np.flatnonzero(train_mask)
        if len(train_idx) == 0:
            continue
        yield train_idx, test_idx


def combinatorial_purged_splits(t1: "np.ndarray | Sequence", n_groups: int = 6,
                                n_test_groups: int = 2,
                                embargo_pct: float = 0.01
                                ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Kombinatoryal Purged CV (CPCV, AFML 12.4).

    Tek bir örneklem-dışı yol yerine C(n_groups, n_test_groups) farklı yol
    üretir → TEK BİR SAYI değil, Sharpe DAĞILIMI verir. "Şanslı pencere"
    sorusunu ortadan kaldırır (bu projede tek-pencere BTC +%14,4'ün şans
    olduğu ancak çok-pencere testiyle anlaşılmıştı)."""
    from itertools import combinations

    t1 = np.asarray(t1)
    n = len(t1)
    embargo = int(n * float(embargo_pct))
    groups = [np.array(g, dtype=int) for g in np.array_split(np.arange(n), n_groups)]

    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for combo in combinations(range(n_groups), n_test_groups):
        test_idx = np.sort(np.concatenate([groups[g] for g in combo]))
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        for g in combo:
            start, stop = int(groups[g][0]), int(groups[g][-1]) + 1
            test_end = int(max(stop, t1[groups[g]].max() + 1))
            train_mask[(np.arange(n) < start) & (t1 >= start)] = False
            train_mask[stop:min(n, test_end + embargo)] = False
        train_idx = np.flatnonzero(train_mask)
        if len(train_idx) and len(test_idx):
            out.append((train_idx, test_idx))
    return out


# ===========================================================================
# PBO — Aşırı uyum olasılığı (CSCV, Bailey ve ark. 2015)
# ===========================================================================
def pbo(perf_matrix: "np.ndarray", n_splits: int = 8) -> Dict[str, float]:
    """Probability of Backtest Overfitting (Combinatorially Symmetric CV).

    perf_matrix: (T × N) — T dönem × N aday konfig getiri matrisi.
    Yöntem: dönemleri S bloğa böl, her yarı-yarıya kombinasyonda
    örneklem-içinde EN İYİ konfigi seç, onun örneklem-DIŞI sıralamasına bak.
    En iyi konfig OOS'ta sürekli medyanın altındaysa seçim süreci overfit'tir.

    PBO > 0,30 → strateji seçimi güvenilmez (kapı: <0,30)."""
    from itertools import combinations

    M = np.asarray(perf_matrix, dtype=float)
    # ⚠️ ÖLÇÜLEMEDİ ≠ KESİN AŞIRI UYUM — ölçülerek bulunan hata.
    # Eskiden yetersiz veride 1.0 dönüyordu; çağıran bunu "PBO = 1,0, seçim
    # tamamen aşırı uyum" diye okuyordu. 594/594 hücrede tam 1,0 çıkması
    # bundandı: 12 dilimlik matris `n_splits=8` ile istendiğinde
    # `T < n_splits*2` dalına düşüyordu. Artık `None` döner ve kapı bunu
    # "ölçülmedi" olarak ele alır.
    if M.ndim != 2 or M.shape[1] < 2:
        return {"pbo": None, "n_combinations": 0, "verdict": "ölçülemedi: yetersiz aday"}

    T, N = M.shape
    n_splits = max(4, n_splits - (n_splits % 2))          # çift olmalı
    if T < n_splits * 2:
        # Dilim sayısını veriye UYDUR; yine de yetmezse ölçme.
        n_splits = max(4, (T // 2) - ((T // 2) % 2))
        if T < n_splits * 2 or n_splits < 4:
            return {"pbo": None, "n_combinations": 0,
                    "verdict": f"ölçülemedi: {T} dilim CSCV için yetersiz"}

    blocks = np.array_split(np.arange(T), n_splits)
    half = n_splits // 2
    logits: List[float] = []

    for combo in combinations(range(n_splits), half):
        is_rows = np.concatenate([blocks[b] for b in combo])
        oos_rows = np.concatenate([blocks[b] for b in range(n_splits) if b not in combo])

        is_perf = np.array([sharpe(M[is_rows, j]) for j in range(N)])
        oos_perf = np.array([sharpe(M[oos_rows, j]) for j in range(N)])

        best = int(np.argmax(is_perf))
        # seçilen konfigin OOS göreli sırası (0=en kötü, 1=en iyi)
        rank = float((oos_perf < oos_perf[best]).sum()) / max(1, N - 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1 - rank)))

    arr = np.array(logits)
    p = float((arr <= 0).mean())                          # OOS medyanın altına düşme oranı
    return {"pbo": round(p, 4), "n_combinations": len(logits),
            "n_splits": n_splits, "n_periods": T, "n_candidates": N,
            "median_logit": round(float(np.median(arr)), 4),
            "verdict": "GEÇTİ" if p < GATE_MAX_PBO else "KALDI"}


# ===========================================================================
# Deneme günlüğü — DSR'ın ihtiyaç duyduğu N
# ===========================================================================
def _trials_path(output_dir: str = "runs") -> Path:
    p = Path(output_dir)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p
    p.mkdir(parents=True, exist_ok=True)
    return p / "trials.jsonl"


def _signature(name: str, params: Dict) -> str:
    """Bir denemenin kimliği = ad + parametreler. Aynı imza = aynı hipotez."""
    return name + "|" + json.dumps(params, sort_keys=True, default=str)


def trial_log(name: str, params: Dict, result: Dict,
              output_dir: str = "runs", superseded: bool = False,
              note: str = "") -> int:
    """Her backtest denemesini kaydeder. DSR bu sayıyı kullanır.

    ZORUNLU: parametre taraması yapan HER betik her denemeyi buraya yazmalı.
    Yazılmayan deneme = DSR'ın olduğundan yüksek çıkması = sahte güven.

    superseded=True: uygulama hatası içerdiği tespit edilmiş koşu. Kayıt SİLİNMEZ
    (denetlenebilirlik), ama istatistiklerden dışlanır — hatalı bir uygulamanın
    Sharpe'ını "denenen hipotez dağılımına" saymak sr_std'yi yapay şişirir."""
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "name": name, "params": params, "result": result,
           "sig": _signature(name, params)}
    if superseded:
        rec["superseded"] = True
    if note:
        rec["note"] = note
    path = _trials_path(output_dir)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return trial_count(name, output_dir)


def _read_trials(name: Optional[str], output_dir: str) -> Dict[str, Dict]:
    """FARKLI imzaların en son kaydı. Aynı betiği iki kez çalıştırmak yeni bir
    hipotez denemek değildir; DSR ayrı hipotez sayısını ister."""
    path = _trials_path(output_dir)
    if not path.exists():
        return {}
    out: Dict[str, Dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if name is not None and r.get("name") != name:
                continue
            if r.get("superseded"):
                continue
            out[r.get("sig") or _signature(r.get("name", "?"), r.get("params", {}))] = r
    return out


def trial_count(name: Optional[str] = None, output_dir: str = "runs") -> int:
    """Denenen FARKLI hipotez sayısı (tekrar koşumlar sayılmaz)."""
    return len(_read_trials(name, output_dir))


def trial_sharpes(name: Optional[str] = None, output_dir: str = "runs") -> List[float]:
    """Farklı hipotezlerin Sharpe'ları — DSR'ın sr_std'si için."""
    out: List[float] = []
    for r in _read_trials(name, output_dir).values():
        s = (r.get("result") or {}).get("sharpe")
        if isinstance(s, (int, float)) and math.isfinite(s):
            out.append(float(s))
    return out


# ===========================================================================
# Kabul kapısı
# ===========================================================================
@dataclass
class AcceptanceResult:
    passed: bool
    name: str
    checks: Dict[str, Dict] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    def __str__(self) -> str:
        head = f"{'✅ KABUL' if self.passed else '❌ RET'} — {self.name}"
        return head + "\n" + "\n".join("  " + r for r in self.reasons)


def acceptance_gate(name: str,
                    returns: Sequence[float],
                    n_trials: Optional[int] = None,
                    book_returns: Optional[Sequence[float]] = None,
                    perf_matrix: Optional["np.ndarray"] = None,
                    locked_test_returns: Optional[Sequence[float]] = None,
                    baseline_test_returns: Optional[Sequence[float]] = None,
                    periods_per_year: float = 365.0,
                    output_dir: str = "runs") -> AcceptanceResult:
    """Bir sleeve/özellik canlıya girebilir mi? Plandaki BEŞ kapının tamamı.

    returns              : aday stratejinin dönem getirileri (train+val)
    n_trials             : denenen konfig sayısı (None → trials.jsonl'dan okunur)
    book_returns         : mevcut kitabın getirileri → korelasyon kapısı
    perf_matrix          : (T × N) aday konfig matrisi → PBO kapısı
    locked_test_returns  : KİLİTLİ test dönemindeki getiriler
    baseline_test_returns: aynı dönemde mevcut baseline getirileri
    """
    res = AcceptanceResult(passed=True, name=name)
    a = np.asarray(returns, dtype=float)
    a = a[np.isfinite(a)]

    # ---- 0) örnek yeterliliği
    if len(a) < 30:
        res.passed = False
        res.reasons.append(f"❌ Örnek yetersiz ({len(a)} < 30)")
        return res

    sr_ann = sharpe(a, periods_per_year)
    res.checks["sharpe"] = {"value": round(sr_ann, 3), "periods": len(a)}

    # ---- 1) şüphe eşiği: çok yüksek Sharpe = önce bug varsayılır
    if sr_ann > GATE_SUSPECT_SHARPE:
        res.passed = False
        res.reasons.append(
            f"❌ Sharpe {sr_ann:.2f} > {GATE_SUSPECT_SHARPE} — ÖNCE BUG VARSAYILIR "
            f"(look-ahead / .shift(1) / sızıntı kontrolü yapılmadan kabul edilemez)")

    # ---- 2) Deflated Sharpe
    if n_trials is None:
        n_trials = max(1, trial_count(name, output_dir))
    hist = trial_sharpes(name, output_dir)
    d = deflated_sharpe(a, n_trials=n_trials,
                        trial_sharpes=hist if len(hist) > 1 else None,
                        periods_per_year=periods_per_year)
    res.checks["dsr"] = d
    if d["dsr"] < GATE_MIN_DSR:
        res.passed = False
        res.reasons.append(
            f"❌ DSR {d['dsr']:.3f} < {GATE_MIN_DSR} (yıllık SR {d['sr_annual']:.2f} vs "
            f"şansla beklenen {d['sr0_annual']:.2f}, {n_trials} deneme)")
    else:
        res.reasons.append(f"✅ DSR {d['dsr']:.3f} (yıllık SR {d['sr_annual']:.2f} > "
                           f"şans {d['sr0_annual']:.2f})")

    # ---- 3) PBO
    if perf_matrix is not None:
        p = pbo(perf_matrix)
        res.checks["pbo"] = p
        if p["pbo"] >= GATE_MAX_PBO:
            res.passed = False
            res.reasons.append(f"❌ PBO {p['pbo']:.3f} ≥ {GATE_MAX_PBO} — seçim süreci overfit")
        else:
            res.reasons.append(f"✅ PBO {p['pbo']:.3f}")
    else:
        res.reasons.append("⚠️ PBO ölçülmedi (perf_matrix verilmedi)")

    # ---- 4) mevcut kitapla korelasyon
    if book_returns is not None:
        b = np.asarray(book_returns, dtype=float)
        n = min(len(a), len(b))
        if n >= 30:
            c = float(np.corrcoef(a[-n:], b[-n:])[0, 1])
            res.checks["corr_to_book"] = {"value": round(c, 3), "n": n}
            if abs(c) > GATE_MAX_CORR:
                res.passed = False
                res.reasons.append(
                    f"❌ Kitapla korelasyon {c:+.2f} > ±{GATE_MAX_CORR} — "
                    f"yeni getiri akışı değil, mevcut bahsin tekrarı")
            else:
                res.reasons.append(f"✅ Kitapla korelasyon {c:+.2f} (bağımsız akış)")

    # ---- 5) kilitli testte baseline'ı geçme
    if locked_test_returns is not None and baseline_test_returns is not None:
        s_new = sharpe(locked_test_returns, periods_per_year)
        s_base = sharpe(baseline_test_returns, periods_per_year)
        res.checks["locked_test"] = {"candidate": round(s_new, 3),
                                     "baseline": round(s_base, 3)}
        if s_new <= s_base:
            res.passed = False
            res.reasons.append(
                f"❌ Kilitli testte baseline geçilemedi ({s_new:.2f} ≤ {s_base:.2f})")
        else:
            res.reasons.append(f"✅ Kilitli test {s_new:.2f} > baseline {s_base:.2f}")
    else:
        res.reasons.append("⚠️ Kilitli test çalıştırılmadı — canlıya alınamaz")
        res.passed = False

    return res


# ===========================================================================
# Sızıntı testi
# ===========================================================================
def shuffle_test(strategy_fn, data, n_shuffles: int = 20,
                 rng_seed: int = 0) -> Dict[str, float]:
    """Etiket karıştırma testi: getiri sırası bozulduğunda strateji Sharpe'ı
    0'a düşmeli. Düşmüyorsa hatta yüksek kalıyorsa SIZINTI vardır.

    strategy_fn(data) -> getiri dizisi döndürmeli."""
    rng = np.random.default_rng(rng_seed)
    real = sharpe(strategy_fn(data))
    shuffled: List[float] = []
    for _ in range(n_shuffles):
        d = data.copy()
        idx = rng.permutation(len(d))
        d = d.iloc[idx].reset_index(drop=True) if hasattr(d, "iloc") else np.asarray(d)[idx]
        try:
            shuffled.append(sharpe(strategy_fn(d)))
        except Exception:
            continue
    if not shuffled:
        return {"real": real, "shuffled_mean": float("nan"), "leak": True}
    m = float(np.mean(shuffled))
    s = float(np.std(shuffled) + 1e-9)
    z = (real - m) / s
    return {"real": round(real, 3), "shuffled_mean": round(m, 3),
            "shuffled_std": round(s, 3), "z": round(z, 2),
            "leak": bool(abs(m) > 0.5),
            "verdict": ("SIZINTI ŞÜPHESİ" if abs(m) > 0.5 else
                        "temiz" if z > 2 else "gerçek edge yok (z<2)")}
