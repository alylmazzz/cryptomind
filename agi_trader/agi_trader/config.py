"""
Konfigürasyon yükleyici.

Öncelik sırası:
  1. config.yaml (varsa) — opsiyonel, PyYAML yoksa atlanır
  2. Ortam değişkenleri (.env veya gerçek env) — API anahtarları için
  3. Koddaki güvenli varsayılanlar (DEFAULTS)

API anahtarları ASLA koda gömülmez; yalnızca ortamdan okunur.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------------------------------------------------------
# Güvenli varsayılanlar
# -----------------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    # Hangi borsalardan veri çekilecek (ccxt id'leri)
    "exchanges": ["binance", "bybit", "okx", "kraken", "kucoin", "gateio", "bitfinex"],
    # Analiz edilecek pariteler
    "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
    # Çok zaman dilimli analiz (spec'teki tüm TF'ler desteklenir)
    "timeframes": ["15m", "1h", "4h", "1d", "1w", "1M"],
    # Karar verirken ana (tetikleyici) zaman dilimi
    "primary_timeframe": "4h",
    "ohlcv_limit": 400,

    # Veri kaynağı: "auto" (ccxt varsa canlı, yoksa sentetik), "live", "synthetic"
    "data_source": "auto",

    # Karar motoru
    "decision": {
        # KONSENSÜS KAPISI (eski tek-sayı %90 kapısı ulaşılamazdı → 0 işlem).
        # "Her şey aynı sinyali gösteriyorsa gir" mantığı: aktif katmanların
        # ağırlıklı en az %70'i aynı yönde olmalı + asgari güven + R/R.
        "min_confidence": 0.74,          # nihai güven alt sınırı (seçici konsensüs)
        "min_risk_reward": 0.8,          # hibritte scalp(rr~1.0)+atr(rr~1.5) ikisi geçsin
        "consensus_min_agreement": 0.82, # aktif katmanların ağırlıklı yön uyumu ≥ %82
        "consensus_min_layers": 3,       # en az 3 aktif (kararlı) katman olmalı
        "news_veto": True,               # haber ters yöndeyse işlemi engelle
        "news_veto_min_conf": 0.45,      # haber vetosu için asgari haber güveni
        "trend_gate_min_adx": 20.0,      # ADX<20 (choppy) → işlem yok (whipsaw filtresi)
        # Her analiz katmanının başlangıç ağırlığı (dinamik güncellenir)
        "layer_weights": {
            "technical": 0.18,
            "pattern": 0.10,
            "trendline": 0.12,
            "smc": 0.09,
            "multi_timeframe": 0.12,
            "ai_ensemble": 0.13,
            "sentiment": 0.08,
            "fear_greed": 0.04,
            "news": 0.05,
            "onchain": 0.06,
            "macro": 0.03,
        },
    },

    # Risk yönetimi
    "risk": {
        "portfolio_value": 10_000.0,
        "max_portfolio_risk": 0.02,
        "max_position_risk": 0.01,
        "max_drawdown": 0.20,
        "kelly_fraction_cap": 0.20,
        "atr_stop_mult": 2.5,
        "tp_r_multiples": [1.5, 2.5, 4.0],   # trend modunda: kazananı koştur (1.5/2.5/4R)
        # HEDEF MODU: "scalp" (sabit %) | "atr" (ATR-R) | "hybrid" (ADX'e göre seç)
        "tp_mode": "hybrid",
        "hybrid_adx_trend": 25.0,              # ADX>=bu → ATR-R; altı → scalp
        "scalp_tp_pct": [0.01, 0.02, 0.03],    # range scalp: geniş %1/2/3 (ücret sürtünmesi için)
        "scalp_sl_pct": 0.01,
        # FAZ4 — gerçekçi yürütme maliyeti (backtest)
        "fee_taker": 0.0004,                   # %0.04 taker (futures, likit majör)
        "slippage_base": 0.00015,              # 1.5 bps taban kayma (her yön)
        "slippage_vol_k": 0.03,                # kayma += k × (ATR/fiyat); ~%0.2 round-trip toplam
        # FAZ3 — risk-temelli pozisyon boyutlama
        "sizing_mode": "fixed_fraction",       # "fixed_fraction" | "full"
        "risk_per_trade": 0.01,                # işlem başına equity'nin %1'i riske edilir
        "max_leverage": 1.0,                   # spot: pozisyon equity'yi aşamaz
        "dd_risk_scaling": True,               # drawdown'da riski azalt
        "loss_cooldown_n": 3,                  # N ardışık kayıptan sonra boyutu yarıla
    },

    # Execution — GÜVENLİK: varsayılan paper, canlı için çift onay şart
    "execution": {
        "mode": "paper",          # "paper" | "live"
        "allow_live": False,      # mode=live olsa bile bu False ise canlıya geçilmez
        "testnet": True,          # GÜVENLİK: canlı modda bile varsayılan Binance testnet (sahte para)
        "live_max_order_usdt": 100.0,   # canlı emir başına üst sınır
        "max_open_positions": 5,
        "kill_switch_drawdown": 0.15,   # bu DD'ye ulaşılırsa tüm sistem durur
    },

    # Sentiment / Twitter
    "sentiment": {
        "enabled": True,          # API yoksa otomatik nötr fallback
        "min_impact_for_signal": 70.0,
    },

    # Çıktı / loglama
    "output_dir": "runs",
    "log_level": "INFO",
}

# Ortamdan okunacak anahtarlar — TEK KAYNAK: providers.py kataloğu.
# Yeni sağlayıcı eklemek için providers.PROVIDERS'a ekle; burası otomatik güncellenir.
from .providers import env_keys as _provider_env_keys

ENV_KEYS = _provider_env_keys()


@dataclass
class Config:
    data: Dict[str, Any] = field(default_factory=lambda: _deep_copy(DEFAULTS))
    secrets: Dict[str, str] = field(default_factory=dict)

    # -- erişim kısayolları -------------------------------------------------
    def get(self, path: str, default: Any = None) -> Any:
        """Noktayla ayrılmış yol ile değer al: cfg.get('risk.max_drawdown')."""
        node: Any = self.data
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def secret(self, name: str) -> Optional[str]:
        return self.secrets.get(name) or os.environ.get(name)

    def has_secret(self, name: str) -> bool:
        return bool(self.secret(name))

    def save_secrets(self, updates: Dict[str, str], env_path: str = ".env") -> None:
        """Verilen anahtarları belleğe + ortam değişkenlerine + .env dosyasına yaz.
        Boş değer ('') ilgili anahtarı temizler. Yalnızca yerel makineye yazılır."""
        # bellek + ortam
        for k, v in updates.items():
            if v is None:
                continue
            if v == "":
                self.secrets.pop(k, None)
                os.environ.pop(k, None)
            else:
                self.secrets[k] = v
                os.environ[k] = v

        # .env'i oku-birleştir-yaz (mevcut key=value'ları korur)
        path = Path(env_path)
        existing: Dict[str, str] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s and not s.startswith("#") and "=" in s:
                    k, _, val = s.partition("=")
                    existing[k.strip()] = val
        for k, v in updates.items():
            if v == "" or v is None:
                existing.pop(k, None)
            else:
                existing[k] = v
        lines = ["# AGI Trader — kayitli kimlik bilgileri (panelden olusturuldu)"]
        lines += [f"{k}={v}" for k, v in existing.items()]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @property
    def symbols(self) -> List[str]:
        return list(self.data["symbols"])

    @property
    def timeframes(self) -> List[str]:
        return list(self.data["timeframes"])


def _deep_copy(d: Any) -> Any:
    import copy
    return copy.deepcopy(d)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_dotenv(path: Path) -> Dict[str, str]:
    secrets: Dict[str, str] = {}
    if not path.exists():
        return secrets
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        secrets[key.strip()] = val.strip().strip('"').strip("'")
    return secrets


def load_config(config_path: Optional[str] = None, env_path: Optional[str] = None) -> Config:
    """Konfigürasyonu yükle (yaml + .env + ortam)."""
    data = _deep_copy(DEFAULTS)

    # 1) config.yaml (opsiyonel)
    candidate = Path(config_path) if config_path else Path("config.yaml")
    if candidate.exists():
        try:
            import yaml  # type: ignore
            user_cfg = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            data = _deep_merge(data, user_cfg)
        except Exception:
            # PyYAML yoksa veya parse hatası varsa varsayılanlarla devam
            pass

    # 2) .env + gerçek ortam
    secrets: Dict[str, str] = {}
    dotenv = Path(env_path) if env_path else Path(".env")
    secrets.update(_load_dotenv(dotenv))
    for key in ENV_KEYS:
        if key in os.environ:
            secrets[key] = os.environ[key]

    return Config(data=data, secrets=secrets)
