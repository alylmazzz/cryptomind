"""
RL v2 — PPO Trading Agent (stable-baselines3).

Eski tek-bar fitted-Q yerine GERÇEK sıralı RL: PPO ajanı TradingEnv'de pozisyon
yönetmeyi (gir, tut, çık, tersine dön) öğrenir. Model symbol+timeframe+veri-imzası
başına diske önbelleklenir (runs/models/ppo_*.zip); imza değişmedikçe yeniden
eğitmez. Çoklu-varlık: fit() birden çok df alıp tek ajanı ortak eğitebilir.

predict() son barın gözlemi için politika dağılımından kalibre p(up) ve aksiyon
(long/short/flat) döndürür → ensemble'a bir backend olarak girer.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.light import LIGHT_MODE

if LIGHT_MODE:
    # hafif mod: stable_baselines3/torch import edilmez (bellek koruması)
    _HAS_RL = False
else:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        import torch
        from .rl_env import TradingEnv, _HAS_GYM
        _HAS_RL = _HAS_GYM
    except Exception:  # pragma: no cover
        _HAS_RL = False

_WARMUP = 50
_MEM_CACHE: Dict[str, "RLAgent"] = {}


def _episode_from_df(df: pd.DataFrame):
    """df -> (features (T,F), returns (T,)) — ısınma barları atılır."""
    from .ensemble import _features
    f = _features(df).values.astype(np.float32)
    close = df["close"].values.astype(np.float32)
    ret = np.zeros(len(close), dtype=np.float32)
    ret[:-1] = close[1:] / (close[:-1] + 1e-9) - 1
    if len(close) <= _WARMUP + 20:
        return None
    return f[_WARMUP:-1], ret[_WARMUP:-1]


class RLAgent:
    def __init__(self, name: str, timesteps: int = 12000):
        self.name = name
        self.timesteps = timesteps
        self.model = None
        self.meta: Dict = {}

    def _path(self) -> Path:
        d = Path("runs") / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"ppo_{self.name}.zip"

    def fit_or_load(self, dfs: List[pd.DataFrame], force: bool = False,
                    max_age_h: float = 24.0) -> Dict:
        if not _HAS_RL:
            self.meta = {"available": False}
            return self.meta
        episodes = [e for e in (_episode_from_df(df) for df in dfs) if e is not None]
        if not episodes:
            self.meta = {"available": False, "reason": "yetersiz veri"}
            return self.meta
        path = self._path()
        # disk önbelleği 24 saatten taze ise yükle (her barda yeniden eğitme)
        if path.exists() and not force:
            age_h = (time.time() - path.stat().st_mtime) / 3600.0
            if age_h < max_age_h:
                try:
                    self.model = PPO.load(str(path), device="cpu")
                    self.meta = {"available": True, "from_cache": True,
                                 "episodes": len(episodes), "age_h": round(age_h, 1)}
                    return self.meta
                except Exception:
                    pass
        try:
            env = DummyVecEnv([lambda: TradingEnv(episodes)])
            self.model = PPO("MlpPolicy", env, verbose=0, device="cpu",
                             n_steps=512, batch_size=128, gae_lambda=0.95, gamma=0.99,
                             ent_coef=0.01, learning_rate=3e-4)
            self.model.learn(total_timesteps=self.timesteps, progress_bar=False)
            self.model.save(str(path))
            self.meta = {"available": True, "trained": True, "episodes": len(episodes),
                         "timesteps": self.timesteps}
        except Exception as e:
            self.meta = {"available": False, "error": f"{type(e).__name__}: {e}"}
        return self.meta

    def predict(self, df: pd.DataFrame) -> Dict:
        """Son bar için kalibre p(up) + aksiyon (long/short/flat)."""
        if not _HAS_RL or self.model is None:
            return {"available": False, "p_up": 0.5, "action": "flat", "score": 0.0}
        ep = _episode_from_df(df)
        if ep is None:
            return {"available": False, "p_up": 0.5, "action": "flat", "score": 0.0}
        feats, _ = ep
        obs = np.concatenate([feats[-1], [0.0]]).astype(np.float32)
        try:
            obs_t = torch.as_tensor(obs).reshape(1, -1)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_t)
                probs = dist.distribution.probs.cpu().numpy().reshape(-1)  # [flat, long, short]
            p_long, p_short = float(probs[1]), float(probs[2])
            score = float(np.clip(p_long - p_short, -1, 1))               # -1..1
            p_up = 0.5 + score * 0.5
            action = ("long" if score > 0.1 else "short" if score < -0.1 else "flat")
            return {"available": True, "p_up": round(p_up, 3), "action": action,
                    "score": round(score, 3), "p_long": round(p_long, 3), "p_short": round(p_short, 3)}
        except Exception:
            return {"available": False, "p_up": 0.5, "action": "flat", "score": 0.0}


def get_rl_agent(name: str, dfs: List[pd.DataFrame]) -> Optional["RLAgent"]:
    """Bellek (oturum) + disk (24s) önbellekli ajan — oturumda bir kez eğitir."""
    if not _HAS_RL:
        return None
    ag = _MEM_CACHE.get(name)
    if ag is None:
        ag = RLAgent(name)
        ag.fit_or_load(dfs)
        if ag.model is not None:
            _MEM_CACHE[name] = ag
    return ag if ag.model is not None else None
