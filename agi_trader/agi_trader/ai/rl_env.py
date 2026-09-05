"""
RL v2 — Trading Gym Ortamı (Gymnasium).

Sıralı karar ortamı: ajan her barda bir POZİSYON tutar (flat/long/short) ve
zamanla pozisyonu yönetir. Tek-bar bandit değil; gerçek RL (PPO/A2C) sürekli
girip-çıkma zamanlamasını öğrenir.

  • Gözlem: o barın normalize özellik vektörü + mevcut pozisyon (one-hot benzeri)
  • Aksiyon: 0=flat · 1=long · 2=short (ayrık)
  • Ödül: pozisyon × sonraki-bar getirisi − işlem maliyeti (pozisyon değişince)

Çoklu-varlık: birden çok varlığın (features, returns) dizileri ardışık eklenerek
tek ortamda eğitilebilir (set_episodes) — multi-asset PPO.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:  # pragma: no cover
    _HAS_GYM = False
    gym = object  # type: ignore


if _HAS_GYM:
    class TradingEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, episodes: List[Tuple[np.ndarray, np.ndarray]], cost: float = 0.0004):
            """episodes: [(features (T,F), returns (T,)), ...] — her biri bir varlık serisi."""
            super().__init__()
            self.episodes = [(f.astype(np.float32), r.astype(np.float32)) for f, r in episodes if len(r) > 10]
            if not self.episodes:
                raise ValueError("RL ortamı için yeterli veri yok")
            self.cost = float(cost)
            n_feat = self.episodes[0][0].shape[1]
            self.action_space = spaces.Discrete(3)               # 0 flat, 1 long, 2 short
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(n_feat + 1,), dtype=np.float32)
            self._ep = 0
            self.t = 0
            self.position = 0.0
            self.feats = None
            self.rets = None

        def reset(self, *, seed: Optional[int] = None, options=None):
            super().reset(seed=seed)
            self._ep = int(self.np_random.integers(0, len(self.episodes)))
            self.feats, self.rets = self.episodes[self._ep]
            self.t = 0
            self.position = 0.0
            return self._obs(), {}

        def _obs(self) -> np.ndarray:
            return np.concatenate([self.feats[self.t], [self.position]]).astype(np.float32)

        def step(self, action: int):
            new_pos = {0: 0.0, 1: 1.0, 2: -1.0}[int(action)]
            ret = float(self.rets[self.t])
            reward = new_pos * ret - (self.cost if new_pos != self.position else 0.0)
            self.position = new_pos
            self.t += 1
            terminated = self.t >= len(self.rets) - 1
            obs = self._obs() if not terminated else np.zeros(self.observation_space.shape, np.float32)
            # ödülü ölçekle (PPO için stabil)
            return obs, float(reward * 100.0), bool(terminated), False, {"position": new_pos}
else:  # pragma: no cover
    class TradingEnv:  # type: ignore
        def __init__(self, *a, **k):
            raise RuntimeError("gymnasium kurulu değil")
