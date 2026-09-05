"""
Gerçek Derin Öğrenme Modelleri (PyTorch): Transformer + LSTM + RL.

Spec: tek model değil; Transformer / LSTM / TFT / RL birlikte. Bu modül GERÇEK
nn.Module mimarileri kurar, çekilen OHLCV üzerinde self-supervised eğitir
(bir sonraki bar yön tahmini) ve ağırlıkları diske önbellekler (runs/models/).
CPU'da hızlı eğitilecek şekilde boyutlandırılmıştır.

  • TransformerForecaster — nn.TransformerEncoder ile dizi → p(up)
  • LSTMForecaster        — LSTM ile dizi → p(up)
  • DQNAgent              — pozisyon politikası (short/flat/long) öğrenir
                            (offline fitted-Q / contextual bandit)

torch yoksa bu modül import edilmez; ensemble klasik ML'e düşer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from ..analysis.indicators import rsi, macd, atr, ema, cci, mfi

SEQ_LEN = 32
torch.manual_seed(42)


# ----------------------------------------------------------------------------
# Özellik üretimi
# ----------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> np.ndarray:
    c = df["close"]
    feats = pd.DataFrame(index=df.index)
    feats["ret1"] = c.pct_change()
    feats["ret3"] = c.pct_change(3)
    feats["rsi"] = rsi(c, 14) / 100 - 0.5
    _, _, hist = macd(c)
    feats["macd_hist"] = hist / (c + 1e-9)
    feats["ema_ratio"] = ema(c, 20) / (ema(c, 50) + 1e-9) - 1
    feats["atr"] = atr(df, 14) / (c + 1e-9)
    feats["cci"] = cci(df) / 200
    feats["mfi"] = mfi(df) / 100 - 0.5
    feats["vol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)
    arr = feats.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)
    return arr


def make_sequences(feats: np.ndarray, closes: np.ndarray, horizon: int = 1):
    X, y, rets = [], [], []
    for i in range(SEQ_LEN, len(feats) - horizon):
        X.append(feats[i - SEQ_LEN:i])
        fut_ret = (closes[i + horizon] - closes[i]) / (closes[i] + 1e-9)
        y.append(1.0 if fut_ret > 0 else 0.0)
        rets.append(fut_ret)
    return np.array(X, np.float32), np.array(y, np.float32), np.array(rets, np.float32)


# ----------------------------------------------------------------------------
# Modeller
# ----------------------------------------------------------------------------
class TransformerForecaster(nn.Module):
    def __init__(self, n_feat: int, d_model: int = 32, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64,
                                         batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.proj(x) + self.pos
        h = self.encoder(h)
        return self.head(h[:, -1, :]).squeeze(-1)


class LSTMForecaster(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 32, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class QNetwork(nn.Module):
    """Durum (son bar özellikleri) → 3 aksiyonun beklenen getirisi (short/flat/long)."""
    def __init__(self, n_feat: int, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------
# Eğitim + önbellek
# ----------------------------------------------------------------------------
def _data_signature(df: pd.DataFrame) -> str:
    key = f"{len(df)}_{df['close'].iloc[-1]:.4f}_{df.index[-1]}"
    return hashlib.md5(key.encode()).hexdigest()[:10]


def _cache_dir() -> Path:
    p = Path("runs/models")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _train_classifier(model, X, y, epochs=40, lr=1e-3) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(X), torch.tensor(y)
    n = len(Xt)
    split = int(n * 0.85)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(split)
        for i in range(0, split, 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            logit = model(Xt[idx])
            loss = loss_fn(logit, yt[idx])
            loss.backward()
            opt.step()
    # doğrulama isabeti
    model.eval()
    with torch.no_grad():
        val_logit = model(Xt[split:])
        val_pred = (torch.sigmoid(val_logit) > 0.5).float()
        acc = (val_pred == yt[split:]).float().mean().item() if n > split else 0.5
    return float(acc)


def _train_dqn(model, X, rets, epochs=40, lr=1e-3) -> float:
    """Contextual-bandit/fitted-Q: her aksiyonun getiri hedefi.
    short=-ret, flat=0, long=+ret. Q-net bunları regresyonla öğrenir."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    states = torch.tensor(X[:, -1, :])         # son bar özelliği
    r = torch.tensor(rets)
    targets = torch.stack([-r, torch.zeros_like(r), r], dim=1)  # [N,3]
    n = len(states)
    split = int(n * 0.85)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(split)
        for i in range(0, split, 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            q = model(states[idx])
            loss = loss_fn(q, targets[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        q = model(states[split:])
        action = q.argmax(dim=1)               # 0 short,1 flat,2 long
        realized = torch.where(action == 2, r[split:],
                               torch.where(action == 0, -r[split:], torch.zeros_like(r[split:])))
        avg_reward = realized.mean().item() if n > split else 0.0
    return float(avg_reward)


class DeepEnsemble:
    """Üç modeli eğitir/yükler ve birleşik p(up) + RL aksiyonu döndürür."""

    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol.replace("/", "")
        self.timeframe = timeframe
        self.transformer: Optional[TransformerForecaster] = None
        self.lstm: Optional[LSTMForecaster] = None
        self.qnet: Optional[QNetwork] = None
        self.meta: Dict = {}

    def _paths(self, sig: str):
        d = _cache_dir()
        base = f"{self.symbol}_{self.timeframe}_{sig}"
        return (d / f"{base}_tf.pt", d / f"{base}_lstm.pt",
                d / f"{base}_dqn.pt", d / f"{base}_meta.json")

    def fit_or_load(self, df: pd.DataFrame, force: bool = False) -> Dict:
        feats = build_features(df)
        n_feat = feats.shape[1]
        sig = _data_signature(df)
        p_tf, p_lstm, p_dqn, p_meta = self._paths(sig)

        self.transformer = TransformerForecaster(n_feat)
        self.lstm = LSTMForecaster(n_feat)
        self.qnet = QNetwork(n_feat)

        if not force and p_tf.exists() and p_lstm.exists() and p_dqn.exists() and p_meta.exists():
            try:
                self.transformer.load_state_dict(torch.load(p_tf))
                self.lstm.load_state_dict(torch.load(p_lstm))
                self.qnet.load_state_dict(torch.load(p_dqn))
                self.meta = json.loads(p_meta.read_text())
                self.meta["from_cache"] = True
                return self.meta
            except Exception:
                pass

        # eğit
        X, y, rets = make_sequences(feats, df["close"].values)
        if len(X) < 50:
            self.meta = {"trained": False, "reason": "yetersiz veri", "n_feat": n_feat}
            return self.meta

        tf_acc = _train_classifier(self.transformer, X, y)
        lstm_acc = _train_classifier(self.lstm, X, y)
        rl_reward = _train_dqn(self.qnet, X, rets)

        try:
            torch.save(self.transformer.state_dict(), p_tf)
            torch.save(self.lstm.state_dict(), p_lstm)
            torch.save(self.qnet.state_dict(), p_dqn)
        except Exception:
            pass

        self.meta = {
            "trained": True, "from_cache": False, "n_samples": len(X), "n_feat": n_feat,
            "transformer_val_acc": round(tf_acc, 3),
            "lstm_val_acc": round(lstm_acc, 3),
            "rl_avg_reward": round(rl_reward, 5),
        }
        try:
            p_meta.write_text(json.dumps(self.meta))
        except Exception:
            pass
        return self.meta

    def predict(self, df: pd.DataFrame) -> Dict:
        feats = build_features(df)
        if len(feats) < SEQ_LEN + 1 or self.transformer is None:
            return {"p_up": 0.5, "rl_action": "flat", "rl_score": 0.0, "models": {}}
        seq = torch.tensor(feats[-SEQ_LEN:][None, :, :])
        last = torch.tensor(feats[-1][None, :])
        with torch.no_grad():
            p_tf = torch.sigmoid(self.transformer(seq)).item()
            p_lstm = torch.sigmoid(self.lstm(seq)).item()
            q = self.qnet(last)[0]
            action_idx = int(q.argmax().item())
            rl_score = float((q[2] - q[0]).item())  # long - short
        action = {0: "short", 1: "flat", 2: "long"}[action_idx]
        # RL aksiyonunu olasılığa çevir
        p_rl = 0.5 + np.tanh(rl_score * 50) * 0.5
        p_up = float(np.mean([p_tf, p_lstm, p_rl]))
        return {
            "p_up": p_up,
            "rl_action": action,
            "rl_score": rl_score,
            "models": {"transformer": round(p_tf, 3), "lstm": round(p_lstm, 3),
                       "rl_p_up": round(float(p_rl), 3)},
        }
