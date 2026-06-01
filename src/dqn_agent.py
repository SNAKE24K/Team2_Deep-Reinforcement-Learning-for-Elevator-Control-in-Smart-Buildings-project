"""
Deep Q-Network agent for elevator control (Assignment 3).

Architecture:
    Input(18) -> FC 128 -> ReLU -> FC 128 -> ReLU -> FC 64 -> ReLU -> FC 3

Stabilisation:
    - Experience replay buffer (capacity 50,000, FIFO)
    - Target network (hard copy every C=500 environment steps)
    - Gradient clipping at L2 norm 10.0

Hyperparameters match the report (build_docx.js, Section 4.3).
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- Network -------------------------------------
class QNetwork(nn.Module):
    """Three-layer fully-connected MLP that maps states to Q-values."""

    def __init__(self, in_dim: int = 18, n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------- Replay buffer -------------------------------
@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s_next: np.ndarray
    done: bool


class ReplayBuffer:
    """Uniformly-sampled FIFO replay buffer."""

    def __init__(self, capacity: int = 50_000, seed: int | None = None):
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, s, a, r, s_next, done):
        self.buffer.append(Transition(s, a, float(r), s_next, bool(done)))

    def sample(self, batch_size: int):
        batch = self._rng.sample(self.buffer, batch_size)
        s = np.stack([t.s for t in batch]).astype(np.float32)
        a = np.array([t.a for t in batch], dtype=np.int64)
        r = np.array([t.r for t in batch], dtype=np.float32)
        s_next = np.stack([t.s_next for t in batch]).astype(np.float32)
        done = np.array([t.done for t in batch], dtype=np.float32)
        return s, a, r, s_next, done

    def __len__(self):
        return len(self.buffer)


# ----------------------------- DQN Agent -----------------------------------
class DQNAgent:
    """Standard DQN with target network and experience replay."""

    def __init__(
        self,
        state_dim: int = 18,
        n_actions: int = 3,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        target_update_interval: int = 500,   # in environment steps
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        grad_clip: float = 10.0,
        device: str | None = None,
        seed: int | None = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.grad_clip = grad_clip

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Networks
        self.online = QNetwork(state_dim, n_actions).to(self.device)
        self.target = QNetwork(state_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

        self._step_count = 0
        self._rng = np.random.default_rng(seed)

    # ---------- Acting -----------------------------------------------------
    def select_action(self, state_vec: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        with torch.no_grad():
            s = torch.from_numpy(state_vec.astype(np.float32)).unsqueeze(0).to(self.device)
            q = self.online(s)
            return int(torch.argmax(q, dim=1).item())

    # ---------- Learning ---------------------------------------------------
    def push(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)

    def update(self) -> float | None:
        """Take one mini-batch SGD step. Returns the loss, or None if not ready."""
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, s_next, done = self.buffer.sample(self.batch_size)
        s_t = torch.from_numpy(s).to(self.device)
        a_t = torch.from_numpy(a).to(self.device)
        r_t = torch.from_numpy(r).to(self.device)
        s_next_t = torch.from_numpy(s_next).to(self.device)
        done_t = torch.from_numpy(done).to(self.device)

        # Q(s, a; θ)
        q_pred = self.online(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

        # y = r + γ * max_a' Q(s', a'; θ⁻) * (1 - done)
        with torch.no_grad():
            q_next = self.target(s_next_t).max(dim=1).values
            y = r_t + self.gamma * q_next * (1.0 - done_t)

        loss = F.mse_loss(q_pred, y)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()

        # Hard target update on schedule
        self._step_count += 1
        if self._step_count % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ---------- I/O --------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self._step_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.epsilon = float(ckpt.get("epsilon", self.epsilon))
        self._step_count = int(ckpt.get("step_count", 0))
