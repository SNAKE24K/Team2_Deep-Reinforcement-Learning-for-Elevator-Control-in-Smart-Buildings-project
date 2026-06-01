"""
Tabular RL agents for elevator control (Assignment 2 baseline).

Implements:
    - QLearningAgent  : off-policy TD control
    - SarsaAgent      : on-policy TD control
    - NearestFloorAgent : deterministic heuristic baseline
"""
from __future__ import annotations

import numpy as np


class EpsilonGreedyAgent:
    """Common ε-greedy machinery shared by Q-Learning and SARSA."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Small uniform initialisation breaks ties symmetrically.
        self.Q = self.rng.uniform(-0.01, 0.01, size=(n_states, n_actions)).astype(np.float64)

    def select_action(self, state: int, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[state]))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class QLearningAgent(EpsilonGreedyAgent):
    """Q-Learning (off-policy TD control)."""

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> float:
        best_next = 0.0 if done else float(np.max(self.Q[s_next]))
        td_target = r + self.gamma * best_next
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        return float(td_error)


class SarsaAgent(EpsilonGreedyAgent):
    """SARSA (on-policy TD control)."""

    def update(
        self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool
    ) -> float:
        next_val = 0.0 if done else float(self.Q[s_next, a_next])
        td_target = r + self.gamma * next_val
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        return float(td_error)


class NearestFloorAgent:
    """
    Deterministic nearest-floor heuristic.

    Rules:
        - If anyone is waiting on the current floor, OPEN the door.
        - Otherwise move toward the nearest waiting floor.
        - If nobody is waiting anywhere, stay (OPEN).
    """

    def __init__(self):
        self.epsilon = 0.0

    def select_action(self, env, greedy: bool = True) -> int:
        from elevator_env import ACTION_UP, ACTION_DOWN, ACTION_OPEN

        floor = env.car_floor
        waiting = env.waiting
        if waiting[floor] > 0:
            return ACTION_OPEN
        targets = [f for f in range(env.n_floors) if waiting[f] > 0]
        if not targets:
            return ACTION_OPEN
        nearest = min(targets, key=lambda x: abs(x - floor))
        if nearest == floor:
            return ACTION_OPEN
        return ACTION_DOWN if nearest < floor else ACTION_UP

    # No-op stubs so the training loop can call them uniformly.
    def update(self, *args, **kwargs):
        return 0.0

    def decay_epsilon(self):
        pass
