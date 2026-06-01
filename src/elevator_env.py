"""
Elevator simulation environment.

A simplified single-elevator, six-floor MDP used across Assignments 2 and 3.

State:
    s_t = (f_t, d_t, l_t, h_t)
        f_t : current floor index           (0..N_FLOORS-1)
        d_t : direction of travel           (UP=0, DOWN=1, IDLE=2)
        l_t : discretised passenger load    (EMPTY=0, PARTIAL=1, FULL=2)
        h_t : 6-bit hall-call bitmap        (integer 0..63)

Actions:
    0 : Move Up
    1 : Move Down
    2 : Open door / stay (service current floor; board/alight)

Reward:
    r_t = - sum_f w_t(f)                    (negative accumulated waiting time)

Transitions:
    Passenger arrivals follow an independent Bernoulli(0.15) process per floor
    per step.  Episode length is fixed at 200 decision steps.

The class exposes a Gymnasium-style reset()/step() API so the same env object
can be passed to both tabular and DQN agents.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ----------------------------- Constants -----------------------------------
N_FLOORS = 6
N_DIR = 3            # UP, DOWN, IDLE
N_LOAD = 3           # EMPTY, PARTIAL, FULL
N_HALL = 1 << N_FLOORS  # 64

UP, DOWN, IDLE = 0, 1, 2
EMPTY, PARTIAL, FULL = 0, 1, 2

ACTION_UP, ACTION_DOWN, ACTION_OPEN = 0, 1, 2
N_ACTIONS = 3

CAR_CAPACITY = 8


# ----------------------------- Helpers -------------------------------------
def load_bucket(load: int) -> int:
    """Discretise integer passenger load into EMPTY / PARTIAL / FULL."""
    if load <= 0:
        return EMPTY
    if load >= CAR_CAPACITY:
        return FULL
    return PARTIAL


def hall_bitmap(waiting: np.ndarray) -> int:
    """Convert a per-floor waiting array into an integer bitmap (0..63)."""
    bits = 0
    for f in range(N_FLOORS):
        if waiting[f] > 0:
            bits |= (1 << f)
    return bits


# ----------------------------- Environment ---------------------------------
@dataclass
class StepInfo:
    waiting_total: float        # cumulative wait_clock sum (used for reward)
    instant_waiting: float      # current number of waiting passengers (used for evaluation)
    boarded: int
    alighted: int


class ElevatorEnv:
    """
    Six-floor, single-car elevator MDP.

    Parameters
    ----------
    n_floors : int
        Number of floors (default 6).
    arrival_prob : float
        Per-floor Bernoulli arrival probability per step (default 0.15).
    episode_length : int
        Number of decision steps per episode (default 200).
    seed : int or None
        RNG seed.
    """

    def __init__(
        self,
        n_floors: int = N_FLOORS,
        arrival_prob: float = 0.15,
        episode_length: int = 200,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        n_floors : int
            Number of floors in the building.
        arrival_prob : float
            Per-floor Bernoulli arrival probability per step.
        episode_length : int
            Number of decision steps per episode.
        seed : int or None
            RNG seed for reproducibility.

        Notes
        -----
        Reward is the negative number of currently waiting passengers
        (bounded per step). The cumulative ``wait_clock`` sum is
        exposed via ``info.waiting_total`` and is the metric reported
        in Assignments 2 and 3.
        """
        self.n_floors = n_floors
        self.arrival_prob = arrival_prob
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)

        # State variables (initialised in reset)
        self.car_floor: int = 0
        self.direction: int = IDLE
        self.car_load: int = 0
        self.waiting: np.ndarray = np.zeros(self.n_floors, dtype=np.int32)
        self.wait_clock: np.ndarray = np.zeros(self.n_floors, dtype=np.float32)
        self.t: int = 0

        self.n_states = self.n_floors * N_DIR * N_LOAD * N_HALL  # 3,456 for 6 floors
        self.n_actions = N_ACTIONS

    # ---------- API ---------------------------------------------------------
    def reset(self) -> int:
        self.car_floor = int(self.rng.integers(0, self.n_floors))
        self.direction = IDLE
        self.car_load = 0
        self.waiting = np.zeros(self.n_floors, dtype=np.int32)
        self.wait_clock = np.zeros(self.n_floors, dtype=np.float32)
        self.t = 0
        return self._state_index()

    def step(self, action: int):
        assert action in (ACTION_UP, ACTION_DOWN, ACTION_OPEN)

        boarded = 0
        alighted = 0

        # -------- Apply action ---------------------------------------------
        if action == ACTION_UP:
            self.direction = UP
            self.car_floor = min(self.car_floor + 1, self.n_floors - 1)
        elif action == ACTION_DOWN:
            self.direction = DOWN
            self.car_floor = max(self.car_floor - 1, 0)
        else:  # ACTION_OPEN
            self.direction = IDLE
            # Alight: passengers in the car randomly leave (~50% per stop in this simplified model)
            if self.car_load > 0:
                alighted = int(self.rng.integers(0, self.car_load + 1))
                self.car_load -= alighted
            # Board: take all waiting passengers up to remaining capacity
            free = CAR_CAPACITY - self.car_load
            board = min(int(self.waiting[self.car_floor]), free)
            self.waiting[self.car_floor] -= board
            self.car_load += board
            boarded = board
            if board > 0:
                # Reset wait clock at this floor
                self.wait_clock[self.car_floor] = 0.0

        # -------- Stochastic arrivals --------------------------------------
        for f in range(self.n_floors):
            if self.rng.random() < self.arrival_prob:
                self.waiting[f] += 1

        # -------- Update wait clocks (one tick for any waiting passenger) --
        # wait_clock accumulates within an episode (used for analysis/diagnostics).
        for f in range(self.n_floors):
            if self.waiting[f] > 0:
                self.wait_clock[f] += 1.0  # one time-unit per waiting floor

        # -------- Reward ---------------------------------------------------
        # Negative number of currently waiting passengers — bounded per step,
        # so total returns scale linearly with episode length.  This is the
        # standard formulation in the elevator-RL literature and keeps DQN
        # gradients well-conditioned.
        instant_waiting = float(self.waiting.sum())
        reward = -instant_waiting

        # Cumulative wait_clock sum is also exposed for diagnostic purposes.
        waiting_total = float(self.wait_clock.sum())

        # -------- Episode termination --------------------------------------
        self.t += 1
        done = self.t >= self.episode_length

        info = StepInfo(
            waiting_total=waiting_total,
            instant_waiting=instant_waiting,
            boarded=boarded,
            alighted=alighted,
        )
        return self._state_index(), reward, done, info

    # ---------- State helpers ----------------------------------------------
    def _state_index(self) -> int:
        f = self.car_floor
        d = self.direction
        lb = load_bucket(self.car_load)
        hb = hall_bitmap(self.waiting)
        idx = (
            f * (N_DIR * N_LOAD * N_HALL)
            + d * (N_LOAD * N_HALL)
            + lb * N_HALL
            + hb
        )
        return int(idx)

    def encode_state_vector(self) -> np.ndarray:
        """
        Return the 18-dim feature vector used by DQN.

        [floor_one_hot(6) | direction_one_hot(3) | load_one_hot(3) | hall_calls(6)]
        """
        f_oh = np.zeros(self.n_floors, dtype=np.float32)
        f_oh[self.car_floor] = 1.0
        d_oh = np.zeros(N_DIR, dtype=np.float32)
        d_oh[self.direction] = 1.0
        l_oh = np.zeros(N_LOAD, dtype=np.float32)
        l_oh[load_bucket(self.car_load)] = 1.0
        h = (self.waiting > 0).astype(np.float32)
        return np.concatenate([f_oh, d_oh, l_oh, h])

    @property
    def state_vector_dim(self) -> int:
        return self.n_floors + N_DIR + N_LOAD + self.n_floors  # 18 for 6 floors


if __name__ == "__main__":
    env = ElevatorEnv(seed=0)
    s = env.reset()
    print(f"Initial state index: {s}")
    print(f"Initial state vector: {env.encode_state_vector()}")
    total_r = 0.0
    for _ in range(10):
        a = int(np.random.randint(0, 3))
        s, r, done, info = env.step(a)
        total_r += r
    print(f"Random 10-step total reward: {total_r:.2f}")
