"""
Training entry point.

Examples
--------
    python train.py --agent qlearning --episodes 1000 --seed 42
    python train.py --agent sarsa     --episodes 1000 --seed 42
    python train.py --agent dqn       --episodes 1000 --seed 42

All curves and final greedy-evaluation metrics are written to results/.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from elevator_env import ElevatorEnv
from tabular_agents import QLearningAgent, SarsaAgent, NearestFloorAgent
from dqn_agent import DQNAgent


# --------------------------------------------------------------------------
def smooth(x: np.ndarray, window: int = 40) -> np.ndarray:
    """Trailing moving average with the same length as `x`."""
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(len(x)):
        s = max(0, i - window + 1)
        out[i] = float(np.mean(x[s : i + 1]))
    return out


# --------------------------------------------------------------------------
def train_tabular(
    agent_name: str,
    episodes: int,
    seed: int,
    out_dir: Path,
):
    """Train Q-Learning or SARSA on the tabular state representation."""
    env = ElevatorEnv(seed=seed)

    if agent_name == "qlearning":
        agent = QLearningAgent(env.n_states, env.n_actions, seed=seed)
    elif agent_name == "sarsa":
        agent = SarsaAgent(env.n_states, env.n_actions, seed=seed)
    else:
        raise ValueError(f"Unknown tabular agent: {agent_name}")

    rewards, waits, eps_log = [], [], []

    for ep in range(episodes):
        s = env.reset()
        if agent_name == "sarsa":
            a = agent.select_action(s)

        ep_reward = 0.0
        wait_sum = 0.0
        steps = 0

        done = False
        while not done:
            if agent_name == "qlearning":
                a = agent.select_action(s)
            s_next, r, done, info = env.step(a)
            wait_sum += info.waiting_total
            ep_reward += r
            steps += 1

            if agent_name == "qlearning":
                agent.update(s, a, r, s_next, done)
                s = s_next
            else:  # SARSA
                a_next = agent.select_action(s_next) if not done else 0
                agent.update(s, a, r, s_next, a_next, done)
                s, a = s_next, a_next

        agent.decay_epsilon()
        rewards.append(ep_reward)
        waits.append(wait_sum / max(steps, 1))
        eps_log.append(agent.epsilon)

        if (ep + 1) % 100 == 0:
            print(
                f"[{agent_name}] ep {ep+1:4d} | "
                f"reward {ep_reward:>10.1f} | "
                f"avg wait {waits[-1]:6.2f} | "
                f"eps {agent.epsilon:.3f}"
            )

    return agent, np.array(rewards), np.array(waits), np.array(eps_log)


# --------------------------------------------------------------------------
def train_dqn(
    episodes: int,
    seed: int,
    out_dir: Path,
):
    """Train the Deep Q-Network agent."""
    env = ElevatorEnv(seed=seed)
    agent = DQNAgent(state_dim=env.state_vector_dim, n_actions=env.n_actions, seed=seed)

    rewards, waits, eps_log, losses = [], [], [], []

    for ep in range(episodes):
        env.reset()
        s_vec = env.encode_state_vector()

        ep_reward = 0.0
        wait_sum = 0.0
        ep_loss = []
        steps = 0

        done = False
        while not done:
            a = agent.select_action(s_vec)
            _, r, done, info = env.step(a)
            s_next_vec = env.encode_state_vector()

            agent.push(s_vec, a, r, s_next_vec, done)
            loss = agent.update()
            if loss is not None:
                ep_loss.append(loss)

            wait_sum += info.waiting_total
            ep_reward += r
            steps += 1
            s_vec = s_next_vec

        agent.decay_epsilon()
        rewards.append(ep_reward)
        waits.append(wait_sum / max(steps, 1))
        eps_log.append(agent.epsilon)
        losses.append(float(np.mean(ep_loss)) if ep_loss else 0.0)

        if (ep + 1) % 50 == 0:
            print(
                f"[dqn] ep {ep+1:4d} | "
                f"reward {ep_reward:>10.1f} | "
                f"avg wait {waits[-1]:6.2f} | "
                f"loss {losses[-1]:7.3f} | "
                f"eps {agent.epsilon:.3f}"
            )

    return agent, np.array(rewards), np.array(waits), np.array(eps_log), np.array(losses)


# --------------------------------------------------------------------------
def evaluate_greedy(env_seed: int, agent, agent_kind: str, n_episodes: int = 100) -> float:
    """
    Run a fully greedy evaluation and return the mean per-step waiting cost.

    The per-step waiting cost for an episode is defined as
        sum_{t} (number of waiting passengers at step t) / T
    averaged over evaluation episodes.  This matches the metric reported
    in Assignment 2 (Section 4.4) and Assignment 3 (Section 5.2).
    """
    env = ElevatorEnv(seed=env_seed)
    per_episode_means = []

    for _ in range(n_episodes):
        s = env.reset()
        s_vec = env.encode_state_vector() if agent_kind == "dqn" else None
        ep_wait_sum = 0.0
        steps = 0
        done = False
        while not done:
            if agent_kind == "tabular":
                a = agent.select_action(s, greedy=True)
            elif agent_kind == "dqn":
                a = agent.select_action(s_vec, greedy=True)
            elif agent_kind == "baseline":
                a = agent.select_action(env, greedy=True)
            else:
                raise ValueError(agent_kind)

            s, _, done, info = env.step(a)
            if agent_kind == "dqn":
                s_vec = env.encode_state_vector()
            ep_wait_sum += info.waiting_total  # cumulative wait_clock at this step
            steps += 1

        per_episode_means.append(ep_wait_sum / max(steps, 1))

    return float(np.mean(per_episode_means))


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train RL agents on the elevator MDP.")
    parser.add_argument("--agent", choices=["qlearning", "sarsa", "dqn", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=99)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agents_to_run = ["qlearning", "sarsa", "dqn"] if args.agent == "all" else [args.agent]
    summary: dict[str, dict] = {}

    # Always evaluate the baseline for reference
    print("\n--- Evaluating nearest-floor baseline ---")
    baseline = NearestFloorAgent()
    baseline_wait = evaluate_greedy(args.eval_seed, baseline, "baseline", args.eval_episodes)
    print(f"Baseline avg waiting time: {baseline_wait:.2f}")
    summary["baseline"] = {"avg_wait": baseline_wait}

    for name in agents_to_run:
        print(f"\n--- Training {name.upper()} ---")
        t0 = time.time()
        if name == "dqn":
            agent, rewards, waits, eps_log, losses = train_dqn(args.episodes, args.seed, out_dir)
            np.save(out_dir / f"{name}_losses.npy", losses)
            agent.save(str(out_dir / f"{name}_model.pt"))
        else:
            agent, rewards, waits, eps_log = train_tabular(name, args.episodes, args.seed, out_dir)
            np.save(out_dir / f"{name}_qtable.npy", agent.Q)

        np.save(out_dir / f"{name}_rewards.npy", rewards)
        np.save(out_dir / f"{name}_waits.npy", waits)
        np.save(out_dir / f"{name}_epsilon.npy", eps_log)

        eval_wait = evaluate_greedy(
            args.eval_seed,
            agent,
            "dqn" if name == "dqn" else "tabular",
            args.eval_episodes,
        )
        elapsed = time.time() - t0

        summary[name] = {
            "avg_wait": float(eval_wait),
            "improvement_vs_baseline_pct": (baseline_wait - eval_wait) / baseline_wait * 100.0,
            "training_time_sec": elapsed,
            "final_train_avg_wait": float(np.mean(waits[-50:])),
        }
        print(
            f"[{name}] eval avg wait: {eval_wait:.2f}  "
            f"({summary[name]['improvement_vs_baseline_pct']:.1f}% vs baseline)  "
            f"trained in {elapsed:.1f}s"
        )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k:>10}: {v}")
    print(f"\nResults written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
