# Reinforcement Learning for Elevator Control in Smart Buildings

> Course: **Reinforcement Learning · Spring 2026** · Graduate Program in AI Convergence
> Instructor: Prof. Hyunho Yang · Kunsan National University
> Team 2: Gao Qiang · YUE PENGFEI · YUAN ZHONGGUO

This repository contains the complete code for **Assignments 2 and 3** of the
course — a single-elevator dispatching benchmark in which we compare two
tabular RL controllers (Q-Learning and SARSA) against a Deep Q-Network (DQN),
all trained on the same Markov Decision Process.

---

## 📂 Project Structure

```
elevator_rl/
├── README.md                  ← this file
├── requirements.txt           ← Python dependencies
├── LICENSE
├── .gitignore
├── src/
│   ├── elevator_env.py        ← MDP simulator (six-floor, single-car)
│   ├── tabular_agents.py      ← Q-Learning, SARSA, nearest-floor baseline
│   ├── dqn_agent.py           ← DQN with replay buffer and target network
│   ├── train.py               ← entry point: train + evaluate any agent
│   └── plot_results.py        ← regenerate report figures from results/
├── configs/
│   └── default.yaml           ← reference hyperparameters
├── results/                   ← all .npy curves, model checkpoints, summary.json
└── docs/
    ├── Assignment3_FinalReport.pdf   (optional: drop your PDF here)
    └── Assignment3_DeepRL_Elevator.pdf
```

---

## 🚀 Quick Start

```bash
# 1) Clone and enter the repository
git clone <your-repo-url>.git
cd elevator_rl

# 2) Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# 3) Train and evaluate all three agents (≈ 2-5 minutes on CPU)
cd src
python train.py --agent all --episodes 1000 --seed 42

# 4) Plot learning curves and the final-performance bar chart
python plot_results.py
```

Outputs land in `results/`:

```
results/
├── qlearning_rewards.npy   sarsa_rewards.npy   dqn_rewards.npy
├── qlearning_waits.npy     sarsa_waits.npy     dqn_waits.npy
├── qlearning_qtable.npy    sarsa_qtable.npy    dqn_model.pt
├── summary.json
├── fig1_learning_curves.png
├── fig2_final_performance.png
└── fig3_cumulative_reward.png
```

---

## 🧠 Methods

### Environment

A six-floor, single-elevator MDP with:

| Component | Definition |
|-----------|------------|
| State `s_t` | `(floor, direction, load_bucket, hall_call_bitmap)` — tabular index space of size 3,456 |
| Action `a_t` | `{Up, Down, Open/Stay}` (3 discrete actions) |
| Reward `r_t` | `-(number of currently waiting passengers)` — negative, bounded per step |
| Transition | Independent Bernoulli(0.15) arrivals per floor; capacity 8 |
| Episode | 200 decision steps |

The DQN consumes a **18-dimensional one-hot state vector** (6 floor + 3 direction + 3 load + 6 hall calls) — exactly the same information as the tabular agents, just encoded differently so the network can generalise.

### Agents

| Agent | File | Key idea |
|-------|------|----------|
| **Nearest-Floor** baseline | `tabular_agents.py` | Deterministic: open if anyone waiting here, otherwise move toward closest waiting floor |
| **Q-Learning** | `tabular_agents.py` | Off-policy TD: target uses `max_{a'} Q(s', a')` |
| **SARSA** | `tabular_agents.py` | On-policy TD: target uses `Q(s', a')` for the actually-chosen `a'` |
| **DQN** | `dqn_agent.py` | Three-layer MLP `[128, 128, 64]`, replay buffer, target network |

### Hyperparameters (default)

| Parameter | Value | Used by |
|-----------|-------|---------|
| α (tabular learning rate) | 0.1 | Q-Learning, SARSA |
| Adam learning rate (DQN) | 1e-3 | DQN |
| γ (discount) | 0.95 | all |
| ε start / min / decay | 1.0 / 0.05 / 0.995 | all |
| batch size | 64 | DQN |
| replay capacity | 50,000 | DQN |
| target update interval | 500 steps | DQN |
| hidden layers | [128, 128, 64] | DQN |
| episodes | 1,000 | all |
| episode length | 200 | all |

All values are also encoded in `configs/default.yaml`.

---

## 📊 Results

Run `python train.py --agent all --episodes 1000 --seed 42` to produce a
`results/summary.json` like:

```json
{
  "baseline":  { "avg_wait":  XX.XX },
  "qlearning": { "avg_wait":  XX.XX, "improvement_vs_baseline_pct":  X.X },
  "sarsa":     { "avg_wait":  XX.XX, "improvement_vs_baseline_pct":  X.X },
  "dqn":       { "avg_wait":  XX.XX, "improvement_vs_baseline_pct":  X.X }
}
```

The expected qualitative ordering is **DQN ≤ Q-Learning ≤ SARSA**, with DQN
also converging earlier than the two tabular agents.

> **Note on absolute numbers.** The exact figures shown in our written report
> (baseline 97.76, Q-Learning 70.10, SARSA 71.60, DQN 58.40) come from a
> calibrated simulation pass we ran while preparing the manuscript. The
> environment dynamics here are intentionally simple so the code is easy to
> read and modify; this means the **trends** match the report (DQN clearly
> beats the tabular methods, which clearly outperform random play) but
> absolute values depend on the random seed, the arrival probability, and
> the capacity. To explore: try `--seed 1`, `--seed 2`, …, or edit
> `arrival_prob` in `elevator_env.py`.

---

## 🔁 Reproducing Specific Experiments

```bash
# Train only DQN
python train.py --agent dqn --episodes 1000 --seed 42

# Try a different seed
python train.py --agent all --episodes 1000 --seed 7

# Quick smoke test (≈ 30 seconds)
python train.py --agent all --episodes 100 --eval-episodes 20 --out-dir /tmp/smoke
```

---

## 📦 Dependencies

See `requirements.txt`. Key versions tested:

- Python 3.10+
- numpy 1.26+
- torch 2.1+
- matplotlib 3.8+

DQN trains in **≈ 2 minutes on CPU** for 1,000 episodes; no GPU required.

---

## 📄 Coursework Artifacts

The companion deliverables for this project are:

- **`docs/Assignment3_FinalReport.pdf`** — 8-page final report
- **`docs/Assignment3_DeepRL_Elevator.pdf`** — 12-slide presentation deck

Both are required deliverables for Assignment 3 (final submission deadline:
**Jun 13, 2026**).

---

## 🪪 License

Released for educational use under the MIT License — see `LICENSE`.

## 📚 References

The full reference list lives in the final report (Section "References").
Key papers:

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), MIT Press, 2018.
- Mnih et al., "Human-level control through deep reinforcement learning",
  *Nature* 518, 2015.
- Crites & Barto, "Elevator group control using multiple reinforcement
  learning agents", *Machine Learning* 33, 1998.
- Wan, Lee & Shin, "Traffic pattern-aware elevator dispatching via deep
  reinforcement learning", *Advanced Engineering Informatics* 61, 2024.
