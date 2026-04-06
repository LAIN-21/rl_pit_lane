import os
import math
import random
from dataclasses import dataclass
from collections import deque

import numpy as np

import matplotlib
matplotlib.use("Agg")  # avoid Tk backend issues
import matplotlib.pyplot as plt

import gymnasium as gym
import gym_race

import torch
import torch.nn as nn
import torch.optim as optim


VERSION_NAME = "DQN_v01"

REPORT_EPISODES = 100
DISPLAY_EPISODES = 1  # render every N episodes (set large for faster training)

# Single switch for the assignment:
# - RENDER = False -> train fast, no window (uses SDL "dummy" drivers)
# - RENDER = True  -> watch behavior with a pygame window
RENDER = False


@dataclass
class DQNConfig:
    num_episodes: int = 20_000
    max_t: int = 2000
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 100_000
    learning_starts: int = 2_000
    train_freq: int = 4
    grad_steps: int = 1
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay_episodes: int = 12_000


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, s, a, r, s2, d) -> None:
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.asarray(s2, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
        )


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

"""
What changed vs. the old tabular Q-learning update (in `Pyrace_RL_QTable.py`)?

Old (tabular):
    best_q = max_a' Q_table[s', a']
    Q_table[s, a] += alpha * (r + gamma * best_q - Q_table[s, a])

New (DQN):
    - We approximate Q(s,a) with a neural network Q_net(s)[a]
    - We store transitions (s,a,r,s',done) in a replay buffer
    - We sample random minibatches and minimize the TD error with gradient descent:
        target = r + gamma*(1-done)*max_a' Q_net(s')[a']
        loss   = Huber( Q_net(s)[a], target )
      and backpropagate `loss` to update Q_net weights.
"""


def linear_eps(episode: int, cfg: DQNConfig) -> float:
    frac = min(1.0, episode / float(cfg.eps_decay_episodes))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def select_action(q_net: QNetwork, state: np.ndarray, eps: float, n_actions: int, device: str) -> int:
    if random.random() < eps:
        return random.randrange(n_actions)
    with torch.no_grad():
        s = torch.from_numpy(state).to(device=device, dtype=torch.float32).unsqueeze(0)
        q = q_net(s)
        return int(torch.argmax(q, dim=1).item())


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    # env gives ints in [0,10]; normalize to [0,1] for stability
    return (obs.astype(np.float32) / 10.0).copy()


def train_step(
    q_net: QNetwork,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer,
    cfg: DQNConfig,
    device: str,
):
    if len(replay) < max(cfg.learning_starts, cfg.batch_size):
        return None

    s, a, r, s2, d = replay.sample(cfg.batch_size)
    s_t = torch.from_numpy(s).to(device)
    a_t = torch.from_numpy(a).to(device)
    r_t = torch.from_numpy(r).to(device)
    s2_t = torch.from_numpy(s2).to(device)
    d_t = torch.from_numpy(d).to(device)

    q_sa = q_net(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)

    with torch.no_grad():
        next_q = q_net(s2_t).max(dim=1).values
        target = r_t + cfg.gamma * (1.0 - d_t) * next_q

    loss = nn.SmoothL1Loss()(q_sa, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def load_checkpoint(
    path: str,
    q_net: QNetwork,
    optimizer: optim.Optimizer | None = None,
    device: str = "cpu",
    play_only: bool = False,
):
    """
    Loads either:
    - a full checkpoint dict with key `model_state_dict`, or
    - a minimal weights-only file produced by `torch.save(q_net.state_dict(), "model_final.pt")`.
    """
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        q_net.load_state_dict(ckpt["model_state_dict"])
        if (not play_only) and (optimizer is not None) and ("optimizer_state_dict" in ckpt) and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        episode = int(ckpt.get("episode", 0))
        cfg_dict = ckpt.get("cfg", None)
        return episode, cfg_dict

    # weights-only
    q_net.load_state_dict(ckpt)
    return 0, None


def simulate(
    learning: bool = True,
    episode_start: int = 0,
    checkpoint_path: str | None = None,
    play_only: bool = False,
):
    cfg = DQNConfig()
    if play_only:
        # Make "load and play" a short demo by default.
        cfg.num_episodes = 5

    if not RENDER:
        # Must be set before pygame initializes (happens inside env init).
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    env = gym.make("Pyrace-v1").unwrapped

    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    q_net = QNetwork(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(q_net.parameters(), lr=cfg.lr) if learning else None
    replay = ReplayBuffer(cfg.replay_size)

    total_rewards = []
    losses = []
    max_reward = -1e9
    global_step = 0

    if not os.path.exists(f"models_{VERSION_NAME}"):
        os.makedirs(f"models_{VERSION_NAME}")

    # render control
    env.set_view(RENDER)

    if checkpoint_path is not None:
        loaded_episode, loaded_cfg = load_checkpoint(
            checkpoint_path,
            q_net=q_net,
            optimizer=optimizer,
            device=device,
            play_only=play_only,
        )
        print(f"Loaded checkpoint from {checkpoint_path} (episode={loaded_episode})")
        if (loaded_cfg is not None) and isinstance(loaded_cfg, dict):
            # Keep code simple: we print it, but we don't try to fully reconstruct the dataclass.
            print("Checkpoint cfg:", loaded_cfg)
        if episode_start == 0:
            episode_start = loaded_episode

    for episode in range(episode_start, cfg.num_episodes + episode_start):
        obs, _ = env.reset()
        state = preprocess_obs(obs)
        total_reward = 0.0
        episode_idx = (episode - episode_start) + 1  # 1..cfg.num_episodes

        if not learning:
            env.pyrace.mode = 2  # continuous display

        eps = linear_eps(episode, cfg) if learning else 0.0

        for t in range(cfg.max_t):
            action = select_action(q_net, state, eps, n_actions, device)
            obs2, reward, done, _, info = env.step(action)
            state2 = preprocess_obs(obs2)

            replay.add(state, action, float(reward), state2, float(done))
            total_reward += float(reward)
            global_step += 1

            if learning and (global_step % cfg.train_freq == 0):
                for _ in range(cfg.grad_steps):
                    # optimizer is always present in learning mode
                    loss = train_step(q_net, optimizer, replay, cfg, device)
                    if loss is not None:
                        losses.append(loss)

            state = state2

            if RENDER:
                do_render = (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2)
                if do_render:
                    env.set_msgs(
                        [
                            "DQN SIMULATE",
                            f"Episode: {episode}",
                            f"Time steps: {t}",
                            f"eps: {eps:.3f}",
                            f"check: {info['check']}",
                            f"dist: {info['dist']}",
                            f"crash: {info['crash']}",
                            f"Reward: {total_reward:.0f}",
                            f"Max Reward: {max_reward:.0f}",
                            f"Replay: {len(replay)}",
                        ]
                    )
                    env.render()

            if done or (t >= cfg.max_t - 1):
                break

        total_rewards.append(total_reward)
        if total_reward > max_reward:
            max_reward = total_reward

        if (episode_idx == 1) or (episode_idx % REPORT_EPISODES == 0) or (episode_idx == cfg.num_episodes):
            print(
                f"Episode {episode_idx}/{cfg.num_episodes} | "
                f"reward={total_reward:.0f} | "
                f"max_reward={max_reward:.0f} | "
                f"eps={eps:.3f} | "
                f"replay={len(replay)}"
            )

        if learning and (episode > 0) and (episode % REPORT_EPISODES == 0):
            plt.figure()
            plt.plot(total_rewards)
            plt.ylabel("rewards")
            plt.xlabel("episode")
            plt.tight_layout()
            plt.savefig(f"models_{VERSION_NAME}/rewards_latest.png")
            plt.close()

            if len(losses) > 0:
                plt.figure()
                plt.plot(losses[-5000:])
                plt.ylabel("loss (last steps)")
                plt.xlabel("train step")
                plt.tight_layout()
                plt.savefig(f"models_{VERSION_NAME}/loss_latest.png")
                plt.close()

    # Minimal final artifact for the assignment: model weights only
    torch.save(q_net.state_dict(), f"models_{VERSION_NAME}/model_final.pt")
    env.close()


def load_and_play(checkpoint_path: str, learning: bool = False):
    # Mirrors the original `load_and_play(...)` structure, but uses a DQN checkpoint.
    simulate(
        learning=learning,
        episode_start=0,
        checkpoint_path=checkpoint_path,
        play_only=True,
    )


if __name__ == "__main__":
    # Typical usage:
    # - Train without window: set RENDER = False
    # - Watch with window:    set RENDER = True
    # - For quick viewing, also set DISPLAY_EPISODES small (e.g., 1 or 5)
    simulate(learning=True, episode_start=0)

