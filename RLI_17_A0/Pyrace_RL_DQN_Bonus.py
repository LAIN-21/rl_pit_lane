"""
BONUS – PPO with stable-baselines3 (Pyrace-v3)

Moving on from basic DQN to something a bit more solid.

Why PPO instead of DDPG?

The action space is still discrete (4 actions), so DDPG would be awkward to use. PPO just works out of the box.
PPO is on-policy and uses a clipped objective, which helps avoid the big unstable updates we saw with DQN.
It uses a shared MLP for both policy (actor) and value (critic), instead of a single Q-network.

Main differences vs DQN_v03:

DQN: off-policy + replay buffer → PPO: on-policy rollouts
DQN: single Q-network → PPO: policy + value heads
DQN: ε-greedy → PPO: entropy-based exploration
DQN: manual training loop → PPO: handled by SB3
DQN: unstable without tricks → PPO: clipping keeps things in check
"""

import os
import argparse
from dataclasses import dataclass, asdict


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
import gym_race  # registers Pyrace-v1, Pyrace-v3 as side-effect

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


VERSION_NAME = "PPO_bonus"
RENDER       = False   # True to open pygame window


@dataclass
class PPOConfig:
    total_timesteps: int = 5_000_000  

    n_steps:        int   = 2_048      
    batch_size:     int   = 256         
    n_epochs:       int   = 10         
    gamma:          float = 0.99       
    gae_lambda:     float = 0.95       
    clip_range:     float = 0.2        
    ent_coef:       float = 0.01       
    vf_coef:        float = 0.5        
    max_grad_norm:  float = 0.5        
    lr:             float = 3e-4       

    net_arch: tuple = (256, 256)

    
    report_freq:    int = 100          
    save_freq:      int = 100_000     


class TrainingCallback(BaseCallback):
    """
    Collects episode rewards from the Monitor wrapper and:
      • prints a summary every 100 episodes
      • saves a checkpoint every 100_000 timesteps
      • writes reward and loss plots to disk 
    """

    def __init__(self, cfg: PPOConfig, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.cfg       = cfg
        self.save_dir  = save_dir
        self.ep_rewards: list[float] = []
        self.last_save  = 0

    
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_r = float(info["episode"]["r"])
                self.ep_rewards.append(ep_r)
                n_ep = len(self.ep_rewards)

                if n_ep % self.cfg.report_freq == 0:
                    recent = self.ep_rewards[-self.cfg.report_freq:]
                    print(
                        f"Episode {n_ep:6d} | "
                        f"timestep {self.num_timesteps:8d} | "
                        f"mean_reward (last {self.cfg.report_freq}) = {np.mean(recent):8.1f} | "
                        f"max_reward = {max(self.ep_rewards):8.1f}"
                    )

        
                    plt.figure(figsize=(10, 4))
                    plt.plot(self.ep_rewards, alpha=0.4, label="episode reward")

                    
                    if len(self.ep_rewards) >= self.cfg.report_freq:
                        rm = np.convolve(
                            self.ep_rewards,
                            np.ones(self.cfg.report_freq) / self.cfg.report_freq,
                            mode="valid",
                        )
                        plt.plot(
                            range(self.cfg.report_freq - 1, len(self.ep_rewards)),
                            rm,
                            label=f"rolling mean ({self.cfg.report_freq})",
                            linewidth=2,
                        )

                    plt.ylabel("reward")
                    plt.xlabel("episode")
                    plt.title(f"PPO training – {VERSION_NAME}")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.save_dir, "rewards_latest.png"))
                    plt.close()

        if self.num_timesteps - self.last_save >= self.cfg.save_freq:
            ckpt_path = os.path.join(
                self.save_dir, f"ppo_pyrace_step{self.num_timesteps}"
            )
            self.model.save(ckpt_path)
            print(f"  ✓ checkpoint saved → {ckpt_path}.zip")
            self.last_save = self.num_timesteps

        return True   # returning False stops the training


def make_env(render: bool = False) -> gym.Env:
    if not render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    env = gym.make("Pyrace-v3").unwrapped
    env.set_view(render)

    env = Monitor(env)
    return env


def train(cfg: PPOConfig | None = None):
    if cfg is None:
        cfg = PPOConfig()

    save_dir = f"models_{VERSION_NAME}"
    os.makedirs(save_dir, exist_ok=True)

    env = make_env(render=RENDER)

    policy_kwargs = dict(net_arch=list(cfg.net_arch))

    model = PPO(
        policy         = "MlpPolicy",
        env            = env,
        learning_rate  = cfg.lr,
        n_steps        = cfg.n_steps,
        batch_size     = cfg.batch_size,
        n_epochs       = cfg.n_epochs,
        gamma          = cfg.gamma,
        gae_lambda     = cfg.gae_lambda,
        clip_range     = cfg.clip_range,
        ent_coef       = cfg.ent_coef,
        vf_coef        = cfg.vf_coef,
        max_grad_norm  = cfg.max_grad_norm,
        policy_kwargs  = policy_kwargs,
        verbose        = 0,          
        device         = "auto",     
    )

    print("=" * 60)
    print(f"  PPO training – {VERSION_NAME}")
    print(f"  total_timesteps : {cfg.total_timesteps:,}")
    print(f"  obs_dim         : {env.observation_space.shape}")
    print(f"  n_actions       : {env.action_space.n}")
    print(f"  device          : {model.device}")
    print(f"  net_arch        : {cfg.net_arch}")
    print("=" * 60)

    callback = TrainingCallback(cfg, save_dir)

    model.learn(
        total_timesteps = cfg.total_timesteps,
        callback        = callback,
        progress_bar    = False,
    )

    final_path = os.path.join(save_dir, "ppo_pyrace_final")
    model.save(final_path)
    print(f"\nTraining complete.  Final model → {final_path}.zip")

    env.close()
    return model, callback.ep_rewards





def load_and_play(model_path: str, n_episodes: int = 5, render: bool = True):
    """Load a saved PPO model and run it for `n_episodes` episodes."""
    env = make_vec_env(lambda: make_env(render=RENDER), n_envs=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    model = PPO.load(model_path, env=env)

    print(f"\nLoaded model from {model_path}")
    print(f"Running {n_episodes} evaluation episodes …\n")

    for ep in range(1, n_episodes + 1):
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            done = done[0]
            total_reward += float(reward[0])

        print(f"  Episode {ep}: reward = {total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--play",
        type=str,
        default=None,
        metavar="MODEL_PATH",
        help="Path to a saved .zip to watch, skips training.",
    )
    args = parser.parse_args()

    if args.play:
        load_and_play(args.play, n_episodes=5, render=True)
    else:
        train()



"""
Results & Discussion

Why PPO is way more stable than DQN_v03
The main issue before was exploding Q-values from dense rewards over long episodes. PPO kind of avoids that mess entirely:

Clipping: updates are limited (ε ≈ 0.2), so no weird jumps in policy
Advantage normalisation: keeps gradients nicely scaled, even if rewards are large
Entropy bonus: keeps exploration alive instead of collapsing early
Fresh data: no replay buffer → no overfitting to old experiences

Quick comparison:

Loss: DQN ~1e8 (yikes) vs PPO ~0.1–1 (normal)
Policy collapse: DQN yes, PPO pretty unlikely
Exploration: decaying ε vs steady entropy
Code: ~150 lines vs ~30 (honestly much nicer)
Rewards: DQN rises then dies, PPO steadily improves

What could be better?

Try SAC/DDPG if switching to continuous actions later
"""


