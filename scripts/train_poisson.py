"""Training run with poisson_crew=True.

Config:
    total_timesteps = 150,000
    n_envs          = 4
    eval_freq       = 50,000   (3 eval snapshots: 50k, 100k, 150k)
    n_eval_episodes = 2
    poisson_crew    = True
    output dir      = runs/poisson_run/

Progress is printed to stdout every 5 minutes of wall-clock time.
"""

from __future__ import annotations

import copy
import os
import sys
import time
from dataclasses import replace

import numpy as np

# Make sure the project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from habitat_sim.config import reference_config, MotorConfig, RLConfig
from habitat_sim.control.training import _make_curriculum_callback


# ---------------------------------------------------------------------------
# Wall-clock progress callback
# ---------------------------------------------------------------------------

def _make_progress_callback(total_timesteps: int, interval_secs: float = 300.0):
    """SB3 callback that prints a status line every `interval_secs` wall-clock seconds."""
    from stable_baselines3.common.callbacks import BaseCallback

    class ProgressCallback(BaseCallback):
        def __init__(self, total_ts: int, interval: float):
            super().__init__(verbose=0)
            self._total_ts = total_ts
            self._interval = interval
            self._next_print = time.time() + interval
            self._t_start = time.time()

        def _on_step(self) -> bool:
            now = time.time()
            if now >= self._next_print:
                elapsed = now - self._t_start
                ts = self.num_timesteps
                pct = 100.0 * ts / self._total_ts
                rate = ts / elapsed if elapsed > 0 else 0.0
                remaining = (self._total_ts - ts) / rate if rate > 0 else 0.0
                print(
                    f"\n[Progress] {ts:>7,}/{self._total_ts:,} steps "
                    f"({pct:.1f}%)  |  "
                    f"elapsed {elapsed/60:.1f} min  |  "
                    f"ETA {remaining/60:.1f} min  |  "
                    f"rate {rate:.0f} steps/s",
                    flush=True,
                )
                self._next_print = now + self._interval
            return True

    return ProgressCallback(total_timesteps, interval_secs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from habitat_sim.control.sac_agent import build_vec_env, build_sac
    from habitat_sim.environment.habitat_env import HabitatEnv

    t_wall_start = time.time()

    # --- Config ---
    cfg = reference_config()
    cfg.stochastic = replace(cfg.stochastic, poisson_crew=True)
    cfg.rl = replace(
        cfg.rl,
        total_timesteps=150_000,
        n_envs=4,
        eval_freq=50_000,
        n_eval_episodes=2,
        checkpoint_freq=50_000,
        log_dir="runs/poisson_run",
        curriculum=True,
    )

    total_timesteps = cfg.rl.total_timesteps
    log_dir = cfg.rl.log_dir
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("  SAC Training — poisson_crew=True")
    print(f"  total_timesteps : {total_timesteps:,}")
    print(f"  n_envs          : {cfg.rl.n_envs}")
    print(f"  eval_freq       : {cfg.rl.eval_freq:,}  ({total_timesteps // cfg.rl.eval_freq} snapshots)")
    print(f"  n_eval_episodes : {cfg.rl.n_eval_episodes}")
    print(f"  output          : {log_dir}/")
    print(f"  estimated time  : ~56 min")
    print("=" * 60, flush=True)

    rl = cfg.rl

    # Training env (motor off — agent controls pumps only)
    train_cfg = copy.deepcopy(cfg)
    train_cfg.motor = MotorConfig(profile="off")
    train_env = build_vec_env(train_cfg, n_envs=rl.n_envs, seed=cfg.seed)

    # Eval env (single, deterministic)
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.motor = MotorConfig(profile="off")
    eval_env = DummyVecEnv([lambda: HabitatEnv(config=copy.deepcopy(eval_cfg))])

    # Model
    try:
        import torch.utils.tensorboard  # noqa: F401
        tb_log = os.path.join(log_dir, "tb")
    except ImportError:
        tb_log = None
    model = build_sac(train_env, rl, seed=cfg.seed, tensorboard_log=tb_log)

    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=max(rl.eval_freq // rl.n_envs, 1),
        n_eval_episodes=rl.n_eval_episodes,
        deterministic=True,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(rl.checkpoint_freq // rl.n_envs, 1),
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="sac_ckpt",
        verbose=1,
    )
    curriculum_cb = _make_curriculum_callback(cfg, total_timesteps)
    progress_cb = _make_progress_callback(total_timesteps, interval_secs=300.0)

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([eval_cb, checkpoint_cb, curriculum_cb, progress_cb]),
        log_interval=10,
        reset_num_timesteps=True,
        progress_bar=False,
    )

    final_path = os.path.join(log_dir, "final_model")
    model.save(final_path)

    elapsed = time.time() - t_wall_start
    print(f"\n{'='*60}")
    print(f"  Training complete in {elapsed/60:.1f} min")
    print(f"  Final model : {final_path}.zip")
    print(f"  Best model  : {log_dir}/best_model/best_model.zip")
    print(f"{'='*60}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
