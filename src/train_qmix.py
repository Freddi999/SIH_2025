# train_qmix.py
import os, time, math
import numpy as np
from types import SimpleNamespace
import torch

from envs.sumo_env import SUMOEnv
# If you already have qmix_models.py, qmix_trainer.py, replay_buffer.py, import them:
from algorithms.qmix_trainer import QMIXTrainer

def make_args():
    return SimpleNamespace(env_args={
        "map_path": "./maps/connaught_place.net.xml",
        "cfg_path": "./maps/connaught_place.sumocfg",
        "step_length": 1.0,
        "decision_interval": 5,
        "episode_limit": 720
    })

def profile_one_episode(env, trainer, n_steps=200):
    """Quick run to measure time per sim step and train step - returns seconds per env step and per train_step call."""
    obs = env.reset()
    state = env.get_state()
    t0 = time.time()
    steps = 0
    while steps < n_steps:
        actions = trainer.select_actions(obs, epsilon=1.0)  # random-ish
        next_obs, reward, done, info = env.step(actions)
        next_state = env.get_state()
        trainer.store_transition(obs, state, actions, reward, next_obs, next_state, done)
        trainer.train_step()
        obs = next_obs
        state = next_state
        steps += 1
        if done: break
    sec = time.time() - t0
    env.close()
    return sec / max(1, steps)  # seconds per step

def main():
    args = make_args()
    env = SUMOEnv(args)
    trainer = QMIXTrainer(env, cfg={"batch_size":32, "gamma":0.99, "lr":5e-4, "reg_lr":1e-3, "target_update":200})

    print("Profiling: running short episode to estimate time...")
    sp = profile_one_episode(env, trainer, n_steps=100)
    print(f"Measured approx {sp:.4f} seconds per environment step (includes traci.simulationStep & training overhead).")
    print("Use this to estimate full training cost. Now starting training...")

    # Training hyperparams
    n_episodes = 500       # lower for initial debug; increase when happy
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 10000

    total_steps = 0
    for ep in range(n_episodes):
        obs = env.reset()
        state = env.get_state()
        done = False
        ep_reward = 0
        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * total_steps / epsilon_decay)
            actions = trainer.select_actions(obs, epsilon=epsilon)
            next_obs, reward, done, info = env.step(actions)
            next_state = env.get_state()
            trainer.store_transition(obs, state, actions, reward, next_obs, next_state, done)

            res = trainer.train_step()
            if res is not None and total_steps % 500 == 0:
                print(f"[Train] step {total_steps} ep {ep+1}: td_loss={res['td_loss']:.4f} reg_loss={res['reg_loss']:.4f}")

            obs = next_obs
            state = next_state
            ep_reward += reward
            total_steps += 1

        print(f"Episode {ep+1}/{n_episodes} finished. reward={ep_reward:.4f}")
        if (ep + 1) % 50 == 0:
            trainer.save_models(tag=f"ep{ep+1}")

    trainer.save_models(tag="final")
    env.close()

if __name__ == "__main__":
    main()
