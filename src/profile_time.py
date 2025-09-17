# profile_time.py
import time
from types import SimpleNamespace
from envs.sumo_env import SUMOEnv
from algorithms.qmix_trainer import QMIXTrainer

def run_profile():
    args = SimpleNamespace(env_args={
        "map_path": "./maps/connaught_place.net.xml",
        "cfg_path": "./maps/connaught_place.sumocfg",
        "step_length": 1.0,
        "decision_interval": 5,
        "episode_limit": 720
    })
    env = SUMOEnv(args)
    trainer = QMIXTrainer(env)
    t0 = time.time()
    obs = env.reset()
    state = env.get_state()
    steps = 0
    while steps < 200:
        actions = trainer.select_actions(obs, epsilon=1.0)
        next_obs, reward, done, info = env.step(actions)
        trainer.store_transition(obs, state, actions, reward, next_obs, env.get_state(), done)
        trainer.train_step()
        obs = next_obs
        steps += 1
        if done: break
    t1 = time.time()
    env.close()
    print(f"Ran {steps} steps in {t1-t0:.2f}s -> {(t1-t0)/max(1,steps):.4f}s per step")

if __name__ == "__main__":
    run_profile()
