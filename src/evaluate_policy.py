import os, time
import torch
from types import SimpleNamespace
from envs.sumo_env import SUMOEnv
from qmix_models import AgentNetwork, QMixer, RegressorNetwork
import numpy as np

def load_models(model_dir, obs_dim, n_actions, n_agents, state_dim):
    agent = AgentNetwork(obs_dim, n_actions)   # reconstruct same arch as training
    mixer = QMixer(n_agents, state_dim)        # reconstruct mixer

    agent.load_state_dict(torch.load(os.path.join(model_dir, "qmix_agent.pth"), map_location="cpu"))
    mixer.load_state_dict(torch.load(os.path.join(model_dir, "qmix_mixing.pth"), map_location="cpu"))

    agent.eval()
    mixer.eval()
    return agent, mixer

def evaluate(env, policy_fn, n_episodes=5):
    results = []
    for ep in range(n_episodes):
        state,obs = env.reset()
        done = False
        while not done:
            actions = policy_fn(env)
            state, obs, reward, done, info = env.step(actions)

        final_reward = env._compute_reward()
        total_wait_est = -final_reward * max(1, env.n_agents)
        results.append(total_wait_est)
    return results

def random_policy(env):
    return env.sample_actions()

def default_policy(env):
    """Run SUMO with its default built-in traffic light program"""
    return None

def qmix_policy(env, agent, mixer, device="cpu"):
    def policy_fn(env):
        obs = env.get_obs()
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # [1, n_agents, obs_dim]

        q_values = []
        for i in range(env.n_agents):
            q = agent(obs[:, i, :])  # [1, n_actions]
            q_values.append(q)
        q_values = torch.stack(q_values, dim=1)  # [1, n_agents, n_actions]

        actions = q_values.argmax(dim=-1).squeeze(0).tolist()
        return actions
    return policy_fn

if __name__ == "__main__":
    args = SimpleNamespace(env_args={
        "map_path": "./maps/connaught_place.net.xml",
        "cfg_path": "./maps/connaught_place.sumocfg",
        "step_length": 1.0,
        "decision_interval": 5,
        "episode_limit": 720
    })
    env = SUMOEnv(args)

    # Reset once to extract dims
    state, obs = env.reset()
    obs = np.array(obs)
    obs = torch.tensor(obs, dtype=torch.float32)
    obs_dim = obs.shape[1]
    n_agents = env.n_agents
    n_actions = env.n_actions
    state_dim = len(state)

    # Default SUMO baseline
    # SUMO built-in default (no RL control)

    # Random baseline
    random_results = evaluate(env, random_policy, n_episodes=5)
    print("Random baseline total waits:", random_results, "mean:", sum(random_results)/len(random_results))

    # Trained QMIX
    agent, mixer = load_models("./models", obs_dim, n_actions, n_agents, state_dim)
    trained_policy = qmix_policy(env, agent, mixer, device="cpu")
    trained_results = evaluate(env, trained_policy, n_episodes=5)
    print("QMIX policy total waits:", trained_results, "mean:", sum(trained_results)/len(trained_results))

    env.close()
    env_default = SUMOEnv(args, control_tls=False)
    default_results = evaluate(env_default, lambda e: None, n_episodes=5)

    print("Default SUMO total waits:", default_results, "mean:", sum(default_results)/len(default_results))
    env_default.close()