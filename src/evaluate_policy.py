# evaluate_policy.py
import os, time
import torch
from types import SimpleNamespace
from envs.sumo_env import SUMOEnv
from qmix_models import AgentNetwork, QMixer, RegressorNetwork
import numpy as np
import torch
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
        obs = np.array(obs)
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) # [1, n_agents, obs_dim]

        done = False
        total_wait = 0.0
        steps = 0
        while not done:
            actions = policy_fn(env)
            state, next_obs, reward, done, info = env.step(actions)
            next_obs = np.array(next_obs)
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
            
            # reward = -total_wait / n_agents; to get total waiting time:
            # We can recover per-step total_wait ≈ -reward * n_agents
            # But better: call env._compute_reward() directly or use traci queries if exposed.
            steps += 1
            obs = next_obs

        # After episode finish call env._compute_reward() to get final snapshot
        # Note: _compute_reward returns -total_wait / n_agents
        final_reward = env._compute_reward()
        total_wait_est = -final_reward * max(1, env.n_agents)
        results.append(total_wait_est)
    return results

def random_policy(env):
    return env.sample_actions()
def qmix_policy(env, agent, mixer, device="cpu"):
    def policy_fn(env):
        obs = env.get_obs() 
        obs = obs.unsqueeze(0) # list of per-agent obs
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # [1, n_agents, obs_dim]

        # Forward pass each agent
        q_values = []
        for i in range(env.n_agents):
            q = agent(obs[:, i, :])  # [1, n_actions]
            q_values.append(q)
        q_values = torch.stack(q_values, dim=1)  # [1, n_agents, n_actions]

        # Pick greedy actions
        actions = q_values.argmax(dim=-1).squeeze(0).tolist()
        return actions
    return policy_fn(env)

if __name__ == "__main__":
    args = SimpleNamespace(env_args={
        "map_path": "./maps/connaught_place.net.xml",
        "cfg_path": "./maps/connaught_place.sumocfg",
        "step_length": 1.0,
        "decision_interval": 5,
        "episode_limit": 720
    })
    env = SUMOEnv(args)
    # baseline random
    random_results = evaluate(env, random_policy, n_episodes=5)
    print("Random baseline total waits:", random_results, "mean:", sum(random_results)/len(random_results))


    # After reset
    state, obs = env.reset()
    obs = np.array(obs)                 # ensures ndarray
    obs = torch.tensor(obs, dtype=torch.float32)  # [n_agents, obs_dim]
    obs_dim = obs.shape[1]
    n_agents = env.n_agents
    n_actions = env.n_actions
    state_dim = len(state)

    agent, mixer = load_models("./models", obs_dim, n_actions, n_agents, state_dim)

    trained_policy = qmix_policy(env, agent, mixer, device="cpu")
    trained_results = evaluate(env, trained_policy, n_episodes=5)
    print("QMIX policy total waits:", trained_results, "mean:", sum(trained_results)/len(trained_results))

    env.close()
