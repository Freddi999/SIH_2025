import os
import sys
import yaml
from types import SimpleNamespace
import torch

from envs.sumo_env import SUMOEnv
from algorithms.qmix_trainer import QMIXTrainer


def load_args_from_yaml(yaml_path: str):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)


def start_training():
    # Load args from YAML
    args = load_args_from_yaml("./src/config/sumo_qmix.yaml")

    # Create environment
    env = SUMOEnv(args)
    # In start_training(), after creating env and trainer
    

    env.reset()
    # Define QMIX trainer
    trainer = QMIXTrainer(
        n_agents=env.n_agents,
        env=env,
        state_dim=env.get_state_size(),
        obs_dim=env.obs_size,
        n_actions=env.n_actions,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    warmup_steps = 2000
    print(f"Warming up replay buffer with {warmup_steps} random steps...")

    state, obs = env.reset()
    done = False
    for _ in range(warmup_steps):
        actions = env.sample_actions()  # random actions
        next_state, next_obs, reward, done, _ = env.step(actions)
        trainer.store_transition(state, obs, actions, reward, next_state, next_obs, done)
        state, obs = next_state, next_obs
        if done:
            state, obs = env.reset()
    print("Replay buffer warm-up complete!")
    n_episodes = args.n_episodes
    target_update_interval = args.target_update_interval
    print_interval = args.print_interval

    for ep in range(n_episodes):
        state, obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Select actions
            actions = trainer.select_action(obs)

            # Step environment
            next_state, next_obs, reward, done, info = env.step(actions)

            # Store transition
            trainer.store_transition(state, obs, actions, reward, next_state, next_obs, done)

            # Train only if we have enough samples
            if len(trainer.replay_buffer) > trainer.batch_size:
                trainer.train_step()

            state, obs = next_state, next_obs
            total_reward += reward if not isinstance(reward, list) else sum(reward)

        # Update target networks periodically
        if ep % target_update_interval == 0:
            trainer.update_target_networks()

        if ep % print_interval == 0:
            print(f"Episode {ep}, Total Reward: {total_reward}, Epsilon: {trainer.epsilon:.2f}")
        # Decay epsilon
        trainer.epsilon = max(trainer.epsilon * trainer.epsilon_decay, trainer.epsilon_min)


    # Save trained models
    os.makedirs("./models", exist_ok=True)
    torch.save(trainer.agent_q.state_dict(), "./models/qmix_agent.pth")
    torch.save(trainer.mixing_net.state_dict(), "./models/qmix_mixing.pth")
    print("Training finished. Models saved in ./models/")


def start_simulation():
    """For evaluation with trained QMIX model"""
    args = load_args_from_yaml("./src/config/sumo_qmix.yaml")
    env = SUMOEnv(args)

    trainer = QMIXTrainer(
        n_agents=env.n_agents,
        state_dim=env.get_state_size(),
        obs_dim=env.obs_dim,
        n_actions=env.n_actions,
        device="cpu"
    )

    # Load trained models
    trainer.agent_q.load_state_dict(torch.load("./models/qmix_agent.pth", map_location="cpu"))
    trainer.mixing_net.load_state_dict(torch.load("./models/qmix_mixing.pth", map_location="cpu"))

    n_eval_episodes = 5
    for ep in range(n_eval_episodes):
        state, obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            actions = trainer.select_action(obs)
            next_state, next_obs, reward, done, info = env.step(actions)
            state, obs = next_state, next_obs
            total_reward += reward if not isinstance(reward, list) else sum(reward)

        print(f"[EVAL] Episode {ep + 1} Total reward: {total_reward}")
