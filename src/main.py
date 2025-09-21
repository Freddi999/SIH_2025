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
    
    # Longer warmup with more diverse experiences
    warmup_steps = 10000  # Increased from 2000
    print(f"Warming up replay buffer with {warmup_steps} random steps...")

    state, obs = env.reset()
    for step in range(warmup_steps):
        actions = env.sample_actions()
        next_state, next_obs, reward, done, _ = env.step(actions)
        trainer.store_transition(state, obs, actions, reward, next_state, next_obs, done)
        state, obs = next_state, next_obs
        if done:
            state, obs = env.reset()
            
            
        # Print warmup progress
        if (step + 1) % 1000 == 0:
            print(f"Warmup progress: {step + 1}/{warmup_steps}")
    
    print("Replay buffer warm-up complete!")
    
    # Training parameters
    n_episodes = args.n_episodes
    target_update_interval = args.target_update_interval
    print_interval = args.print_interval
    
    # Better epsilon decay
    epsilon_start = getattr(args, 'epsilon_start', 1.0)
    epsilon_end = getattr(args, 'epsilon_end', 0.05)
    epsilon_decay = getattr(args, 'epsilon_decay', 15000)
    epsilon_decay_steps = epsilon_decay
    trainer.epsilon = epsilon_start
    trainer.epsilon_min = epsilon_end
    trainer.epsilon_decay = 1.0 - (1.0 / epsilon_decay)
    
    total_steps = 0
    best_reward = float('-inf')
    
    for ep in range(n_episodes):
        state, obs = env.reset()
        done = False
        total_reward = 0
        episode_steps = 0

        while not done:
            # Select actions with current epsilon
            actions = trainer.select_action(obs)

            # Step environment
            next_state, next_obs, reward, done, info = env.step(actions)

            # Store transition
            trainer.store_transition(state, obs, actions, reward, next_state, next_obs, done)

            # Train more frequently
            if len(trainer.replay_buffer) > trainer.batch_size and total_steps % 8 == 0:
                trainer.train_step()

            state, obs = next_state, next_obs
            total_reward += reward if not isinstance(reward, list) else sum(reward)
            episode_steps += 1
            total_steps += 1
            if total_steps < epsilon_decay_steps:
                trainer.epsilon = epsilon_end + (epsilon_start - epsilon_end) * (1 - total_steps / epsilon_decay_steps)
            else:
                trainer.epsilon = epsilon_end

        # Update target networks
        if ep % target_update_interval == 0:
            trainer.update_target_networks()

        # Decay epsilon more gradually
        if (ep + 1) % 500 == 0:
            torch.save(trainer.agent_q.state_dict(), f"./models/qmix_agent_ep{ep+1}.pth")
            torch.save(trainer.mixing_net.state_dict(), f"./models/qmix_mixing_ep{ep+1}.pth")
        
        # Enhanced logging
        if ep % print_interval == 0:
            avg_reward = total_reward / max(1, episode_steps)
            print(f"Episode {ep:4d} | Steps: {episode_steps:3d} | "
                  f"Total Reward: {total_reward:8.3f} | "
                  f"Avg Reward: {avg_reward:6.3f} | "
                  f"Epsilon: {trainer.epsilon:.3f}")
            
            # Save best model
            if total_reward > best_reward:
                best_reward = total_reward
                os.makedirs("./models", exist_ok=True)
                torch.save(trainer.agent_q.state_dict(), "./models/qmix_agent_best.pth")
                torch.save(trainer.mixing_net.state_dict(), "./models/qmix_mixing_best.pth")
                print(f"  → New best model saved! Reward: {total_reward:.3f}")

        # Save checkpoint models periodically
        if (ep + 1) % 200 == 0:
            os.makedirs("./models", exist_ok=True)
            torch.save(trainer.agent_q.state_dict(), f"./models/qmix_agent_ep{ep+1}.pth")
            torch.save(trainer.mixing_net.state_dict(), f"./models/qmix_mixing_ep{ep+1}.pth")
            print(f"Checkpoint saved at episode {ep+1}")

    # Save final models
    os.makedirs("./models", exist_ok=True)
    torch.save(trainer.agent_q.state_dict(), "./models/qmix_agent.pth")
    torch.save(trainer.mixing_net.state_dict(), "./models/qmix_mixing.pth")
    print("Training finished. Models saved in ./models/")

def start_simulation():
    """For evaluation with trained QMIX model"""
    args = load_args_from_yaml("./src/config/sumo_qmix.yaml")
    env = SUMOEnv(args)

    trainer = QMIXTrainer(
    env=env,
    n_agents=env.n_agents,
    state_dim=env.get_state_size(),
    obs_dim=env.obs_size,
    n_actions=env.n_actions,
    hidden_dim=args.hidden_dim,  # Add this
    device="cpu"
)
    total_steps = 0
    epsilon_start = 1.0
    epsilon_end = 0.02
    epsilon_decay_steps = 50000

    trainer.epsilon = epsilon_start
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
