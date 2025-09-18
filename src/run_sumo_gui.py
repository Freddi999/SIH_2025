import os
from types import SimpleNamespace
from envs.sumo_env import SUMOEnv
import time

def run_sumo_gui(control_tls=True):
    args = SimpleNamespace(env_args={
        "map_path": "./maps/connaught_place.net.xml",
        "cfg_path": "./maps/connaught_place.sumocfg",
        "step_length": 1.0,
        "decision_interval": 5,
        "episode_limit": 720,
        "use_gui": True  # Enable GUI for visualization
    })

    env = SUMOEnv(args, control_tls=control_tls)
    print("SUMO GUI is running. Close the GUI to terminate.")

    try:
        env.reset()  # Start the simulation
        for _ in range(720):  # Step the simulation for 720 steps (12 minutes at 1-second steps)
            env.step([0] * env.n_agents)  # Default action (no-op or random actions)
            #time.sleep(0.1)  # Slow down the simulation for better visualization
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        env.close()  # Ensure proper cleanup

if __name__ == "__main__":
    print("1. Running Default SUMO Simulation...")
    run_sumo_gui(control_tls=False)

    print("2. Running SUMO Simulation with QMIX Control...")
    run_sumo_gui(control_tls=True)
