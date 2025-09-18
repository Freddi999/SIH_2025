import os
from types import SimpleNamespace
from envs.sumo_env import SUMOEnv
import time
import traci

def run_sumo_gui(control_tls=True):
    """Run SUMO with GUI - improved version"""
    args = SimpleNamespace(env_args={
        "map_path": "./maps/connaught_place.net.xml",
        "cfg_path": "./maps/connaught_place.sumocfg",
        "step_length": 1.0,
        "decision_interval": 5,
        "episode_limit": 720,
        "use_gui": True
    })

    env = None
    try:
        print("Starting SUMO environment...")
        env = SUMOEnv(args, control_tls=control_tls)
        
        print(f"Environment initialized with {env.n_agents} traffic lights")
        print("SUMO GUI is running. The simulation will run automatically.")
        print("Close the GUI window to stop the simulation early.")
        
        # Reset environment
        state, obs = env.reset()
        print(f"Initial state shape: {len(state)}, obs count: {len(obs)}")
        
        step_count = 0
        max_steps = args.env_args["episode_limit"]
        
        while step_count < max_steps:
            try:
                # Check if we still have a connection
                if not traci.isLoaded():
                    print("SUMO connection lost (GUI closed by user)")
                    break
                
                # Generate actions (you can modify this logic)
                if control_tls and env.n_agents > 0:
                    actions = env.sample_actions()  # Random actions
                else:
                    actions = None
                
                # Step the environment
                state, obs, reward, done, info = env.step(actions)
                step_count += 1
                
                # Print progress periodically
                if step_count % 100 == 0:
                    vehicles = info.get('vehicles', 0)
                    print(f"Step {step_count}/{max_steps}, Vehicles: {vehicles}, Reward: {reward:.3f}")
                
                # Check if simulation should end
                if done:
                    print(f"Episode completed at step {step_count}")
                    break
                
                # Small delay to make GUI more watchable (adjust as needed)
                time.sleep(0.02)  # 50 FPS
                
            except traci.exceptions.FatalTraCIError:
                print(f"TraCI connection lost at step {step_count}")
                break
            except KeyboardInterrupt:
                print("Simulation interrupted by user")
                break
            except Exception as e:
                print(f"Error at step {step_count}: {e}")
                break
        
        print(f"Simulation finished after {step_count} steps")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        if env is not None:
            try:
                env.close()
                print("Environment closed successfully")
            except:
                print("Warning: Could not close environment properly")

def run_sumo_headless(control_tls=True, duration=200):
    """Run SUMO without GUI for faster testing"""
    args = SimpleNamespace(env_args={
        "map_path": "./maps/connaught_place.net.xml",
        "cfg_path": "./maps/connaught_place.sumocfg",
        "step_length": 1.0,
        "decision_interval": 5,
        "episode_limit": duration,
        "use_gui": False  # No GUI for speed
    })

    env = None
    try:
        print("Starting headless SUMO simulation...")
        env = SUMOEnv(args, control_tls=control_tls)
        
        state, obs = env.reset()
        
        for step in range(duration):
            actions = env.sample_actions() if control_tls else None
            state, obs, reward, done, info = env.step(actions)
            
            if step % 50 == 0:
                vehicles = info.get('vehicles', 0)
                print(f"Step {step}, Vehicles: {vehicles}, Reward: {reward:.3f}")
            
            if done:
                break
        
        print("Headless simulation completed")
        
    except Exception as e:
        print(f"Error in headless simulation: {e}")
    finally:
        if env:
            env.close()

if __name__ == "__main__":
    print("=== SUMO Environment Test ===")
    print("1. Testing headless simulation first...")
    
    # Test headless first to check basic functionality
    run_sumo_headless(control_tls=False, duration=100)
    
    print("\n" + "="*50)
    print("2. Running GUI simulation with traffic light control...")
    
    # Then run with GUI
    run_sumo_gui(control_tls=True)
    
    print("\nAll tests completed!")