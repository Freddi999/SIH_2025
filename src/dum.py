import os
import traci
import numpy as np
from envs.multiagentenv import MultiAgentEnv

class SUMOEnv(MultiAgentEnv):
    def __init__(self, args, control_tls=True):
        super().__init__()
        self.control_tls = control_tls
        self.args = args
        self.net_file = args.env_args.get("map_path", "./maps/connaught_place.net.xml")
        self.cfg_file = args.env_args.get("cfg_path", "./maps/connaught_place.sumocfg")
        self.step_length = args.env_args.get("step_length", 1.0)
        self.decision_interval = args.env_args.get("decision_interval", 5)
        self.episode_limit = args.env_args.get("episode_limit", 720)
        self.use_gui = args.env_args.get("use_gui", False) 
        self.time = 0
        self.is_initialized = False

        # Initialize SUMO once
        self._start_sumo()
        self._initialize_env_info()
        self.cumulative_wait_time = 0.0
        self.total_vehicles_processed = 0
        self.vehicle_counts = []  # Add this to track vehicle counts

    def _start_sumo(self):
        """Start SUMO simulation with proper error handling"""
        try:
            if traci.isLoaded():
                traci.close()
        except:
            pass
        
        # Build SUMO command
        binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            binary, 
            "-c", self.cfg_file, 
            "--step-length", str(self.step_length), 
            "--no-warnings",
            "--quit-on-end"  # Important: quit when simulation ends
        ]
        
        try:
            traci.start(sumo_cmd)
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            raise

    def _initialize_env_info(self):
        """Initialize environment information after SUMO starts"""
        try:
            self.tls_ids = traci.trafficlight.getIDList()
            self.n_agents = len(self.tls_ids)

            if self.n_agents == 0:
                print("Warning: No traffic lights found in the network!")
                self.n_actions = 4  # Default fallback
                self.tls_action_counts = {}
            else:
                # Store per-agent action counts
                self.tls_action_counts = {
                    tls: len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0].phases)
                    for tls in self.tls_ids
                }
                # Global max for compatibility
                self.n_actions = max(self.tls_action_counts.values()) if self.tls_action_counts else 4

            # Initialize observation sizes
            if self.tls_ids:
                sample_features = self._build_features(self.tls_ids[0])
                self.obs_size = len(sample_features)
            else:
                self.obs_size = 13  # 4 queue + 4 speeds + 4 phase + 1 elapsed
            
            self.state_size = self.n_agents * self.obs_size
            self.is_initialized = True
            
        except Exception as e:
            print(f"Error initializing environment info: {e}")
            # Fallback values
            self.tls_ids = []
            self.n_agents = 0
            self.n_actions = 4
            self.obs_size = 13
            self.state_size = 0
            self.tls_action_counts = {}

    def reset(self):
        """Reset the simulation"""
        self.time = 0
        
        # If already initialized, just reset simulation time
        if self.is_initialized and traci.isLoaded():
            try:
                # Reset to beginning of simulation
                while traci.simulation.getTime() > 0:
                    traci.simulationStep(-1)  # Step backwards if possible
            except:
                # If we can't step backwards, restart SUMO
                self._start_sumo()
                self._initialize_env_info()
        else:
            # First time or after error, start fresh
            self._start_sumo()
            self._initialize_env_info()

        # Reset vehicle count history
        self.vehicle_counts = []

        # Get initial observations
        try:
            obs = self.get_obs()
            state = self.get_state()
            return state, obs
        except Exception as e:
            print(f"Error getting initial observations: {e}")
            # Return empty observations as fallback
            empty_obs = [np.zeros(self.obs_size) for _ in range(self.n_agents)]
            empty_state = np.zeros(self.state_size)
            return empty_state, empty_obs

    def step(self, actions):
        """Step the simulation"""
        try:
            # Check if simulation is still running
            if not traci.isLoaded():
                return self._get_terminal_state()
            
            if self.control_tls and actions is not None and self.tls_ids:
                # Apply actions (set phases)
                for i, tls in enumerate(self.tls_ids):
                    if i < len(actions):
                        # Clamp action to range for this specific TLS
                        n_actions_tls = self.tls_action_counts.get(tls, self.n_actions)
                        phase_idx = actions[i] % n_actions_tls
                        try:
                            traci.trafficlight.setPhase(tls, phase_idx)
                        except Exception as e:
                            print(f"Warning: Could not set phase for {tls}: {e}")

            # Advance SUMO simulation
            steps_to_advance = self.decision_interval if self.control_tls else 1
            
            for _ in range(steps_to_advance):
                try:
                    traci.simulationStep()
                    self.time += 1
                        
                except traci.exceptions.FatalTraCIError as e:
                    print(f"TraCI connection lost: {e}")
                    return self._get_terminal_state(done=True)
                except Exception as e:
                    print(f"Error during simulation step: {e}")
                    return self._get_terminal_state(done=True)

            # Check if simulation ended naturally AFTER stepping
            if traci.simulation.getMinExpectedNumber() <= 0 and self.time > 100:
                print("Simulation ended: no more vehicles")
                return self._get_terminal_state(done=True)

            # Get observations and compute reward
            obs = self.get_obs()
            state = self.get_state()
            reward = self._compute_reward()
            done = self.time >= self.episode_limit
            info = {"time": self.time, "vehicles": self._get_vehicle_count()}

            return state, obs, reward, done, info
            
        except Exception as e:
            print(f"Critical error in step: {e}")
            return self._get_terminal_state(done=True)

    def _get_terminal_state(self, done=True):
        """Return terminal state when simulation ends or errors occur"""
        empty_obs = [np.zeros(self.obs_size) for _ in range(self.n_agents)]
        empty_state = np.zeros(self.state_size)

        # compute once, reuse - use update_history=False for terminal state
        final_reward = self._compute_reward(update_history=False)
        print(f"Terminal reward: {final_reward}")
        return empty_state, empty_obs, final_reward, done, {"terminal": True}


    def _get_vehicle_count(self):
        """Get current vehicle count safely"""
        try:
            return traci.simulation.getMinExpectedNumber()
        except:
            return 0

    def get_obs(self):
        """Get observations for all agents"""
        obs = []
        try:
            for tls in self.tls_ids:
                features = self._build_features(tls)
                obs.append(np.array(features, dtype=np.float32))
        except Exception as e:
            print(f"Error getting observations: {e}")
            # Return zero observations as fallback
            obs = [np.zeros(self.obs_size, dtype=np.float32) for _ in range(self.n_agents)]
        
        return obs

    def get_obs_size(self):
        return self.obs_size

    def get_state(self):
        """Get global state"""
        try:
            obs = self.get_obs()
            if obs:
                return np.concatenate(obs, axis=0)
            else:
                return np.zeros(self.state_size, dtype=np.float32)
        except Exception as e:
            print(f"Error getting state: {e}")
            return np.zeros(self.state_size, dtype=np.float32)

    def get_state_size(self):
        return self.state_size

    def get_avail_actions(self, agent_id):
        if agent_id < len(self.tls_ids):
            tls = self.tls_ids[agent_id]
            n_actions = self.tls_action_counts.get(tls, self.n_actions)
            return [1] * n_actions
        return [1] * self.n_actions

    def get_total_actions(self):
        return self.n_actions

    def get_env_info(self):
        return {
            "n_agents": self.n_agents,
            "obs_shape": self.obs_size,
            "state_shape": self.state_size,
            "n_actions": self.n_actions,
            "episode_limit": self.episode_limit
        }

    def _compute_reward(self, update_history=True):
        try:
            total_wait = 0
            total_vehicles = 0
            
            for tls in self.tls_ids:
                try:
                    lanes = traci.trafficlight.getControlledLanes(tls)
                    for lane in lanes:
                        wait_time = traci.lane.getWaitingTime(lane)
                        vehicle_count = traci.lane.getLastStepVehicleNumber(lane)
                        total_wait += wait_time
                        total_vehicles += vehicle_count
                except:
                    continue
            
            # Track cumulative statistics
            if update_history:
                self.cumulative_wait_time += total_wait
                self.total_vehicles_processed += total_vehicles
            
            current_vehicles = self._get_vehicle_count()
            
            if update_history:
                self.vehicle_counts.append(current_vehicles)
                recent_counts = self.vehicle_counts[-100:]
                avg_vehicles = max(1, sum(recent_counts) / len(recent_counts))
            else:
                if self.vehicle_counts:
                    avg_vehicles = max(1, sum(self.vehicle_counts) / len(self.vehicle_counts))
                else:
                    avg_vehicles = max(1, current_vehicles)
            
            # Return instantaneous reward (for training)
            reward = -(total_wait / max(1, avg_vehicles))
            reward = np.clip(reward, -10, 0)
            
            return float(reward)
        
        except Exception as e:
            print(f"Error computing reward: {e}")
            return 0.0

# Add method to get cumulative wait time:
    def get_cumulative_wait_time(self):
        return self.cumulative_wait_time
    def sample_actions(self):
        """Sample random actions"""
        import random
        return [
            random.randrange(self.tls_action_counts.get(tls, self.n_actions))
            for tls in self.tls_ids
        ]

    def _build_features(self, tls):
        """Build feature vector for a traffic light"""
        try:
            lanes = traci.trafficlight.getControlledLanes(tls)

            # Always take exactly 4 lanes (pad with 0 if fewer)
            q_lengths = []
            avg_speeds = []
            
            for i in range(4):
                if i < len(lanes):
                    try:
                        q_lengths.append(traci.lane.getLastStepHaltingNumber(lanes[i]))
                        avg_speeds.append(traci.lane.getLastStepMeanSpeed(lanes[i]))
                    except:
                        q_lengths.append(0)
                        avg_speeds.append(0)
                else:
                    q_lengths.append(0)
                    avg_speeds.append(0)

            # Phase one-hot encoding
            try:
                phase = traci.trafficlight.getPhase(tls)
            except:
                phase = 0
                
            phase_onehot = [0] * self.n_actions
            if 0 <= phase < len(phase_onehot):
                phase_onehot[phase] = 1

            # Time until next switch
            try:
                elapsed = max(0, traci.trafficlight.getNextSwitch(tls) - traci.simulation.getTime())
            except:
                elapsed = 0

            return q_lengths + avg_speeds + phase_onehot + [elapsed]
            
        except Exception as e:
            print(f"Error building features for {tls}: {e}")
            # Return zero features as fallback
            return [0] * (4 + 4 + self.n_actions + 1)

    def close(self):
        """Close the environment"""
        try:
            if traci.isLoaded():
                traci.close()
        except Exception as e:
            print(f"Error closing SUMO: {e}")
        finally:
            self.is_initialized = False