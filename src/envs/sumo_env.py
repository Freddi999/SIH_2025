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

        # Start SUMO
        binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [binary, "-c", self.cfg_file, "--step-length", str(self.step_length), "--no-warnings"]
        traci.start(sumo_cmd)

        self.tls_ids = traci.trafficlight.getIDList()
        self.n_agents = len(self.tls_ids)

        # Action/obs sizes (fixed for now)
        self.n_actions = max(len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0].phases)
                             for tls in self.tls_ids)
         # Example: queue lengths, speeds, phase one-hot, elapsed time
        

    def reset(self):
        if traci.isLoaded():
            traci.close()
        traci.start(["sumo", "-c", self.cfg_file, "--step-length", str(self.step_length), "--no-warnings"])
        self.time = 0

        self.tls_ids = traci.trafficlight.getIDList()
        self.n_agents = len(self.tls_ids)

        # ⚡ Store per-agent action counts
        self.tls_action_counts = {
            tls: len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0].phases)
            for tls in self.tls_ids
        }

        # Keep global max for compatibility (used in sample_actions etc.)
        self.n_actions = max(self.tls_action_counts.values())

        # Dynamically compute obs_size
        sample_features = self._build_features(self.tls_ids[0])
        self.obs_size = len(sample_features)
        self.state_size = self.n_agents * self.obs_size

        obs = self.get_obs()
        state = self.get_state()
        return state, obs



    def step(self, actions):
        if self.control_tls and actions is not None:
        # Apply actions (set phases)
            for i, tls in enumerate(self.tls_ids):
                if i < len(actions):
                    # ⚡ Clamp action to range for this specific TLS
                    n_actions_tls = self.tls_action_counts[tls]
                    phase_idx = actions[i] % n_actions_tls
                    try:
                        traci.trafficlight.setPhase(tls, phase_idx)
                    except Exception:
                        pass

            # Advance SUMO by decision interval
            for _ in range(self.decision_interval):
                traci.simulationStep()
                self.time += 1

            obs = self.get_obs()
            state = self.get_state()
            reward = self._compute_reward()
            done = self.time >= self.episode_limit
            info = {}

            return state,obs, reward, done, info
        else:
            # If not controlling TLS, just step the simulation
            traci.simulationStep()
            self.time += 1

            obs = self.get_obs()
            state = self.get_state()
            reward = self._compute_reward()
            done = self.time >= self.episode_limit
            info = {}

            return state, obs, reward, done, info

    def get_obs(self):
        obs = []
        for tls in self.tls_ids:
            lanes = traci.trafficlight.getControlledLanes(tls)
            q_lengths = [traci.lane.getLastStepHaltingNumber(l) for l in lanes[:4]]
            avg_speeds = [traci.lane.getLastStepMeanSpeed(l) for l in lanes[:4]]
            phase = traci.trafficlight.getPhase(tls)
            phase_onehot = [0] * self.n_actions
            if phase < len(phase_onehot):
                phase_onehot[phase] = 1
            elapsed = traci.trafficlight.getNextSwitch(tls) - traci.simulation.getTime()

            features = self._build_features(tls)
            obs.append(np.array(features, dtype=np.float32))
        return obs

    def get_obs_size(self):
        return self.obs_size

    def get_state(self):
        return np.concatenate(self.get_obs(), axis=0)

    def get_state_size(self):
        return self.state_size

    def get_avail_actions(self, agent_id):
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

    def _compute_reward(self):
        """Compute reward based on traffic performance"""
        try:
            # Total waiting time across all lanes
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
            
            # Avoid division by zero
            if total_vehicles == 0:
                return 0.0
                
            # Average waiting time per vehicle (more stable reward)
            avg_wait_per_vehicle = total_wait / max(1, total_vehicles)
            
            # Use exponential scaling instead of linear
            reward = -np.exp(avg_wait_per_vehicle / 30.0) + 1.0
            
            # Less aggressive clipping
            reward = np.clip(reward, -5.0, 1.0)
            
            return reward
            
        except Exception as e:
            print(f"Error computing reward: {e}")
            return 0.0

    def sample_actions(self):
        import random
        return [
            random.randrange(self.tls_action_counts[tls])
            for tls in self.tls_ids
        ]

    def _build_features(self, tls):
        lanes = traci.trafficlight.getControlledLanes(tls)

        # Always take exactly 4 lanes (pad with 0 if fewer)
        q_lengths = [traci.lane.getLastStepHaltingNumber(l) for l in lanes[:4]]
        q_lengths += [0] * (4 - len(q_lengths))

        avg_speeds = [traci.lane.getLastStepMeanSpeed(l) for l in lanes[:4]]
        avg_speeds += [0] * (4 - len(avg_speeds))

        # Always one-hot of global max
        phase = traci.trafficlight.getPhase(tls)
        phase_onehot = [0] * self.n_actions
        if phase < len(phase_onehot):
            phase_onehot[phase] = 1

        elapsed = traci.trafficlight.getNextSwitch(tls) - traci.simulation.getTime()

        return q_lengths + avg_speeds + phase_onehot + [elapsed]

    def close(self):
        if traci.isLoaded():
            traci.close()
