"""
We use Ant V-5 robot from gymnasium. This file acts as a wrapper around the enviornemnt.
We use this to define custom reward functions.

Velocity-Constrained Ant Environment with Natural Gait Reward
    1. Walk stright ahead with a fixed velocity
    2. Replace the default reward with a multi-component reward that
        encourages natural quadruped locomotion at the commanded velocity.
    3. Domain Randomization
"""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

def quat_to_rpy(quat):
    """
    Converts Quaternion (w, x, y, z) -> (roll, pitch, yaw) in radians.
    """
    w, x, y, z = quat
    sinr = 2.0 * (w*x + y*x)
    cosr = 1.0 - 2.0*(x*x + y*y)
    roll = np.arctan2(sinr, cosr)
    
    sinp = np.clip(2.0 * (w*y - z*x), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny = 2.0 * (w*z + x*y)
    cosy = 1.0 - 2.0*(y*y + z*z)
    yaw = np.arctan2(siny, cosy)

    return roll, pitch, yaw

class VelocityAntEnv(gym.Wrapper):
    """
    Gymnasium Ant with velocity command conditioning and a natural gait reward.

    Observation layout: [base_obs (27)]
    Action Space: Unchanged (8 continous torques)

    Walk straight ahead at a fixed speed

    Domain Randomization:
    - Total Mass scaling
    - Tangential Friction Scaling
    - Action Transport delay
    - Observation transport delay
    """

    HEALTHY_Z_RANGE = (0.3, 1.0)
    TARGET_HEIGHT = 0.57
    TARGET_VX = 1.0

    # --------------------------------------------------------------------------------
    # Reward Weights.
    W_FORWARD = 3.0
    W_LATERAL = 1.0
    W_VZ = 0.2
    W_HEIGHT = 0.3
    W_ORIENT = 0.3
    W_ENERGY_TORQUE = 0.003
    W_ENERGY_JVEL = 0.0003
    W_SMOOTH = 0.04
    W_SYMMETRY = 0.03
    W_ALIVE = 0.5
    W_STAND_PENALTY=0.8

    ACTION_FILTER_ALPHA=0.5

    def __init__(
            self,
            render_mode = None,
            max_episode_length=1000,
            randomize_mass = False,
            mass_scale_range=(0.9, 1.1),
            randomize_friction = False,
            friction_scale_range=(0.7, 1.3),
            randomize_action_delay = False,
            action_delay_range=(0, 2),
            randomize_obs_delay=False,
            obs_delay_range=(0,0),
            randomization_seed=42,
    ):
        base_env = gym.make(
            "Ant-v4",
            use_contact_forces=False,
            terminate_when_unhealthy=True,
            healthy_z_range=self.HEALTHY_Z_RANGE,
            render_mode=render_mode
        )
        super().__init__(base_env)

        self.max_episode_length = max_episode_length
        
        # Domain Randomization Config
        self._rng = np.random.default_rng(randomization_seed)
        self.randomize_mass = randomize_mass
        self.mass_scale_range = tuple(mass_scale_range)
        self.randomize_friction = randomize_friction
        self.friction_scale_range = tuple(friction_scale_range)
        self.randomize_action_delay = randomize_action_delay
        self.action_delay_range = tuple(int(x) for x in action_delay_range)
        self.randomize_obs_delay = randomize_obs_delay
        self.obs_delay_range = tuple(int(x) for x in obs_delay_range)

        # Cache nominal MuJuCo model properties so randomization ever compoints
        model = self.env.unwrapped.model
        self._nominal_body_mass = model.body_mass.copy()
        self._nominal_geom_friction = model.geom_friction.copy()

        # DR bookkeeping. Set Properly on every reset
        self._action_delay_steps = 0
        self._obs_delay_steps = 0
        self._action_buffer: deque = deque()
        self.obs_buffer: deque = deque()
        self._dr_info = {
            "mass_scale": 1.0,
            "friction_scale": 1.0,
            "action_delay_steps": 0,
            "obs_delay_steps": 0,
        }

        # Observation space stays at base 27 dims. 
        base_obs_dim = self.observation_space.shape[0] # 27
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_obs_dim,),
            dtype=np.float32)

        self._prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self._step_count = 0
        self._episode_return = 0.0

    # --------------------------------------------------------------------------------
    # Domain Randomization

    def _apply_domain_randomization(self):
        model = self.env.unwrapped.model

        # Always restore nominal values first
        model.body_mass[:] = self._nominal_body_mass
        model.geom_friction[:] = self._nominal_geom_friction

        mass_scale = 1.0
        friction_scale = 1.0
        action_delay = 0
        obs_delay = 0

        if self.randomize_mass:
            mass_scale = float(self._rng.uniform(*self.mass_scale_range))
            model.body_mass[1:] = self._nominal_body_mass[1:] * mass_scale

        if self.randomize_friction:
            friction_scale = float(self._rng.uniform(*self.friction_scale_range))
            model.geom_friction[:, 0] = self._nominal_geom_friction[:, 0] * friction_scale

        if self.randomize_action_delay:
            action_delay = int(self._rng.integers(
                self.action_delay_range[0], self.action_delay_range[1] + 1,
            ))

        if self.randomize_obs_delay:
            obs_delay = int(self._rng.integers(
                self.obs_delay_range[0], self.obs_delay_range[1] + 1,
            ))

        self._action_delay_steps = action_delay
        self._obs_delay_steps = obs_delay
        self._dr_info = {
            "mass_scale": mass_scale,
            "friction_scale": friction_scale,
            "action_delay_steps": action_delay,
            "obs_delay_steps": obs_delay,
        }

    def _reset_delay_buffer(self, initial_obs):
        zero_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self._action_buffer = deque(
            [zero_action.copy() for _ in range(self._action_delay_steps + 1)],
            maxlen=self._action_delay_steps + 1,
        )
        self._obs_buffer = deque(
            [initial_obs.copy() for _ in range(self._obs_delay_steps + 1)],
            maxlen=self._obs_delay_steps + 1,
        )
        
    # --------------------------------------------------------------------------------
    # Gym API
    
    def reset(self, **kwargs):
        self._apply_domain_randomization()
        obs, info = self.env.reset(**kwargs)
        self._prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self._step_count = 0

        obs = obs.astype(np.float32)
        self._reset_delay_buffer(obs)

        info.update(self._dr_info)
        return self._obs_buffer[0].copy(), info
    
    def step(self, action):
        self._action_buffer.append(np.asarray(action, dtype=np.float32).copy())
        command = self._action_buffer[0]
        
        filtered = (self.ACTION_FILTER_ALPHA * command
                    + (1.0 - self.ACTION_FILTER_ALPHA) * self._prev_action)
        
        obs, _reward, terminated, truncated, info = self.env.step(filtered)
        self._step_count += 1

        reward = self._compute_reward(obs, filtered)
        self._prev_action = filtered.copy()
        self._episode_return += reward

        # Exit after reaching max number of steps in an episode
        if self._step_count >= self.max_episode_length:
            truncated = True

        obs = obs.astype(np.float32)
        self._obs_buffer.append(obs.copy())

        vx, vy = obs[13], obs[14]
        info["forward_speed"] = float(vx)
        info["lateral_speed"] = float(vy)
        info["velocity_error"] = float(np.abs(vx - self.TARGET_VX))

        if terminated or truncated:
            info["episode_return"] = self._episode_return

        return self._obs_buffer[0].copy(), reward, terminated, truncated, info
    

    # --------------------------------------------------------------------------------
    # Custom Reward Function

    def _compute_reward(self, obs, action):
        """
        Multi-componnt rward promoting natural velocity tracking gait

        Observation indices (Ant V5, user_contact_forces=False, 27-dim):
            0       : Z Height
            1-4     : Quaternion (w,x,y,z)
            5-12    : 8 Joint Angles
            13-15   : Linear Velocity (vx, vy, vz) in world frame
            16-18   : Angular velocity (wx, wy,wz) in world frame
            19-26   : 8 Joint velocity
        """

        z = obs[0]
        quat = obs[1:5]
        joint_vel = obs[19:27]
        vx, vy, vz = obs[13], obs[14], obs[15]
        yaw_rate = obs[18]

        roll, pitch, _ = quat_to_rpy(quat)

        # 1. Velocity Tracking.
        r_forward = self.W_FORWARD * np.exp(-2.0 * (vx - self.TARGET_VX) ** 2)
        r_lateral = -self.W_LATERAL * vy ** 2
        r_vz = self.W_VZ * vz ** 2 

        # 2. Posture
        r_height = self.W_HEIGHT *  np.exp(-40.0 * (z - self.TARGET_HEIGHT) **2)
        r_orient = self.W_ORIENT * np.exp(-5.0 * (roll **2 + pitch **2))

        # 3. Energy Efficiency
        r_energy = (
            -self.W_ENERGY_TORQUE * np.sum(action **2)
            - self.W_ENERGY_JVEL * np.sum(joint_vel **2)
        )

        # 4. Action Smoothness (Penalise Jerks)
        r_smooth = -self.W_SMOOTH * np.sum((action * self._prev_action) **2)

        # 5. Gait Symmetry: All 4 legs should share work evenly.
        #       Leg activity = |hip_vel| + |ankle_vel| per leg
        leg_activity = np.array([
            np.abs(joint_vel[0]) + np.abs(joint_vel[1]),
            np.abs(joint_vel[2]) + np.abs(joint_vel[3]),
            np.abs(joint_vel[4]) + np.abs(joint_vel[5]),
            np.abs(joint_vel[6]) + np.abs(joint_vel[7]),
        ])
        r_symmetry = -self.W_SYMMETRY * np.std(leg_activity)

        # 6. Alive Bonus
        r_alive = self.W_ALIVE

        # 7. Stand Penalty
        speed_xy = np.sqrt(vx**2 + vy**2)
        r_stand = -self.W_STAND_PENALTY * np.exp(-10.0 * speed_xy)


        # Final Reward
        reward = (
            r_forward
            + r_lateral
            + r_vz
            + r_height
            + r_orient
            + r_energy
            + r_smooth
            + r_symmetry
            + r_alive
            + r_stand
        )

        return float(reward)