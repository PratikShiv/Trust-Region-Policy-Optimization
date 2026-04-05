"""
We use Ant V-5 robot from gymnasium. This file acts as a wrapper around the enviornemnt.
We use this to define custom reward functions.

Velocity-Constrained Ant Environment with Natural Gait Reward
    1. Append a velocity command (Vx, Vy, Vz) to the observation
    2. Replace the default reward with a multi-component reward that
        encourages natural quadruped locomotion at the commanded velocity.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

def quat_to_rpy(quat):
    """
    Converts Quaternion (w, x, y, z) -> (roll, pitch, yaw) in radians.
    """
    w, x, y, z = quat
    sinr = 2.0 * (w*x + y*x)
    cosr = 1.9 - 2.0*(x*x + y*y)
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

    Observation layout: [base_obs (27), cmd_vx, cmd_vy, cmd_vz, cmd_yaw_rate]
    Action Space: Unchanged (8 continous torques)

    Velocity commands are re-sampled every ''cmd_change_steps''' steps so the
    policy learns to track a range of velocities rather than a single one.
    """

    HEALTHY_Z_RANGE = (0.3, 1.0)
    TARGET_HEIGHT = 0.57

    # --------------------------------------------------------------------------------
    # Reward Weights.
    W_VEL_XY = 1.5
    W_YAW = 0.5
    W_VZ = 0.3
    W_HEIGHT = 0.5
    W_ORIENT = 0.3
    W_ENERGY_TORQUE = 0.005
    W_ENERGY_JVEL = 0.0005
    W_SMOOTH = 0.15
    W_SYMMETRY = 0.05
    W_ALIVE = 0.5
    ACTION_FILTER_ALPHA=0.2

    def __init__(
            self,
            render_mode = None,
            max_episode_length=1000,
            cmd_change_steps=200,
    ):
        base_env = gym.make(
            "Ant-v5",
            terminate_when_unhealthy=True,
            healthy_z_range=self.HEALTHY_Z_RANGE,
            render_mode=render_mode
        )
        super().__init__(base_env)

        self.max_episode_length = max_episode_length
        self.cmd_change_steps = cmd_change_steps

        base_obs_dim = self.observation_space.shape[0] # 27
        low = np.full(base_obs_dim + 4, -np.inf, dtype=np.float32)
        high = np.full(base_obs_dim + 4, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._cmd_vel = np.zeros(4, dtype=np.float32)
        self._prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self._step_count = 0

    # --------------------------------------------------------------------------------
    # Velocity Command Sampling

    def _sample_velocity_command(self):
        # Sample a random direction
        vx = np.random.uniform(0.3, 1.0)
        vy = np.random.uniform(-0.3, 0.3)
        return np.array([
            vx,   # Vx (Forward / Backward)
            vy,   # Vy (Lateral)
            0.0,                            # Vz
            np.random.uniform(-0.3, 0.3),   # Yaw Rate
        ], dtype=np.float32)
    

    # --------------------------------------------------------------------------------
    # Observation Augmentation
    
    def _augment_obs(self, obs):
        return np.concatenate([obs, self._cmd_vel]).astype(np.float32)
    

    # --------------------------------------------------------------------------------
    # Gym API
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._cmd_vel = self._sample_velocity_command()
        self._prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self._step_count = 0
        return self._augment_obs(obs), info
    
    def step(self, action):
        filtered = (self.ACTION_FILTER_ALPHA * action
                    + (1.0 - self.ACTION_FILTER_ALPHA) * self._prev_action)
        obs, _reward, terminated, truncated, info = self.env.step(filtered)
        self._step_count += 1

        # Change command velocity every 'cmd_change_steps' to allow the NN to learn various input commands.
        if self._step_count % self.cmd_change_steps == 0:
            self._cmd_vel = self._sample_velocity_command()

        reward = self._compute_reward(obs, filtered)
        self._prev_action = filtered.copy()

        # Exit after reaching max number of steps in an episode
        if self._step_count >= self.max_episode_length:
            truncated = True

        info["cmd_vel"] = self._cmd_vel.copy()
        info["velocity_error_xy"] = np.sqrt(
            (obs[13] - self._cmd_vel[0]) ** 2 + (obs[14] - self._cmd_vel[1]) **2
        )

        return self._augment_obs(obs), reward, terminated, truncated, info
    

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
        cmd_vx, cmd_vy, cmd_vz, cmd_yaw = self._cmd_vel

        roll, pitch, _ = quat_to_rpy(quat)

        # 1. Velocity Tracking.
        vel_err_xy = np.sqrt((vx - cmd_vx)**2 + (vy - cmd_vy)**2)
        r_vel_xy = np.exp(-vel_err_xy)
        r_yaw = np.exp(-np.abs(yaw_rate - cmd_yaw))
        r_vz = np.exp(np.abs(vz))

        # 2. Posture
        r_height = np.exp(-40.0 * (z - self.TARGET_HEIGHT) **2)
        r_orient = np.exp(-5.0 * (roll **2 + pitch **2))

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
        r_alive = self.W_ALIVE * r_vel_xy

        # Final Reward
        reward = (
            self.W_VEL_XY * r_vel_xy
            + self.W_YAW * r_yaw
            + self.W_VZ * r_vz
            + self.W_HEIGHT * r_height
            + self.W_ORIENT * r_orient
            + r_energy
            + r_smooth
            + r_symmetry
            + r_alive
        )

        return float(reward)