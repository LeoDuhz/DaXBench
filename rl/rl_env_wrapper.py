


# rl_env_wrapper.py
import os
import pickle
import numpy as np
import jax.numpy as jnp
from typing import List, Dict, Any, Tuple
import gym
from gym import spaces
import time
from datetime import datetime
from daxbench.core.envs.fold_env import FoldEnv, DefaultConf, spatial_sampling
from daxbench.core.envs.basic.cloth_env import ClothEnv


from icecream import ic as print


class RLClothFoldEnv(gym.Env):
    """
    强化学习环境包装器，支持分阶段子目标奖励和批量环境
    """
    
    def __init__(self, 
                 conf=None,
                 obs_type="particle",  # "particle" or "depth"
                 single_arm=True,
                 reward_type="final_goal",  # "final_goal", "subgoal", or "combined"
                 action_type="continuous",  # "continuous" or "discrete"
                 num_sampled_particles=64,  # 采样后的粒子数量
                 sampling_method="kmeans",  # 空间采样方法 ("kmeans", "grid", "farthest_point")
                 rl_type="ppo",  # RL算法类型，用于路径命名
                 mode="train",  # 模式：train或eval
                 timestamp=None):  # 统一时间戳
        """
        初始化RL环境
        
        Args:
            conf: 环境配置
            obs_type: 观察类型 ("particle" 或 "depth")
            single_arm: 是否使用单臂 (True) 或双臂 (False)
            reward_type: 奖励类型 ("final_goal", "subgoal", "combined")
            action_type: 动作类型 ("continuous" 或 "discrete")
            num_sampled_particles: 采样后的目标粒子数量
            sampling_method: 空间采样方法 ("kmeans", "grid", "farthest_point")
            rl_type: RL算法类型，用于路径命名
            mode: 模式：train或eval
            timestamp: 统一时间戳，用于路径和实验命名的一致性
        """
        
        if conf is None:
            conf = DefaultConf()
            conf.batch_size = 1  # 默认单个环境，可以通过conf覆盖
            conf.cloth_type = 'square'
            conf.record_video = False

        
        self.conf = conf
        self.task = conf.task
        self.batch_size = conf.batch_size
        self.obs_type = obs_type
        self.single_arm = single_arm
        self.reward_type = reward_type
        self.action_type = action_type
        self.num_sampled_particles = num_sampled_particles  # 目标采样粒子数量
        self.sampling_method = sampling_method  # 空间采样方法
        self.rl_type = rl_type
        self.mode = mode
        self.episode_count = 0
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        self.rollout_path = f'rollout/{self.task}/{self.rl_type}_{self.timestamp}/{self.mode}'

        # Cache for sampled particles to avoid redundant sampling
        self.cached_sampled_particles = None
        
        # 初始化基础环境，子目标路径由FoldEnv内部根据conf.task自动确定
        self.env = FoldEnv(
            conf=conf, 
            seed=conf.seed,
            reward_type=reward_type
        )
        
        # 设置动作空间 - 批量动作
        if action_type == "continuous":
            if single_arm:
                # 批量4维连续动作: (batch_size, 4) - pick_x, pick_y, place_x, place_y
                self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.batch_size, 4), dtype=np.float32)
            else:
                # 批量8维连续动作: (batch_size, 8) 
                self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.batch_size, 8), dtype=np.float32)
        elif action_type == "discrete":
            # 离散动作空间：每个动作是[pick_idx, place_idx]，范围是采样后的粒子数量
            if single_arm:
                # 批量2维离散动作: (batch_size, 2) - pick_particle_idx, place_particle_idx
                self.action_space = spaces.MultiDiscrete([self.num_sampled_particles, self.num_sampled_particles] * self.batch_size)
            else:
                # 批量4维离散动作: (batch_size, 4) - pick1_idx, place1_idx, pick2_idx, place2_idx
                self.action_space = spaces.MultiDiscrete([self.num_sampled_particles, self.num_sampled_particles, 
                                                        self.num_sampled_particles, self.num_sampled_particles] * self.batch_size)
        
        # 设置观察空间 - 批量观察
        if obs_type == "particle":
            # 批量粒子位置观察: (batch_size, num_particles, 3)
            self.observation_space = spaces.Box(
                low=0, high=1, 
                shape=(self.batch_size, self.num_sampled_particles, 3), 
                dtype=np.float32
            )
        elif obs_type == "depth":
            # 批量深度图观察: (batch_size, H, W, C)
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(self.batch_size, 400, 600, 1),
                dtype=np.uint8
            )
        else:
            raise ValueError(f"Unsupported obs_type: {obs_type}")
            
        # 初始化状态
        self.reset()
    
    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        """将RL动作转换为环境动作格式 - 支持批量处理"""
        # action shape: (batch_size, action_dim)
        batch_size = action.shape[0]
        
        if self.action_type == "continuous":
            return self._convert_continuous_action(action, batch_size)
        elif self.action_type == "discrete":
            return self._convert_discrete_action(action, batch_size)
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")
    
    def _convert_continuous_action(self, action: np.ndarray, batch_size: int) -> np.ndarray:
        """转换连续动作"""
        if self.single_arm:
            # 单臂：(batch_size, 4) -> (batch_size, 12)
            env_action = np.zeros((batch_size, 12))
            env_action[:, 0] = action[:, 0]  # pick_x
            env_action[:, 1] = 0
            env_action[:, 2] = action[:, 1]
            env_action[:, 3] = action[:, 2]
            env_action[:, 4] = 0
            env_action[:, 5] = action[:, 3]
            env_action[:, 6] = action[:, 0]
            env_action[:, 7] = 0
            env_action[:, 8] = action[:, 1]
            env_action[:, 9] = action[:, 2]
            env_action[:, 10] = 0
            env_action[:, 11] = action[:, 3]
        else:
            # 双臂：(batch_size, 8) -> (batch_size, 12)
            env_action = np.zeros((batch_size, 12))
            env_action[:, 0] = action[:, 0]  # pick_x
            env_action[:, 1] = 0
            env_action[:, 2] = action[:, 1]
            env_action[:, 3] = action[:, 2]
            env_action[:, 4] = 0
            env_action[:, 5] = action[:, 3]
            env_action[:, 6] = action[:, 0]
            env_action[:, 7] = 0
            env_action[:, 8] = action[:, 1]
            env_action[:, 9] = action[:, 2]
            env_action[:, 10] = 0
            env_action[:, 11] = action[:, 3]
        return env_action
    
    def _convert_discrete_action(self, action: np.ndarray, batch_size: int) -> np.ndarray:
        """转换离散动作 - 粒子索引转换为坐标"""
        # Cache sampled particles for all batches
        if not hasattr(self, 'cached_sampled_particles') or self.cached_sampled_particles is None:
            state = self.env.get_state()
            sampled_particles_list = []
            for i in range(batch_size):
                valid_mask = state.x[i][:, 2] != 0
                valid_particles = state.x[i][valid_mask]
                num_valid = valid_particles.shape[0]
                if num_valid <= self.num_sampled_particles:
                    sampled_state = valid_particles
                else:
                    sampled_state = spatial_sampling(valid_particles, self.num_sampled_particles, method=self.sampling_method)

                sampled_particles_list.append(sampled_state)
            self.cached_sampled_particles = sampled_particles_list
        else:
            sampled_particles_list = self.cached_sampled_particles
        
        if self.single_arm:
            # action shape: (batch_size, 2) - [pick_idx, place_idx]
            env_action = np.zeros((batch_size, 12))
            
            for i in range(batch_size):
                pick_idx = int(action[i, 0])
                place_idx = int(action[i, 1])
                
                # 检查是否为无效动作（pick和place索引相同）
                if pick_idx == place_idx and pick_idx == 0:
                    # 生成无害的动作：将pick和place都设置在远离布料的位置
                    # 使用 (-1, 0, -1) 坐标，这个位置远离布料范围 [0, 1]
                    print(f'invalid action: {i}', 'pick_idx', pick_idx, 'position', sampled_particles_list[i][pick_idx, :])
                    env_action[i, 0] = -1.0   # pick_x
                    env_action[i, 1] = 0.0    # pick_y (固定为0)
                    env_action[i, 2] = -1.0   # pick_z
                    env_action[i, 3] = -1.0   # place_x (与pick相同，无移动)
                    env_action[i, 4] = 0.0    # place_y (固定为0)
                    env_action[i, 5] = -1.0   # place_z (与pick相同，无移动)
                    # 复制到第二个臂
                    env_action[i, 6:12] = env_action[i, 0:6]
                    continue
                
                # Use cached sampled particles
                sampled_state = sampled_particles_list[i]
                # print(sampled_state[:, 0].min(), sampled_state[:, 0].max())
                # print(sampled_state[:, 1].min(), sampled_state[:, 1].max())
                # print(sampled_state[:, 2].min(), sampled_state[:, 2].max())
                # 确保索引不越界
                max_idx = sampled_state.shape[0] - 1
                pick_idx = min(pick_idx, max_idx)
                place_idx = min(place_idx, max_idx)
                
                # 获取粒子坐标
                pick_pos = sampled_state[pick_idx, :]  # (3,) - [x, y, z]
                place_pos = sampled_state[place_idx, :]  # (3,) - [x, y, z]
                # print(pick_idx, place_idx)
                # print(pick_pos, place_pos)
                # 转换为12维动作格式 [pick_x, pick_y, pick_z, place_x, place_y, place_z, ...]
                env_action[i, 0] = pick_pos[0]   # pick_x
                env_action[i, 1] = 0             # pick_y (固定为0)
                env_action[i, 2] = pick_pos[2]   # pick_z
                env_action[i, 3] = place_pos[0]  # place_x
                env_action[i, 4] = 0             # place_y (固定为0)
                env_action[i, 5] = place_pos[2]  # place_z
                # 复制到第二个臂
                env_action[i, 6:12] = env_action[i, 0:6]
        else:
            # 双臂：action shape: (batch_size, 4) - [pick1_idx, place1_idx, pick2_idx, place2_idx]
            env_action = np.zeros((batch_size, 12))
            
            for i in range(batch_size):
                # Use cached sampled particles
                sampled_state = sampled_particles_list[i]
                
                max_idx = sampled_state.shape[0] - 1
                pick1_idx = min(int(action[i, 0]), max_idx)
                place1_idx = min(int(action[i, 1]), max_idx)
                pick2_idx = min(int(action[i, 2]), max_idx)
                place2_idx = min(int(action[i, 3]), max_idx)
                
                # 检查第一个臂是否为无效动作
                if pick1_idx == place1_idx:
                    # 第一个臂无效动作：设置在远离布料的位置
                    env_action[i, 0] = -1.0   # pick1_x
                    env_action[i, 1] = 0.0    # pick1_y
                    env_action[i, 2] = -1.0   # pick1_z
                    env_action[i, 3] = -1.0   # place1_x
                    env_action[i, 4] = 0.0    # place1_y
                    env_action[i, 5] = -1.0   # place1_z
                else:
                    # 第一个臂正常动作
                    pick1_pos = sampled_state[pick1_idx, :]
                    place1_pos = sampled_state[place1_idx, :]
                    env_action[i, 0] = pick1_pos[0]   # pick1_x
                    env_action[i, 1] = 0              # pick1_y
                    env_action[i, 2] = pick1_pos[2]   # pick1_z
                    env_action[i, 3] = place1_pos[0]  # place1_x
                    env_action[i, 4] = 0              # place1_y
                    env_action[i, 5] = place1_pos[2]  # place1_z
                
                # 检查第二个臂是否为无效动作
                if pick2_idx == place2_idx:
                    # 第二个臂无效动作：设置在远离布料的位置
                    env_action[i, 6] = -1.0   # pick2_x
                    env_action[i, 7] = 0.0    # pick2_y
                    env_action[i, 8] = -1.0   # pick2_z
                    env_action[i, 9] = -1.0   # place2_x
                    env_action[i, 10] = 0.0   # place2_y
                    env_action[i, 11] = -1.0  # place2_z
                else:
                    # 第二个臂正常动作
                    pick2_pos = sampled_state[pick2_idx, :]
                    place2_pos = sampled_state[place2_idx, :]
                    env_action[i, 6] = pick2_pos[0]   # pick2_x
                    env_action[i, 7] = 0              # pick2_y
                    env_action[i, 8] = pick2_pos[2]   # pick2_z
                    env_action[i, 9] = place2_pos[0]  # place2_x
                    env_action[i, 10] = 0             # place2_y
                    env_action[i, 11] = place2_pos[2] # place2_z
                
        return env_action
    
    def _get_observation(self) -> np.ndarray:
        """获取批量观察 - 自动过滤填充粒子并智能采样，保持粒子维度结构"""
        state = self.env.get_state()
        
        if self.obs_type == "particle":
            observations = []
            sampled_particles_cache = []  # Cache sampled particles for action conversion
            
            for i in range(self.batch_size):
                # 过滤掉无效粒子
                valid_mask = (
                    (state.x[i][:, 2] != 0) &      # z坐标不为0
                    (state.x[i][:, 0] != 0) &      # x坐标不为0  
                    (state.x[i][:, 0] > 0.1) &     # x坐标大于0.1
                    (state.x[i][:, 0] < 0.9) &      # x坐标小于0.9
                    (state.x[i][:, 2] > 0.1) &      # y坐标大于0.1
                    (state.x[i][:, 2] < 0.9)       # y坐标小于0.9
                )
                valid_particles = state.x[i][valid_mask]
                
                # 使用空间采样算法
                num_valid = valid_particles.shape[0]
                if num_valid <= self.num_sampled_particles:
                    # 如果有效粒子数量不足，使用所有有效粒子并填充
                    x_sampled = valid_particles
                    # 如果粒子数不足，用最后一个有效粒子填充到目标数量
                    if num_valid < self.num_sampled_particles:
                        padding_needed = self.num_sampled_particles - num_valid
                        last_particle = valid_particles[-1:, :]  # 最后一个有效粒子
                        padding = jnp.repeat(last_particle, padding_needed, axis=0)
                        x_sampled = jnp.concatenate([x_sampled, padding], axis=0)
                else:
                    # 使用空间采样算法进行智能采样
                    x_sampled = spatial_sampling(np.array(valid_particles), self.num_sampled_particles, method=self.sampling_method)
                    x_sampled = jnp.array(x_sampled)
                
                # 直接使用粒子坐标，保持(num_particles, 3)的形状
                observations.append(x_sampled)
                sampled_particles_cache.append(x_sampled)
                
                # 调试信息（可选）
                # if i == 0:  # 只打印第一个环境的信息
                #     print(f"Batch {i}: 总粒子数={state.x[i].shape[0]}, 有效粒子数={num_valid}")
                #     print(f"Batch {i}: 采样后粒子形状={x_sampled.shape}")
            
            # Cache the sampled particles for action conversion
            self.cached_sampled_particles = sampled_particles_cache
            
            result = np.array(observations)  # shape: (batch_size, num_particles, 3)
            return result
            
        elif self.obs_type == "depth":
            raise ValueError(f"Depth observation is not supported")
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """执行批量动作"""
        # 转换动作格式
        start_time = time.time()
        env_action = self._convert_action(action)
        
        # 执行动作 - FoldEnv现在内部处理所有奖励计算
        obs, rewards, done, info = self.env.step_fold(env_action, visualize=True, visualize_path=f'{self.rollout_path}/{self.episode_count}')
        
        # 获取新的观察
        observation = self._get_observation()
        
        # FoldEnv已经处理了所有奖励和子目标逻辑
        # 我们只需要确保返回正确的格式
        if isinstance(rewards, (int, float)):
            # 如果是标量，转换为数组
            rewards = np.full(self.batch_size, rewards)
        
        if isinstance(done, bool):
            # 如果是标量，转换为数组
            done = np.full(self.batch_size, done)
        time_cost = time.time() - start_time
        print(f"Time cost: {time_cost} seconds")
        return observation, np.array(rewards), np.array(done), info
    
    def reset(self) -> np.ndarray:
        """重置批量环境"""
        self.episode_count += 1
        # 重置基础环境 - FoldEnv内部会处理子目标重置
        self.env.reset_env()
        
        # Clear cached particles on reset
        self.cached_sampled_particles = None
        
        # 获取初始观察
        observation = self._get_observation()
        
        return observation
    
    def render(self, mode='rgb_array', idx=0):
        """渲染指定batch的环境"""
        state = self.env.get_state()
        rgb, _ = self.env.render(state, visualize=(mode=='human'), idx=idx)
        return rgb
    
    def close(self):
        """关闭环境"""
        pass
    
    def get_current_subgoal_info(self) -> Dict:
        """获取批量子目标信息"""
        # 从FoldEnv获取子目标信息
        if hasattr(self.env, '_get_subgoal_info'):
            subgoal_info = self.env._get_subgoal_info()
            subgoal_info['batch_size'] = self.batch_size
            return subgoal_info
        else:
            return {'batch_size': self.batch_size}

class SingleRLClothFoldEnv(gym.Env):
    """
    单环境包装器，用于强化学习训练
    内部使用批量环境但只暴露单个环境接口
    """
    
    def __init__(self, 
                 conf=None,
                 obs_type="particle",
                 single_arm=True,
                 reward_type="final_goal",
                 action_type="continuous",
                 num_sampled_particles=64,
                 sampling_method="kmeans",
                 rl_type="ppo",
                 mode="train",
                 timestamp=None):
        """
        初始化单环境包装器
        """
        if conf is None:
            conf = DefaultConf()
            conf.batch_size = 1  # 强制使用单个环境
            conf.cloth_type = 'square'
            conf.record_video = False
        else:
            # 确保batch_size为1
            conf.batch_size = 1
            
        # 使用批量环境
        self.batch_env = RLClothFoldEnv(
            conf=conf,
            obs_type=obs_type,
            single_arm=single_arm,
            reward_type=reward_type,
            action_type=action_type,
            num_sampled_particles=num_sampled_particles,
            sampling_method=sampling_method,
            rl_type=rl_type,
            mode=mode,
            timestamp=timestamp
        )
        
        # 设置单环境的动作和观察空间
        if action_type == "continuous":
            if single_arm:
                self.action_space = spaces.Box(low=0.3, high=0.7, shape=(4,), dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=0.3, high=0.7, shape=(8,), dtype=np.float32)
        elif action_type == "discrete":
            # 离散动作空间：粒子索引
            if single_arm:
                # 2维离散动作: [pick_particle_idx, place_particle_idx]
                self.action_space = spaces.MultiDiscrete([num_sampled_particles, num_sampled_particles])
            else:
                # 4维离散动作: [pick1_idx, place1_idx, pick2_idx, place2_idx]
                self.action_space = spaces.MultiDiscrete([num_sampled_particles, num_sampled_particles, 
                                                        num_sampled_particles, num_sampled_particles])
        
        if obs_type == "particle":
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(num_sampled_particles, 3), 
                dtype=np.float32
            )
        elif obs_type == "depth":
            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(320, 640, 1),
                dtype=np.float32
            )
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行单步动作"""
        # 将单个动作扩展为批量
        batch_action = action.reshape(1, -1)
        
        # 执行批量步骤
        batch_obs, batch_rewards, batch_done, batch_info = self.batch_env.step(batch_action)
        
        # 提取第一个环境的结果
        obs = batch_obs[0]
        reward = batch_rewards[0]
        done = batch_done[0]
        
        # 转换info字典
        info = {}
        for key, value in batch_info.items():
            if isinstance(value, np.ndarray) and len(value) > 0:
                info[key] = value[0]
            else:
                info[key] = value
        
        return obs, float(reward), bool(done), info
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        batch_obs = self.batch_env.reset()
        return batch_obs[0]  # 返回第一个环境的观察
    
    def render(self, mode='rgb_array'):
        """渲染环境"""
        return self.batch_env.render(mode=mode, idx=0)
    
    def close(self):
        """关闭环境"""
        self.batch_env.close()
    
    def get_current_subgoal_info(self) -> Dict:
        """获取当前子目标信息"""
        batch_info = self.batch_env.get_current_subgoal_info()
        return {
            'current_subgoal_idx': batch_info['current_subgoal_idx'][0],
            'current_subgoal_step': batch_info['current_subgoal_step'][0],
            'total_subgoals': batch_info['total_subgoals'],
            'max_subgoal_steps': batch_info['max_subgoal_steps']
        }