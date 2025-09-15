# train_rl_updated.py
import os
import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Union
from datetime import datetime

# 添加项目路径
sys.path.append('/root/DaXBench')

from rl_env_wrapper import SingleRLClothFoldEnv, RLClothFoldEnv
from rl_algorithms import PPOAgent, SACAgent
from daxbench.core.envs.fold_env import DefaultConf

from icecream import ic as print

# 尝试导入swanlab，如果没有安装则跳过
try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    print("警告: swanlab未安装，将跳过实验记录功能")
    HAS_SWANLAB = False


class RLTrainer:
    """强化学习训练器 - 支持批量和单环境"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 生成统一时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 初始化SwanLab (如果可用)
        if HAS_SWANLAB:
            # 格式：{task}_{rl_type}_{timestamp}
            experiment_name = f"{config.get('task', 'unknown')}_{config['algorithm']}_{self.timestamp}"
            swanlab.init(
                project=config.get('project_name', 'cloth-folding-rl'),
                experiment_name=experiment_name,
                config=config
            )

            
        print(config)
        # 创建环境
        env_conf = DefaultConf()
        env_conf.record_video = True
        env_conf.seed = config.get('seed', 42)
        env_conf.task = config.get('task', 'S_Corner_All_Middle')
        env_conf.sampling_method = config.get('sampling_method', 'farthest_point')
        env_conf.num_sampled_particles = config.get('num_sampled_particles', 15)
        print(env_conf.task)
        if env_conf.task.startswith('S'):
            env_conf.cloth_type = 'square'
        elif env_conf.task.startswith('T'):
            env_conf.cloth_type = 'tshirt'
        elif env_conf.task.startswith('P'):
            env_conf.cloth_type = 'pant'
        elif env_conf.task.startswith('R'):
            env_conf.cloth_type = 'rectangle'
        else:
            raise ValueError(f"Unsupported task: {env_conf.task}")
        
        # 根据配置选择单环境或批量环境
        use_batch_env = config.get('use_batch_env', False)
        batch_size = config.get('batch_size', 1)
        
        if use_batch_env and batch_size > 1:
            # 使用批量环境
            env_conf.batch_size = batch_size
            self.env = RLClothFoldEnv(
                conf=env_conf,
                obs_type=config['obs_type'],
                single_arm=config.get('single_arm', True),
                reward_type=config.get('reward_type', 'final_goal'),
                action_type=config.get('action_type', 'continuous'),
                num_sampled_particles=config.get('num_sampled_particles', 70),
                sampling_method=config.get('sampling_method', 'farthest_point_jax'),
                rl_type=config['algorithm'],
                mode="train",
                timestamp=self.timestamp
            )
            self.is_batch_env = True
        else:
            # 使用单环境包装器
            env_conf.batch_size = 1
            self.env = SingleRLClothFoldEnv(
                conf=env_conf,
                obs_type=config['obs_type'],
                single_arm=config.get('single_arm', True),
                reward_type=config.get('reward_type', 'final_goal'),
                action_type=config.get('action_type', 'continuous'),
                num_sampled_particles=config.get('num_sampled_particles', 70),
                sampling_method=config.get('sampling_method', 'farthest_point_jax'),
                rl_type=config['algorithm'],
                mode="train",
                timestamp=self.timestamp
            )
            self.is_batch_env = False
        
        # 创建评估环境 (使用batch_size=10的多环境)
        eval_conf = DefaultConf()
        eval_conf.task = config.get('task', 'S_Corner_All_Middle')
        eval_conf.sampling_method = config.get('sampling_method', 'farthest_point')
        eval_conf.num_sampled_particles = config.get('num_sampled_particles', 15)
        if eval_conf.task.startswith('S'):
            eval_conf.cloth_type = 'square'
        elif eval_conf.task.startswith('T'):
            eval_conf.cloth_type = 'tshirt'
        elif eval_conf.task.startswith('P'):
            eval_conf.cloth_type = 'pant'
        elif eval_conf.task.startswith('R'):
            eval_conf.cloth_type = 'rectangle'
        eval_conf.record_video = True  # 评估时记录视频
        eval_conf.seed = config.get('seed', 42)
        self.eval_batch_size = 10
        eval_conf.batch_size = self.eval_batch_size
        
        self.eval_env = RLClothFoldEnv(
            conf=eval_conf,
            obs_type=config['obs_type'],
            single_arm=config.get('single_arm', True),
            reward_type=config.get('reward_type', 'final_goal'),
            action_type=config.get('action_type', 'continuous'),
            num_sampled_particles=config.get('num_sampled_particles', 70),
            sampling_method=config.get('sampling_method', 'farthest_point_jax'),
            rl_type=config['algorithm'],
            mode="eval",
            timestamp=self.timestamp
        )
        
        # 初始化智能体
        self.agent = self._create_agent()
        
        # 创建保存目录
        self.save_dir = Path(config.get('save_dir', './rl_results'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练统计
        self.episode = 0
        self.total_steps = 0
        self.best_eval_reward = -float('inf')
    

        
    def _create_agent(self) -> Union[PPOAgent, SACAgent]:
        """创建智能体"""
        algorithm = self.config['algorithm'].lower()
        action_type = self.config.get('action_type', 'continuous')
        
        # 获取粒子数量 - 直接使用配置中的值
        num_particles = self.config.get('num_sampled_particles', 70)
        
        if action_type == "continuous":
            action_dim = 4 if self.config.get('single_arm', True) else 8
        elif action_type == "discrete":
            # 对于离散动作空间，action_dim是粒子数量（用于网络输出层）
            action_dim = num_particles
        
        if algorithm == 'ppo':
            if self.config['obs_type'] == 'particle':
                return PPOAgent(
                    num_particles=num_particles,
                    action_dim=action_dim,
                    action_type=action_type,
                    lr=self.config.get('lr', 3e-4),
                    gamma=self.config.get('gamma', 0.6),
                    eps_clip=self.config.get('eps_clip', 0.2),
                    k_epochs=self.config.get('k_epochs', 4),
                    hidden_dim=self.config.get('hidden_dim', 256)
                )
            else:  # depth
                obs_shape = (400, 600, 1)  # 固定深度图尺寸
                return PPOAgent(
                    obs_shape=obs_shape,
                    action_dim=action_dim,
                    lr=self.config.get('lr', 3e-4),
                    gamma=self.config.get('gamma', 0.99),
                    eps_clip=self.config.get('eps_clip', 0.2),
                    k_epochs=self.config.get('k_epochs', 4),
                    hidden_dim=self.config.get('hidden_dim', 256)
                )
        elif algorithm == 'sac':
            if self.config['obs_type'] == 'particle':
                return SACAgent(
                    num_particles=num_particles,
                    action_dim=action_dim,
                    action_type=action_type,
                    lr=self.config.get('lr', 3e-4),
                    gamma=self.config.get('gamma', 0.6),
                    tau=self.config.get('tau', 0.005),
                    alpha=self.config.get('alpha', 0.2),
                    buffer_size=self.config.get('buffer_size', 100000),
                    batch_size=self.config.get('rl_batch_size', 256),
                    hidden_dim=self.config.get('hidden_dim', 256),
                    random_exploration_steps=self.config.get('random_exploration_steps', 15000)
                )
            else:  # depth
                obs_shape = (400, 600, 1)  # 固定深度图尺寸
                return SACAgent(
                    obs_shape=obs_shape,
                    action_dim=action_dim,
                    lr=self.config.get('lr', 3e-4),
                    gamma=self.config.get('gamma', 0.99),
                    tau=self.config.get('tau', 0.005),
                    alpha=self.config.get('alpha', 0.2),
                    buffer_size=self.config.get('buffer_size', 100000),
                    batch_size=self.config.get('rl_batch_size', 256),
                    hidden_dim=self.config.get('hidden_dim', 256),
                    random_exploration_steps=self.config.get('random_exploration_steps', 15000)
                )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def train_episode(self) -> Dict[str, float]:
        """训练一个回合"""
        obs = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # 修复：正确初始化批量环境的统计
        if self.is_batch_env:
            batch_size = getattr(self.env, 'batch_size', obs.shape[0])
            # 为每个环境单独跟踪统计
            env_subgoals_reached = np.zeros(batch_size, dtype=int)
            env_max_subgoal_idx = np.zeros(batch_size, dtype=int)
            env_success = np.zeros(batch_size, dtype=bool)
            env_active = np.ones(batch_size, dtype=bool)  # 跟踪哪些环境还在运行
            env_episode_rewards = np.zeros(batch_size, dtype=float)  # 跟踪每个环境的累积奖励
        else:
            episode_info = {
                'subgoals_reached': 0,
                'max_subgoal_idx': 0,
                'success': False
            }
        
        while True:
            # 选择动作
            if self.is_batch_env:
                # 批量环境：为每个环境生成动作
                batch_actions = []
                for i in range(batch_size):
                    action = self.agent.select_action(obs[i], deterministic=False)
                    batch_actions.append(action)
                action = np.array(batch_actions)
            else:
                # 单环境
                action = self.agent.select_action(obs, deterministic=False)
            
            # 执行动作
            next_obs, reward, done, info = self.env.step(action)
            
            # 存储经验
            if self.is_batch_env:
                # 批量环境：为每个环境存储经验
                reward_array = np.array(reward) if not isinstance(reward, np.ndarray) else reward
                done_array = np.array(done) if not isinstance(done, np.ndarray) else done
                
                for i in range(batch_size):
                    if hasattr(self.agent, 'store_reward_done'):  # PPO
                        self.agent.store_reward_done(float(reward_array[i]), bool(done_array[i]))
                    elif hasattr(self.agent, 'store_transition'):  # SAC
                        self.agent.store_transition(obs[i], action[i], float(reward_array[i]), next_obs[i], bool(done_array[i]))
                
                # 为每个环境累积奖励
                env_episode_rewards += reward_array
                
                # 更新统计（使用平均值）
                episode_reward += float(np.mean(reward_array))
                episode_steps += 1
                self.total_steps += batch_size
                
                # 修复：正确处理批量环境的子目标和成功统计
                for i in range(batch_size):
                    if env_active[i]:  # 只为仍在运行的环境更新统计
                        # 子目标统计 - 只在当前步骤达成时计数，避免重复累加
                        if 'subgoal_reached' in info and len(info['subgoal_reached']) > i and info['subgoal_reached'][i]:
                            env_subgoals_reached[i] += 1
                        
                        # 更新最大子目标索引
                        if 'subgoal_idx' in info and len(info['subgoal_idx']) > i:
                            env_max_subgoal_idx[i] = max(env_max_subgoal_idx[i], int(info['subgoal_idx'][i]))
                        
                        # 成功标记 - 一旦成功就保持True
                        if 'episode_success' in info and len(info['episode_success']) > i and info['episode_success'][i]:
                            env_success[i] = True
                        
                        # 如果环境完成，标记为不活跃
                        if len(done_array) > i and done_array[i]:
                            env_active[i] = False
                
                # 检查是否所有环境都完成
                if not np.any(env_active):
                    break
                    
            else:
                # 单环境
                if hasattr(self.agent, 'store_reward_done'):  # PPO
                    self.agent.store_reward_done(float(reward), bool(done))
                elif hasattr(self.agent, 'store_transition'):  # SAC
                    self.agent.store_transition(obs, action, float(reward), next_obs, bool(done))
                
                # 更新统计
                episode_reward += float(reward)
                episode_steps += 1
                self.total_steps += 1
                
                # 更新episode信息
                if info.get('subgoal_reached', False):
                    episode_info['subgoals_reached'] += 1
                episode_info['max_subgoal_idx'] = max(episode_info['max_subgoal_idx'], 
                                                      info.get('subgoal_idx', 0))
                if info.get('episode_success', False):
                    episode_info['success'] = True
                
                if done:
                    break
            
            obs = next_obs
        
        # 返回统计结果
        if self.is_batch_env:
            return {
                'episode_reward': episode_reward,
                'episode_reward_mean': float(np.mean(env_episode_rewards)),  # 各环境奖励的平均值
                'episode_reward_std': float(np.std(env_episode_rewards)),    # 各环境奖励的标准差
                'episode_reward_min': float(np.min(env_episode_rewards)),    # 最小奖励
                'episode_reward_max': float(np.max(env_episode_rewards)),    # 最大奖励
                'episode_steps': episode_steps,
                'subgoals_reached': int(np.sum(env_subgoals_reached)),
                'average_subgoals_reached': float(np.mean(env_subgoals_reached)),
                'max_subgoal_idx': int(np.max(env_max_subgoal_idx)),
                'success': bool(np.any(env_success)),
                'success_rate': float(np.mean(env_success)),  # 修复：正确的成功率计算
                'individual_successes': int(np.sum(env_success))  # 成功的环境数量
            }
        else:
            return {
                'episode_reward': episode_reward,
                'episode_reward_mean': episode_reward,  # 单环境情况下，平均值就是奖励本身
                'episode_reward_std': 0.0,              # 单环境情况下，标准差为0
                'episode_reward_min': episode_reward,   # 单环境情况下，最小值就是奖励本身
                'episode_reward_max': episode_reward,   # 单环境情况下，最大值就是奖励本身
                'episode_steps': episode_steps,
                'subgoals_reached': episode_info['subgoals_reached'],
                'average_subgoals_reached': episode_info['subgoals_reached'],
                'max_subgoal_idx': episode_info['max_subgoal_idx'],
                'success': episode_info['success'],
                'success_rate': float(episode_info['success']),
                'individual_successes': int(episode_info['success'])
            }
    
    def evaluate(self, num_episodes: int = 10, save_success_videos: bool = True) -> Dict[str, float]:
        """评估智能体 - 使用批量环境"""
        import random
        
        eval_rewards = []
        eval_successes = []
        eval_subgoals = []
        
        # 用于收集成功的episode数据
        success_episodes = []  # 存储成功episode的数据
        
        # 计算需要多少轮来完成num_episodes个评估
        episodes_per_batch = self.eval_batch_size
        num_batches = (num_episodes + episodes_per_batch - 1) // episodes_per_batch
        
        total_evaluated = 0
        
        for batch_idx in range(num_batches):
            # 重置评估环境并启用视频记录
            if save_success_videos:
                self.eval_env.env.record_video = True
                self.eval_env.env.episode_data = {'states': []}
            
            obs = self.eval_env.reset()
            batch_episode_rewards = np.zeros(episodes_per_batch)
            batch_subgoals_reached = np.zeros(episodes_per_batch)
            batch_success = np.zeros(episodes_per_batch, dtype=bool)
            
            while True:
                # 为每个环境生成动作
                batch_actions = []
                for i in range(episodes_per_batch):
                    if total_evaluated + i < num_episodes:
                        action = self.agent.select_action(obs[i], deterministic=True)
                        batch_actions.append(action)
                    else:
                        # 超出需要评估的数量，使用零动作
                        # 默认动作维度
                        action_dim = 4 if self.config.get('single_arm', True) else 8
                        action = np.zeros(action_dim)
                        batch_actions.append(action)
                
                action = np.array(batch_actions)
                obs, rewards, done, info = self.eval_env.step(action)
                
                # 累积奖励和统计
                batch_episode_rewards += rewards
                
                if 'subgoal_reached' in info:
                    batch_subgoals_reached += info['subgoal_reached'].astype(int)
                if 'episode_success' in info:
                    batch_success |= info['episode_success']
                
                # 检查是否有环境完成
                if np.any(done):
                    break
            
            # 收集成功的episode数据
            if save_success_videos:
                episode_data = self.eval_env.env.get_episode_data()
                for i in range(episodes_per_batch):
                    if total_evaluated + i < num_episodes and batch_success[i]:
                        # 记录成功的episode信息
                        success_info = {
                            'batch_idx': i,
                            'chosen_id': self.eval_env.env.chosen_ids[i] if hasattr(self.eval_env.env, 'chosen_ids') else i,
                            'episode_data': episode_data,
                            'reward': float(batch_episode_rewards[i]),
                            'subgoals_reached': int(batch_subgoals_reached[i])
                        }
                        success_episodes.append(success_info)
            
            # 收集这一批的结果
            for i in range(episodes_per_batch):
                if total_evaluated < num_episodes:
                    eval_rewards.append(float(batch_episode_rewards[i]))
                    eval_successes.append(bool(batch_success[i]))
                    eval_subgoals.append(int(batch_subgoals_reached[i]))
                    total_evaluated += 1
        
        # 保存成功episode的视频
        if save_success_videos and success_episodes:
            self._save_success_videos(success_episodes)
        
        return {
            'eval_reward_mean': float(np.mean(eval_rewards)),
            'eval_reward_std': float(np.std(eval_rewards)),
            'eval_success_rate': float(np.mean(eval_successes)),
            'eval_subgoals_mean': float(np.mean(eval_subgoals)),
            'eval_subgoals_std': float(np.std(eval_subgoals)),
            'num_success_episodes': len(success_episodes)
        }
    
    def _save_success_videos(self, success_episodes):
        """保存成功episode的视频，每个ID最多保存2个"""
        import random
        from collections import defaultdict
        import os
        
        if not success_episodes:
            return
        
        # 按chosen_id分组成功的episodes
        episodes_by_id = defaultdict(list)
        for episode in success_episodes:
            episodes_by_id[episode['chosen_id']].append(episode)
        
        print(f"找到 {len(success_episodes)} 个成功的episodes，涉及 {len(episodes_by_id)} 个不同的ID")
        
        # 为每个ID随机选择最多2个episode保存视频
        for chosen_id, episodes in episodes_by_id.items():
            # 如果成功的episode数量大于2个，随机选择2个
            if len(episodes) > 2:
                selected_episodes = random.sample(episodes, 2)
                print(f"ID {chosen_id}: 从 {len(episodes)} 个成功episodes中随机选择 2 个保存视频")
            else:
                selected_episodes = episodes
                print(f"ID {chosen_id}: 保存所有 {len(episodes)} 个成功episodes的视频")
            
            # 为选中的episodes保存视频
            for idx, episode in enumerate(selected_episodes):
                try:
                    # 创建保存目录：eval_videos/{chosen_id}/
                    video_dir = os.path.join(self.save_dir, 'eval_videos', str(chosen_id))
                    os.makedirs(video_dir, exist_ok=True)
                    
                    # 保存视频，使用FoldEnv的render_video方法
                    video_files = self.eval_env.env.render_video(
                        episode['episode_data'], 
                        output_dir=video_dir,
                        batch_idx=episode['batch_idx'],
                        format="gif",
                        fps=20,
                        filename_prefix=f"success_episode_{idx+1}"
                    )
                    
                    if video_files:
                        print(f"成功保存视频: {video_files[0]}")
                        
                        # 保存episode信息到文本文件
                        info_file = os.path.join(video_dir, f"success_episode_{idx+1}_info.txt")
                        with open(info_file, 'w') as f:
                            f.write(f"Episode Info:\n")
                            f.write(f"Chosen ID: {chosen_id}\n")
                            f.write(f"Batch Index: {episode['batch_idx']}\n")
                            f.write(f"Reward: {episode['reward']:.4f}\n")
                            f.write(f"Subgoals Reached: {episode['subgoals_reached']}\n")
                            f.write(f"Timestamp: {self.timestamp}\n")
                    
                except Exception as e:
                    print(f"保存视频时出错 (ID {chosen_id}): {e}")
        
        print(f"视频保存完成，保存路径: {os.path.join(self.save_dir, 'eval_videos')}")
    
    def train(self):
        """主训练循环"""
        print(f"开始训练 - 算法: {self.config['algorithm']}, 观察类型: {self.config['obs_type']}")
        print(f"环境: 动作维度={self.env.action_space.shape}, 观察维度={self.env.observation_space.shape}")
        print(f"批量环境: {self.is_batch_env}")
        
        # 调整更新频率以适应批量环境
        if self.is_batch_env:
            # 对于批量环境，基于回合数更新而不是步数
            update_frequency_episodes = self.config.get('update_frequency_episodes', 10)  # 每10个回合更新
            update_frequency = None  # 不使用步数更新
        else:
            # 对于单环境，使用传统的步数更新
            update_frequency = self.config.get('update_frequency', 2048 if self.config['algorithm'].lower() == 'ppo' else 1)
            update_frequency_episodes = None
            
        eval_frequency = self.config.get('eval_frequency', 100)
        save_frequency = self.config.get('save_frequency', 1000)
        max_episodes = self.config.get('max_episodes', 10000)
        
        episode_rewards = []
        recent_rewards = []
        
        for episode in range(max_episodes):
            self.episode = episode
            
            # 训练一个回合
            episode_stats = self.train_episode()
            episode_rewards.append(episode_stats['episode_reward'])
            recent_rewards.append(episode_stats['episode_reward'])
            
            # 保持最近100个回合的奖励
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            
            # 记录训练数据
            train_log = {
                'train/episode_reward': episode_stats['episode_reward'],
                'train/episode_steps': episode_stats['episode_steps'],
                'train/average_subgoals_reached': episode_stats['average_subgoals_reached'],
                'train/subgoals_reached': episode_stats['subgoals_reached'],
                'train/max_subgoal_idx': episode_stats['max_subgoal_idx'],
                'train/success': int(episode_stats['success']),
                'train/success_rate': episode_stats['success_rate'],
                'train/individual_successes': episode_stats['individual_successes'],
                'train/recent_reward_mean': np.mean(recent_rewards),
                'train/recent_reward_std': np.std(recent_rewards),
                'train/total_steps': self.total_steps,
                'train/episode': episode
            }
            
            # 如果是批量环境，添加批量环境特有的统计
            if self.is_batch_env:
                train_log.update({
                    'train/batch_episode_reward_mean': episode_stats['episode_reward_mean'],
                    'train/batch_episode_reward_std': episode_stats['episode_reward_std'],
                    'train/batch_episode_reward_min': episode_stats['episode_reward_min'],
                    'train/batch_episode_reward_max': episode_stats['episode_reward_max'],
                })
            
            # 为SAC算法添加随机探索阶段的信息
            if (self.config['algorithm'].lower() == 'sac' and 
                isinstance(self.agent, SACAgent)):
                is_random_exploration = self.total_steps < self.agent.random_exploration_steps
                train_log['train/random_exploration'] = int(is_random_exploration)
                train_log['train/exploration_progress'] = min(1.0, self.total_steps / self.agent.random_exploration_steps)
            
            print('Total steps: ', self.total_steps)
            # 更新智能体
            should_update = False
            
            if self.config['algorithm'].lower() == 'ppo':
                if self.is_batch_env and update_frequency_episodes is not None:
                    # 批量环境：每N个回合更新一次
                    should_update = (episode % update_frequency_episodes == 0 and episode > 0)
                elif not self.is_batch_env and update_frequency is not None:
                    # 单环境：按步数更新
                    should_update = (self.total_steps % update_frequency == 0)
                    
                if should_update:
                    update_stats = self.agent.update()
                    train_log.update({f'train/{k}': v for k, v in update_stats.items()})
                    print(f"PPO更新完成 (Episode {episode}, Total steps {self.total_steps})")
                    
            elif (self.config['algorithm'].lower() == 'sac' and 
                  isinstance(self.agent, SACAgent) and
                  hasattr(self.agent, 'replay_buffer') and
                  len(self.agent.replay_buffer) > self.config.get('rl_batch_size', 256)):
                update_stats = self.agent.update()
                train_log.update({f'train/{k}': v for k, v in update_stats.items()})
            
            # 评估
            if episode % eval_frequency == 0:
                eval_stats = self.evaluate()
                train_log.update({f'eval/{k}': v for k, v in eval_stats.items()})
                
                # 保存最佳模型
                if eval_stats['eval_reward_mean'] > self.best_eval_reward:
                    self.best_eval_reward = eval_stats['eval_reward_mean']
                    self.agent.save(str(self.save_dir / 'best_model.pth'))
                    print(f"新的最佳模型保存! 平均奖励: {self.best_eval_reward:.2f}")
            
            # 定期保存
            if episode % save_frequency == 0:
                self.agent.save(str(self.save_dir / f'model_episode_{episode}.pth'))
            
            # 记录到SwanLab
            if HAS_SWANLAB:
                swanlab.log(train_log)
            
            # 打印进度
            if episode % 10 == 0:
                if self.is_batch_env:
                    progress_msg = (f"Episode {episode}: "
                                   f"AvgReward={episode_stats['episode_reward_mean']:.2f}±{episode_stats['episode_reward_std']:.2f}, "
                                   f"Range=[{episode_stats['episode_reward_min']:.2f}, {episode_stats['episode_reward_max']:.2f}], "
                                   f"Steps={episode_stats['episode_steps']}, "
                                   f"Subgoals={episode_stats['subgoals_reached']}, "
                                   f"Success={episode_stats['success']}, "
                                   f"SuccessRate={episode_stats['success_rate']:.2%}")
                else:
                    progress_msg = (f"Episode {episode}: "
                                   f"Reward={episode_stats['episode_reward']:.2f}, "
                                   f"Steps={episode_stats['episode_steps']}, "
                                   f"Subgoals={episode_stats['subgoals_reached']}, "
                                   f"Success={episode_stats['success']}, "
                                   f"SuccessRate={episode_stats['success_rate']:.2%}")
                
                # 为SAC算法添加随机探索信息
                if (self.config['algorithm'].lower() == 'sac' and 
                    isinstance(self.agent, SACAgent)):
                    is_random_exploration = self.total_steps < self.agent.random_exploration_steps
                    if is_random_exploration:
                        exploration_progress = self.total_steps / self.agent.random_exploration_steps
                        progress_msg += f", 随机探索: {exploration_progress:.1%}"
                    else:
                        progress_msg += ", 策略探索"
                
                print(progress_msg)
        
        # 训练完成
        print("训练完成!")
        self.agent.save(str(self.save_dir / 'final_model.pth'))
        if HAS_SWANLAB:
            swanlab.finish()


def main():
    parser = argparse.ArgumentParser(description='布料折叠强化学习训练 - 支持批量环境')
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'sac'], 
                       default='ppo', help='强化学习算法')
    parser.add_argument('--obs_type', type=str, choices=['particle', 'depth'],
                       default='particle', help='观察类型')
    parser.add_argument('--single_arm', action='store_true', default=True,
                       help='使用单臂（否则双臂）')
    parser.add_argument('--action_type', type=str, choices=['continuous', 'discrete'],
                       default='continuous', help='动作空间类型：连续或离散')
    parser.add_argument('--use_batch_env', action='store_true', default=False,
                       help='使用批量环境进行训练')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='批量环境大小')
    parser.add_argument('--cloth_type', type=str, default='square',
                       help='布料类型')
    parser.add_argument('--reward_type', type=str, default='final_goal',
                       help='奖励类型')
    parser.add_argument('--max_episodes', type=int, default=5000,
                       help='最大训练回合数')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./rl_results',
                       help='结果保存目录')
    parser.add_argument('--project_name', type=str, default='cloth-folding-rl',
                       help='SwanLab项目名称')
    parser.add_argument('--experiment_name', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                       help='实验名称')
    parser.add_argument('--update_frequency_episodes', type=int, default=10,
                       help='批量环境的更新频率（回合数）')
    parser.add_argument('--num_sampled_particles', type=int, default=30,
                       help='采样后的粒子数量')
    parser.add_argument('--sampling_method', type=str, default='farthest_point_jax',
                       help='采样方法')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='隐藏层维度')
    parser.add_argument('--task', type=str, default='S_Corner_All_Middle',
                       help='任务名称')
    parser.add_argument('--random_exploration_steps', type=int, default=10000,
                       help='SAC算法随机探索步数')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 配置
    config = {
        'algorithm': args.algorithm,
        'obs_type': args.obs_type,
        'reward_type': args.reward_type,
        'single_arm': args.single_arm,
        'action_type': args.action_type,
        'use_batch_env': args.use_batch_env,
        'batch_size': args.batch_size,
        'cloth_type': args.cloth_type,
        'max_episodes': args.max_episodes,
        'lr': args.lr,
        'seed': args.seed,
        'save_dir': args.save_dir,
        'project_name': args.project_name,
        'experiment_name': args.experiment_name or f"{args.algorithm}_{args.obs_type}_{int(time.time())}",
        'task': args.task,
        # 训练参数
        'gamma': 0.6,
        'hidden_dim': args.hidden_dim,
        'max_subgoal_steps': 2,
        'eval_frequency': 20,
        'save_frequency': 1000,
        
        # PPO特定参数
        'eps_clip': 0.2,
        'k_epochs': 4,
        'update_frequency': 2048,  # 单环境的步数更新频率
        'update_frequency_episodes': args.update_frequency_episodes,  # 批量环境的回合更新频率
        
        # SAC特定参数
        'tau': 0.005,
        'alpha': 0.1,
        'buffer_size': 100000,
        'rl_batch_size': 512,
        'random_exploration_steps': args.random_exploration_steps,
        
        # 环境参数
        'num_sampled_particles': args.num_sampled_particles,
        'sampling_method': args.sampling_method,
    }
    
    # 创建训练器并开始训练
    trainer = RLTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main() 