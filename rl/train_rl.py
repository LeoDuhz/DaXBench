# train_rl_updated.py
import os
import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any
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
        
        # 初始化SwanLab (如果可用)
        if HAS_SWANLAB:
            swanlab.init(
                project=config.get('project_name', 'cloth-folding-rl'),
                experiment_name=config.get('experiment_name', f"{config['algorithm']}_{config['obs_type']}"),
                config=config
            )
        print(config)
        # 创建环境
        env_conf = DefaultConf()
        env_conf.record_video = True
        env_conf.seed = config.get('seed', 42)
        env_conf.task = config.get('task', 'S_Corner_All_Middle')
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
                num_sampled_particles=config.get('num_sampled_particles', 70)
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
                num_sampled_particles=config.get('num_sampled_particles', 70)
            )
            self.is_batch_env = False
        
        # 创建评估环境 (使用batch_size=10的多环境)
        eval_conf = DefaultConf()
        eval_conf.task = config.get('task', 'S_Corner_All_Middle')
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
            num_sampled_particles=config.get('num_sampled_particles', 70)
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
        
    def _create_agent(self):
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
                    hidden_dim=self.config.get('hidden_dim', 256)
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
                    hidden_dim=self.config.get('hidden_dim', 256)
                )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def train_episode(self) -> Dict[str, float]:
        """训练一个回合"""
        obs = self.env.reset()
        episode_reward = 0
        episode_steps = 0
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
                batch_size = getattr(self.env, 'batch_size', obs.shape[0])
                for i in range(batch_size):
                    action = self.agent.select_action(obs[i], deterministic=False)
                    batch_actions.append(action)
                action = np.array(batch_actions)
            else:
                # 单环境
                action = self.agent.select_action(obs, deterministic=False)
            
            # 执行动作
            next_obs, reward, done, info = self.env.step(action)
            print(reward)
            # 存储经验
            if self.is_batch_env:
                # 批量环境：为每个环境存储经验
                batch_size = getattr(self.env, 'batch_size', obs.shape[0])
                for i in range(batch_size):
                    if hasattr(self.agent, 'store_reward_done'):  # PPO
                        self.agent.store_reward_done(float(reward[i]), bool(done[i]))
                    elif hasattr(self.agent, 'store_transition'):  # SAC
                        self.agent.store_transition(obs[i], action[i], float(reward[i]), next_obs[i], bool(done[i]))
                
                # 更新统计（使用平均值）
                print('Before add reward: ', episode_reward, 'after add reward: ', episode_reward + float(np.mean(reward)))
                episode_reward += float(np.mean(reward))
                episode_steps += 1
                self.total_steps += batch_size
                
                # 更新episode信息
                if 'subgoal_reached' in info:
                    episode_info['subgoals_reached'] += int(np.sum(info['subgoal_reached']))
                if 'subgoal_idx' in info:
                    episode_info['max_subgoal_idx'] = max(episode_info['max_subgoal_idx'], 
                                                          int(np.max(info['subgoal_idx'])))
                if 'episode_success' in info:
                    episode_info['success'] = bool(np.any(info['episode_success']))
                
                # 检查是否有环境完成
                if np.any(done):
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
        # if self.episode % self.config.get('update_frequency_episodes', 10) == 0:
        #     episode_data = self.env.env.get_episode_data()
        #     self.env.env.render_video(episode_data, output_dir=f'rl_vis/{self.episode}', batch_idx=0, format="gif", fps=20, filename_prefix="episode")

        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'subgoals_reached': episode_info['subgoals_reached'],
            'max_subgoal_idx': episode_info['max_subgoal_idx'],
            'success': episode_info['success']
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """评估智能体 - 使用批量环境"""
        eval_rewards = []
        eval_successes = []
        eval_subgoals = []
        
        # 计算需要多少轮来完成num_episodes个评估
        episodes_per_batch = self.eval_batch_size
        num_batches = (num_episodes + episodes_per_batch - 1) // episodes_per_batch
        
        total_evaluated = 0
        
        for batch_idx in range(num_batches):
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
                        action = np.zeros(self.eval_env.action_space.shape[1])
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
            
            # 收集这一批的结果
            for i in range(episodes_per_batch):
                if total_evaluated < num_episodes:
                    eval_rewards.append(float(batch_episode_rewards[i]))
                    eval_successes.append(bool(batch_success[i]))
                    eval_subgoals.append(int(batch_subgoals_reached[i]))
                    total_evaluated += 1
        
        return {
            'eval_reward_mean': float(np.mean(eval_rewards)),
            'eval_reward_std': float(np.std(eval_rewards)),
            'eval_success_rate': float(np.mean(eval_successes)),
            'eval_subgoals_mean': float(np.mean(eval_subgoals)),
            'eval_subgoals_std': float(np.std(eval_subgoals))
        }
    
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
                'train/subgoals_reached': episode_stats['subgoals_reached'],
                'train/max_subgoal_idx': episode_stats['max_subgoal_idx'],
                'train/success': int(episode_stats['success']),
                'train/recent_reward_mean': np.mean(recent_rewards),
                'train/recent_reward_std': np.std(recent_rewards),
                'train/total_steps': self.total_steps,
                'train/episode': episode
            }
            print('Total steps: ', self.total_steps)
            # 更新智能体
            should_update = False
            
            if self.config['algorithm'].lower() == 'ppo':
                if self.is_batch_env:
                    # 批量环境：每N个回合更新一次
                    should_update = (episode % update_frequency_episodes == 0 and episode > 0)
                else:
                    # 单环境：按步数更新
                    should_update = (self.total_steps % update_frequency == 0)
                    
                if should_update:
                    update_stats = self.agent.update()
                    train_log.update({f'train/{k}': v for k, v in update_stats.items()})
                    print(f"PPO更新完成 (Episode {episode}, Total steps {self.total_steps})")
                    
            elif (self.config['algorithm'].lower() == 'sac' and 
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
                print(f"Episode {episode}: "
                      f"Reward={episode_stats['episode_reward']:.2f}, "
                      f"Steps={episode_stats['episode_steps']}, "
                      f"Subgoals={episode_stats['subgoals_reached']}, "
                      f"Success={episode_stats['success']}")
        
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
    parser.add_argument('--num_sampled_particles', type=int, default=70,
                       help='采样后的粒子数量')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='隐藏层维度')
    parser.add_argument('--task', type=str, default='S_Corner_All_Middle',
                       help='任务名称')
    
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
        
        # 环境参数
        'num_sampled_particles': args.num_sampled_particles,
    }
    
    # 创建训练器并开始训练
    trainer = RLTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main() 