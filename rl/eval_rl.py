# evaluate_rl.py
import argparse
import numpy as np
import torch
from pathlib import Path

from rl_env_wrapper import RLClothFoldEnv
from rl_algorithms import PPOAgent, SACAgent
from daxbench.core.envs.fold_env import DefaultConf


def evaluate_model(model_path: str, config: dict, num_episodes: int = 10):
    """评估训练好的模型"""
    
    # 创建环境
    env_conf = DefaultConf()
    env_conf.batch_size = 1
    env_conf.cloth_type = config.get('cloth_type', 'square')
    env_conf.record_video = True  # 评估时记录视频
    env_conf.seed = config.get('seed', 42)
    
    env = RLClothFoldEnv(
        conf=env_conf,
        subgoals_dir=config['subgoals_dir'],
        obs_type=config['obs_type'],
        single_arm=config.get('single_arm', True),
        max_subgoal_steps=config.get('max_subgoal_steps', 50)
    )
    
    # 创建智能体
    algorithm = config['algorithm'].lower()
    action_dim = 4 if config.get('single_arm', True) else 8
    
    if algorithm == 'ppo':
        if config['obs_type'] == 'particle':
            obs_dim = env.observation_space.shape[0]
            agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim)
        else:
            obs_shape = env.observation_space.shape
            agent = PPOAgent(obs_shape=obs_shape, action_dim=action_dim)
    elif algorithm == 'sac':
        if config['obs_type'] == 'particle':
            obs_dim = env.observation_space.shape[0]
            agent = SACAgent(obs_dim=obs_dim, action_dim=action_dim)
        else:
            obs_shape = env.observation_space.shape
            agent = SACAgent(obs_shape=obs_shape, action_dim=action_dim)
    
    # 加载模型
    agent.load(model_path)
    print(f"模型已从 {model_path} 加载")
    
    # 评估
    results = []
    success_count = 0
    
    for episode in range(num_episodes):
        print(f"\n评估回合 {episode + 1}/{num_episodes}")
        
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        subgoals_reached = 0
        max_subgoal_idx = 0
        success = False
        
        episode_states = []
        
        while True:
            action = agent.select_action(obs, deterministic=True)
            next_obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            if info.get('subgoal_reached', False):
                subgoals_reached += 1
                print(f"  达到子目标 {info['subgoal_idx']}")
            
            max_subgoal_idx = max(max_subgoal_idx, info.get('subgoal_idx', 0))
            
            if info.get('episode_success', False):
                success = True
                success_count += 1
                print(f"  回合成功!")
            
            obs = next_obs
            
            if done:
                break
        
        result = {
            'episode': episode + 1,
            'reward': episode_reward,
            'steps': episode_steps,
            'subgoals_reached': subgoals_reached,
            'max_subgoal_idx': max_subgoal_idx,
            'success': success
        }
        
        results.append(result)
        
        print(f"  奖励: {episode_reward:.2f}")
        print(f"  步数: {episode_steps}")
        print(f"  达成子目标数: {subgoals_reached}")
        print(f"  最大子目标索引: {max_subgoal_idx}")
        
        # 保存视频
        if hasattr(env, 'env') and hasattr(env.env, 'get_episode_data'):
            episode_data = env.env.get_episode_data()
            if episode_data['states']:
                video_dir = Path(config.get('save_dir', './eval_results')) / 'videos'
                video_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    env.env.render_video(
                        episode_data, 
                        output_dir=str(video_dir),
                        batch_idx=0,
                        format="gif",
                        fps=10,
                        filename_prefix=f"eval_episode_{episode}"
                    )
                    print(f"  视频已保存到 {video_dir}")
                except Exception as e:
                    print(f"  视频保存失败: {e}")
    
    # 计算统计结果
    rewards = [r['reward'] for r in results]
    steps = [r['steps'] for r in results]
    subgoals = [r['subgoals_reached'] for r in results]
    
    print(f"\n=== 评估结果 ({num_episodes} 回合) ===")
    print(f"平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"平均步数: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print(f"平均子目标达成数: {np.mean(subgoals):.1f} ± {np.std(subgoals):.1f}")
    print(f"成功率: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='评估训练好的强化学习模型')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'sac'], 
                       default='ppo', help='强化学习算法')
    parser.add_argument('--obs_type', type=str, choices=['particle', 'depth'],
                       default='particle', help='观察类型')
    parser.add_argument('--single_arm', action='store_true', default=True,
                       help='使用单臂')
    parser.add_argument('--subgoals_dir', type=str,
                       default='/root/DaXBench/oracle/S_Corner_All_Middle/0/subgoals',
                       help='子目标文件夹路径')
    parser.add_argument('--cloth_type', type=str, default='square',
                       help='布料类型')
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='评估回合数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./eval_results',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    config = {
        'algorithm': args.algorithm,
        'obs_type': args.obs_type,
        'single_arm': args.single_arm,
        'subgoals_dir': args.subgoals_dir,
        'cloth_type': args.cloth_type,
        'seed': args.seed,
        'save_dir': args.save_dir,
        'max_subgoal_steps': 50,
    }
    
    results = evaluate_model(args.model_path, config, args.num_episodes)


if __name__ == '__main__':
    main()