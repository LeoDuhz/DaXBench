# rl_algorithms.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
from typing import Tuple, List, Dict, Any
from icecream import ic as print


class ActorCritic(nn.Module):
    """PPO算法的Actor-Critic网络 - 支持粒子结构输入"""
    
    def __init__(self, num_particles: int, action_dim: int, hidden_dim: int = 512, action_type: str = "continuous"):
        super(ActorCritic, self).__init__()
        self.action_type = action_type
        self.action_dim = action_dim
        self.num_particles = num_particles
        
        # 粒子特征提取层 - 处理每个粒子的3D坐标
        self.particle_encoder = nn.Sequential(
            nn.Linear(3, 64),  # 每个粒子的3D坐标 -> 64维特征
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 全局特征聚合 - 将所有粒子特征聚合
        particle_feature_dim = num_particles * 64
        self.global_encoder = nn.Sequential(
            nn.Linear(particle_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        if action_type == "continuous":
            # 连续动作空间：Actor网络 (策略网络)
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        elif action_type == "discrete":
            # 离散动作空间：为pick和place分别输出概率分布
            self.pick_logits = nn.Linear(hidden_dim, num_particles)  # pick点选择
            self.place_logits = nn.Linear(hidden_dim, num_particles)  # place点选择
        else:
            raise ValueError(f"Unsupported action_type: {action_type}")
        
        # Critic网络 (价值网络)
        self.critic = nn.Linear(hidden_dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if 'logits' in str(m):  # 对输出层使用较小的初始化
                torch.nn.init.orthogonal_(m.weight, gain=0.01)
            else:
                torch.nn.init.orthogonal_(m.weight, gain=1.0)
            torch.nn.init.constant_(m.bias, 0)

    
        
    def forward(self, obs: torch.Tensor):
        # obs shape: (batch_size, num_particles, 3)
        batch_size = obs.shape[0]
        
        # 处理每个粒子的特征
        # 重塑为 (batch_size * num_particles, 3)
        particles_flat = obs.reshape(-1, 3)
        # 通过粒子编码器 -> (batch_size * num_particles, 64)
        particle_features = self.particle_encoder(particles_flat)
        # 重塑回 (batch_size, num_particles, 64)
        particle_features = particle_features.reshape(batch_size, self.num_particles, 64)
        # 展平为全局特征 (batch_size, num_particles * 64)
        global_features = particle_features.reshape(batch_size, -1)
        
        # 全局特征编码
        features = self.global_encoder(global_features)
        
        if self.action_type == "continuous":
            # 连续动作空间输出
            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_log_std.clamp(-20, 2))
            value = self.critic(features)
            return action_mean, action_std, value
        elif self.action_type == "discrete":
            # 离散动作空间输出
            pick_logits = self.pick_logits(features)
            place_logits = self.place_logits(features)
            value = self.critic(features)
            return pick_logits, place_logits, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        forward_output = self.forward(obs)
        
        if self.action_type == "continuous":
            action_mean, action_std, value = forward_output
            
            if deterministic:
                a = torch.tanh(action_mean)
                action = 0.5 * (a + 1.0)
                log_prob = torch.zeros(obs.shape[0])  # 返回零概率而不是None
            else:
                dist = Normal(action_mean, action_std)
                u = dist.rsample()
                a = torch.tanh(u)
                action = 0.5 * (a + 1.0)
                eps = 1e-6
                log_prob = dist.log_prob(u) - torch.log(1 - a.pow(2) + eps)
                log_prob = log_prob.sum(dim=-1)
                log_prob = log_prob - action.shape[-1] * np.log(2.0)
            
            return action, log_prob, value
            
        elif self.action_type == "discrete":
            pick_logits, place_logits, value = forward_output
            
            # 使用Categorical分布进行采样
            pick_dist = torch.distributions.Categorical(logits=pick_logits)
            place_dist = torch.distributions.Categorical(logits=place_logits)
            
            if deterministic:
                pick_action = torch.argmax(pick_logits, dim=-1)
                place_action = torch.argmax(place_logits, dim=-1)
                log_prob = torch.zeros(obs.shape[0])  # 返回零概率而不是None
            else:
                pick_action = pick_dist.sample()
                place_action = place_dist.sample()
                pick_log_prob = pick_dist.log_prob(pick_action)
                place_log_prob = place_dist.log_prob(place_action)
                log_prob = pick_log_prob + place_log_prob
            
            # 将离散动作打包为[pick_idx, place_idx]格式
            action = torch.stack([pick_action, place_action], dim=-1).float()
            
            return action, log_prob, value


class DepthActorCritic(nn.Module):
    """处理深度图输入的Actor-Critic网络"""
    
    def __init__(self, input_shape: Tuple[int, int, int], action_dim: int, hidden_dim: int = 256):
        super(DepthActorCritic, self).__init__()
        
        # CNN特征提取
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 计算卷积输出尺寸
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor和Critic头
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)
        
    def _get_conv_output_size(self, input_shape: Tuple[int, int, int]) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
            output = self.conv_net(dummy_input)
            return output.shape[1]
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 调整维度：(batch, H, W, C) -> (batch, C, H, W)
        if obs.dim() == 4 and obs.shape[-1] in [1, 3]:
            obs = obs.permute(0, 3, 1, 2)
        
        conv_features = self.conv_net(obs)
        features = self.fc(conv_features)
        
        # Actor输出
        action_mean = torch.sigmoid(self.actor_mean(features))
        action_std = torch.exp(self.actor_log_std.clamp(-20, 2))
        
        # Critic输出
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, action_std, value = self.forward(obs)
        
        if deterministic:
            action = action_mean
            log_prob = None
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        action = torch.clamp(action, 0.0, 1.0)
        
        return action, log_prob, value


class PPOAgent:
    """PPO算法实现"""
    
    def __init__(self, 
                 num_particles: int = None,
                 obs_shape: Tuple[int, int, int] = None,
                 action_dim: int = 4,
                 action_type: str = "continuous",  # "continuous" or "discrete"
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 hidden_dim: int = 256,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.action_type = action_type
        
        # 根据输入类型选择网络
        if num_particles is not None:
            # 粒子位置输入
            self.policy = ActorCritic(num_particles, action_dim, hidden_dim, action_type).to(device)
        elif obs_shape is not None:
            # 深度图输入 - 暂不支持离散动作空间
            if action_type == "discrete":
                raise ValueError("Discrete action space not supported for depth observation yet")
            self.policy = DepthActorCritic(obs_shape, action_dim, hidden_dim).to(device)
        else:
            raise ValueError("Must specify either num_particles or obs_shape")
            
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 打印网络参数量
        self.print_network_parameters()
        
        # 经验缓冲区
        self.buffer = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.get_action(state_tensor, deterministic)
            
            if not deterministic:
                # 存储到缓冲区
                self.buffer['states'].append(state)
                self.buffer['actions'].append(action.cpu().numpy()[0])
                self.buffer['log_probs'].append(log_prob.cpu().numpy()[0])
                self.buffer['values'].append(value.cpu().numpy()[0])
            
            return action.cpu().numpy()[0]
    
    def store_reward_done(self, reward: float, done: bool):
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
    
    def update(self):
        # 计算折扣奖励（不标准化）
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.buffer['rewards']), reversed(self.buffer['dones'])):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        # 转换为张量
        old_states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        old_actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        old_values = torch.FloatTensor(self.buffer['values']).to(self.device)

        # 计算优势（只标准化一次）
        advantages = rewards - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 添加调试信息
        print(f"=== PPO Update Debug ===")
        print(f"Action type: {self.action_type}")
        print(f"Buffer size: states={len(self.buffer['states'])}, rewards={len(self.buffer['rewards'])}")
        print(f"Rewards: mean={rewards.mean():.6f}, std={rewards.std():.6f}, range=[{rewards.min():.6f}, {rewards.max():.6f}]")
        print(f"Values: mean={old_values.mean():.6f}, std={old_values.std():.6f}")
        print(f"Advantages: mean={advantages.mean():.6f}, std={advantages.std():.6f}")
        print(f"Old log_probs: mean={old_log_probs.mean():.6f}, std={old_log_probs.std():.6f}")
        
        # PPO更新
        for epoch in range(self.k_epochs):
            if self.action_type == "continuous":
                # 连续动作空间的PPO更新
                if hasattr(self.policy, 'conv_net'):
                    action_mean, action_std, values = self.policy(old_states)
                else:
                    action_mean, action_std, values = self.policy(old_states)
                
                dist = Normal(action_mean, action_std)
                
                # 修正：使用与get_action一致的计算方式
                a = torch.clamp(old_actions, 0.0, 1.0)
                # 逆变换：[0,1] -> [-1,1] -> pre_tanh
                a_scaled = 2.0 * a - 1.0  # [0,1] -> [-1,1]
                pre_tanh = torch.atanh(torch.clamp(a_scaled, -1 + 1e-6, 1 - 1e-6))
                
                # 重新计算log_prob（与get_action保持一致）
                eps = 1e-6
                new_log_probs = dist.log_prob(pre_tanh).sum(dim=-1)
                new_log_probs = new_log_probs - torch.log(1 - a_scaled.pow(2) + eps).sum(dim=-1)
                new_log_probs = new_log_probs - a.shape[-1] * np.log(2.0)
                
                entropy = dist.entropy().sum(dim=-1)
                
            elif self.action_type == "discrete":
                # 离散动作空间的PPO更新
                pick_logits, place_logits, values = self.policy(old_states)
                
                # 创建分布
                pick_dist = torch.distributions.Categorical(logits=pick_logits)
                place_dist = torch.distributions.Categorical(logits=place_logits)
                
                # 提取旧动作的pick和place索引
                old_pick_actions = old_actions[:, 0].long()
                old_place_actions = old_actions[:, 1].long()
                
                # 计算新的log概率（与get_action保持一致）
                pick_log_probs = pick_dist.log_prob(old_pick_actions)
                place_log_probs = place_dist.log_prob(old_place_actions)
                new_log_probs = pick_log_probs + place_log_probs
                
                # 计算熵
                entropy = pick_dist.entropy() + place_dist.entropy()
                
                # 调试信息（离散动作空间特有）
                if epoch == 0:
                    print(f"Pick actions range: [{old_pick_actions.min():.0f}, {old_pick_actions.max():.0f}]")
                    print(f"Place actions range: [{old_place_actions.min():.0f}, {old_place_actions.max():.0f}]")
                    print(f"Pick logits: mean={pick_logits.mean():.6f}, std={pick_logits.std():.6f}")
                    print(f"Place logits: mean={place_logits.mean():.6f}, std={place_logits.std():.6f}")
                    print(f"Pick log_probs: mean={pick_log_probs.mean():.6f}, std={pick_log_probs.std():.6f}")
                    print(f"Place log_probs: mean={place_log_probs.mean():.6f}, std={place_log_probs.std():.6f}")
        
            # 调试信息（通用）
            if epoch == 0:
                print(f"New log_probs: mean={new_log_probs.mean():.6f}, std={new_log_probs.std():.6f}")
                ratio = torch.exp(new_log_probs - old_log_probs)
                print(f"Log prob diff: mean={(new_log_probs - old_log_probs).mean():.6f}, std={(new_log_probs - old_log_probs).std():.6f}")
                print(f"Ratio: mean={ratio.mean():.6f}, std={ratio.std():.6f}, range=[{ratio.min():.6f}, {ratio.max():.6f}]")
                
                # 检查是否有异常值
                if torch.any(torch.isnan(ratio)) or torch.any(torch.isinf(ratio)):
                    print("WARNING: Ratio contains NaN or Inf values!")
                if torch.any(ratio < 0):
                    print("WARNING: Ratio contains negative values!")
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), rewards)
            entropy_loss = -entropy.mean()
            
            # 修正权重
            total_loss = 100*actor_loss + 0.1 * critic_loss + 0.01 * entropy_loss
            
            if epoch == 0:
                print(f"Raw losses - Actor: {actor_loss.item():.6f}, Critic: {critic_loss.item():.6f}, Entropy: {entropy_loss.item():.6f}")
                print(f"Surr1: mean={surr1.mean():.6f}, std={surr1.std():.6f}")
                print(f"Surr2: mean={surr2.mean():.6f}, std={surr2.std():.6f}")
                print(f"Min(surr1, surr2): mean={torch.min(surr1, surr2).mean():.6f}")
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # 清空缓冲区
        self.clear_buffer()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            # 添加调试信息
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
            'ratio_mean': ratio.mean().item(),
            'log_prob_diff_mean': (new_log_probs - old_log_probs).mean().item()
        }
    
    def print_network_parameters(self):
        """打印网络参数量"""
        total_params = sum(p.numel() for p in self.policy.parameters())
        trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        
        print(f"=== PPO网络参数统计 ===")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print(f"参数大小: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")
        
        # 分层统计
        print(f"\n各层参数详情:")
        for name, module in self.policy.named_modules():
            if len(list(module.parameters())) > 0:
                module_params = sum(p.numel() for p in module.parameters())
                print(f"  {name}: {module_params:,} 参数")
        print("=" * 30)
    
    def clear_buffer(self):
        for key in self.buffer:
            self.buffer[key].clear()
    
    def save(self, filepath: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class SACAgent:
    """SAC算法实现"""
    
    def __init__(self,
                 num_particles: int = None,
                 obs_shape: Tuple[int, int, int] = None,
                 action_dim: int = 4,
                 action_type: str = "continuous",  # "continuous" or "discrete"
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 hidden_dim: int = 256,
                 # 新增参数：随机采样阶段
                 random_exploration_steps: int = 10000,  # 随机探索步数
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.action_type = action_type
        self.num_particles = num_particles  # 存储粒子数量
        
        # 随机探索参数
        self.random_exploration_steps = random_exploration_steps
        self.total_steps = 0  # 总步数计数器
        
        # 网络初始化
        if num_particles is not None:
            self.actor = SACActorParticle(num_particles, action_dim, hidden_dim, action_type).to(device)
            # 对于离散动作空间，critic的action_dim需要调整
            critic_action_dim = 2 if action_type == "discrete" else action_dim  # 离散时是[pick_idx, place_idx]
            self.critic1 = SACCriticParticle(num_particles, critic_action_dim, hidden_dim).to(device)
            self.critic2 = SACCriticParticle(num_particles, critic_action_dim, hidden_dim).to(device)
            self.target_critic1 = SACCriticParticle(num_particles, critic_action_dim, hidden_dim).to(device)
            self.target_critic2 = SACCriticParticle(num_particles, critic_action_dim, hidden_dim).to(device)
        elif obs_shape is not None:
            if action_type == "discrete":
                raise ValueError("Discrete action space not supported for depth observation yet")
            self.actor = SACActorDepth(obs_shape, action_dim, hidden_dim).to(device)
            self.critic1 = SACCriticDepth(obs_shape, action_dim, hidden_dim).to(device)
            self.critic2 = SACCriticDepth(obs_shape, action_dim, hidden_dim).to(device)
            self.target_critic1 = SACCriticDepth(obs_shape, action_dim, hidden_dim).to(device)
            self.target_critic2 = SACCriticDepth(obs_shape, action_dim, hidden_dim).to(device)
        else:
            raise ValueError("Must specify either num_particles or obs_shape")
        
        # 复制参数到目标网络
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size, num_particles or obs_shape, action_dim)
        
        # 自动调节温度参数
        if action_type == "discrete":
            # 离散动作空间的目标熵：-log(1/N) = log(N)，其中N是动作数量
            # 对于单臂是2个动作(pick, place)，双臂是4个动作
            num_discrete_actions = 2  # 单臂：pick + place
            self.target_entropy = -np.log(1.0 / action_dim) * 2 
        else:
            self.target_entropy = -action_dim
        self.log_alpha = torch.log(torch.tensor([alpha], device=device))  # 用初始alpha值初始化log_alpha
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # 打印网络参数量
        self.print_network_parameters()
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作 - 支持初始随机探索阶段"""
        # 在随机探索阶段使用随机动作
        if self.total_steps < self.random_exploration_steps and not deterministic:
            return self._select_random_action()
        
        # 正常的SAC动作选择
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(state_tensor, deterministic)
            return action.cpu().numpy()[0]
    
    def _select_random_action(self) -> np.ndarray:
        """随机选择动作 - 根据动作类型进行不同处理"""
        if self.action_type == "continuous":
            # 连续动作空间：在[0,1]范围内随机采样
            return np.random.uniform(0, 1, self.action_dim)
        elif self.action_type == "discrete":
            # 离散动作空间：随机选择pick和place的粒子索引
            if self.num_particles is None:
                raise ValueError("num_particles must be specified for discrete action space")
            pick_idx = np.random.randint(0, self.num_particles)
            place_idx = np.random.randint(0, self.num_particles)
            # print(pick_idx, place_idx)
            return np.array([pick_idx, place_idx], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验并更新步数计数器"""
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1
    
    def update(self):
        print('Replay buffer size: ', len(self.replay_buffer))
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # 采样经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 更新Critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones.float()) * target_q
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # 更新Actor
        new_actions, log_probs = self.actor.sample(states)
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新温度参数
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # 软更新目标网络
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item()
        }
    
    def print_network_parameters(self):
        """打印SAC网络参数量"""
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic1_params = sum(p.numel() for p in self.critic1.parameters())
        critic2_params = sum(p.numel() for p in self.critic2.parameters())
        target_critic1_params = sum(p.numel() for p in self.target_critic1.parameters())
        target_critic2_params = sum(p.numel() for p in self.target_critic2.parameters())
        
        total_params = actor_params + critic1_params + critic2_params + target_critic1_params + target_critic2_params
        
        print(f"=== SAC网络参数统计 ===")
        print(f"Actor网络: {actor_params:,} 参数")
        print(f"Critic1网络: {critic1_params:,} 参数")
        print(f"Critic2网络: {critic2_params:,} 参数")
        print(f"Target Critic1网络: {target_critic1_params:,} 参数")
        print(f"Target Critic2网络: {target_critic2_params:,} 参数")
        print(f"总参数量: {total_params:,}")
        print(f"参数大小: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")
        print("=" * 30)
    
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, filepath: str):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])


# SAC网络组件
class SACActorParticle(nn.Module):
    def __init__(self, num_particles, action_dim, hidden_dim, action_type="continuous"):
        super().__init__()
        self.action_type = action_type
        self.action_dim = action_dim
        self.num_particles = num_particles
        
        # 粒子特征提取层
        self.particle_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 全局特征聚合
        particle_feature_dim = num_particles * 64
        self.net = nn.Sequential(
            nn.Linear(particle_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        if action_type == "continuous":
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std_head = nn.Linear(hidden_dim, action_dim)
        elif action_type == "discrete":
            # 离散动作空间：为pick和place分别输出概率分布
            self.pick_logits = nn.Linear(hidden_dim, num_particles)
            self.place_logits = nn.Linear(hidden_dim, num_particles)
        else:
            raise ValueError(f"Unsupported action_type: {action_type}")
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if 'logits' in str(m):  # 对输出层使用较小的初始化
                torch.nn.init.orthogonal_(m.weight, gain=0.01)
            else:
                torch.nn.init.orthogonal_(m.weight, gain=1.0)
            torch.nn.init.constant_(m.bias, 0)
        
        # 特别处理输出层
        if self.action_type == "continuous":
            torch.nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
            torch.nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        elif self.action_type == "discrete":
            torch.nn.init.uniform_(self.pick_logits.weight, -3e-3, 3e-3)
            torch.nn.init.uniform_(self.place_logits.weight, -3e-3, 3e-3)
        
    def forward(self, obs):
        # obs shape: (batch_size, num_particles, 3)
        batch_size = obs.shape[0]
        
        # 处理每个粒子的特征
        particles_flat = obs.reshape(-1, 3)
        particle_features = self.particle_encoder(particles_flat)
        particle_features = particle_features.reshape(batch_size, self.num_particles, 64)
        global_features = particle_features.reshape(batch_size, -1)
        
        # 全局特征编码
        features = self.net(global_features)
        
        if self.action_type == "continuous":
            mean = self.mean_head(features)
            log_std = self.log_std_head(features).clamp(-20, 2)
            return mean, log_std
        elif self.action_type == "discrete":
            pick_logits = self.pick_logits(features)
            place_logits = self.place_logits(features)
            return pick_logits, place_logits
    
    def sample(self, obs, deterministic=False):
        forward_output = self.forward(obs)
        
        if self.action_type == "continuous":
            mean, log_std = forward_output
            std = torch.exp(log_std)
            if deterministic:
                # tanh-squash 再映射到 [0,1]
                action = torch.tanh(mean)
                action = 0.5 * (action + 1.0)
                return action, torch.zeros(obs.shape[0], 1)

            # 重新参数化采样
            dist = Normal(mean, std)
            u = dist.rsample()                  # 未约束空间样本
            a = torch.tanh(u)                   # squash 到 [-1,1]
            action = 0.5 * (a + 1.0)            # 映射到 [0,1]

            # log_prob 修正：tanh 的雅可比项 + 缩放常数 (0.5) 的 logJacobian
            # 原始：sum(log_prob(u)) - sum(log(1 - tanh(u)^2 + eps))
            eps = 1e-6
            log_prob = dist.log_prob(u) - torch.log(1 - a.pow(2) + eps)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            log_prob = log_prob - action.shape[-1] * np.log(2.0)  # 缩放到 [0,1] 的常数项
            return action, log_prob
            
        elif self.action_type == "discrete":
            pick_logits, place_logits = forward_output
            
            # 使用Categorical分布进行采样
            pick_dist = torch.distributions.Categorical(logits=pick_logits)
            place_dist = torch.distributions.Categorical(logits=place_logits)
            
            if deterministic:
                pick_action = torch.argmax(pick_logits, dim=-1)
                place_action = torch.argmax(place_logits, dim=-1)
                log_prob = torch.zeros(obs.shape[0], 1)
            else:
                pick_action = pick_dist.sample()
                place_action = place_dist.sample()
                pick_log_prob = pick_dist.log_prob(pick_action)
                place_log_prob = place_dist.log_prob(place_action)
                log_prob = (pick_log_prob + place_log_prob).unsqueeze(-1)
            
            # 将离散动作打包为[pick_idx, place_idx]格式
            action = torch.stack([pick_action, place_action], dim=-1).float()
            
            return action, log_prob

class SACCriticParticle(nn.Module):
    def __init__(self, num_particles, action_dim, hidden_dim):
        super().__init__()
        self.num_particles = num_particles
        
        # 粒子特征提取层
        self.particle_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 全局特征聚合 + 动作
        particle_feature_dim = num_particles * 64
        self.net = nn.Sequential(
            nn.Linear(particle_feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs, action):
        # obs shape: (batch_size, num_particles, 3)
        batch_size = obs.shape[0]
        
        # 处理每个粒子的特征
        particles_flat = obs.reshape(-1, 3)
        particle_features = self.particle_encoder(particles_flat)
        particle_features = particle_features.reshape(batch_size, self.num_particles, 64)
        global_features = particle_features.reshape(batch_size, -1)
        
        return self.net(torch.cat([global_features, action], dim=-1))


class SACActorDepth(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_dim):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(obs_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        conv_output_size = self._get_conv_output_size(obs_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
    
    def _get_conv_output_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
            output = self.conv_net(dummy_input)
            return output.shape[1]
    
    def forward(self, obs):
        if obs.dim() == 4 and obs.shape[-1] in [1, 3]:
            obs = obs.permute(0, 3, 1, 2)
        
        conv_features = self.conv_net(obs)
        features = self.fc(conv_features)
        mean = self.mean_head(features)  # 不再 sigmoid
        log_std = self.log_std_head(features).clamp(-20, 2)
        return mean, log_std
    
    def sample(self, obs, deterministic=False):
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        if deterministic:
            a = torch.tanh(mean)
            action = 0.5 * (a + 1.0)
            return action, None

        dist = Normal(mean, std)
        u = dist.rsample()
        a = torch.tanh(u)
        action = 0.5 * (a + 1.0)

        eps = 1e-6
        log_prob = dist.log_prob(u) - torch.log(1 - a.pow(2) + eps)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob = log_prob - action.shape[-1] * np.log(2.0)
        return action, log_prob


class SACCriticDepth(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_dim):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(obs_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        conv_output_size = self._get_conv_output_size(obs_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def _get_conv_output_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
            output = self.conv_net(dummy_input)
            return output.shape[1]
    
    def forward(self, obs, action):
        if obs.dim() == 4 and obs.shape[-1] in [1, 3]:
            obs = obs.permute(0, 3, 1, 2)
        
        conv_features = self.conv_net(obs)
        state_action = torch.cat([conv_features, action], dim=-1)
        q_value = self.fc(state_action)
        
        return q_value


class ReplayBuffer:
    def __init__(self, capacity, obs_dim_or_shape, action_dim):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

# DDPG算法实现
class DDPGActorParticle(nn.Module):
    """DDPG算法的Actor网络 - 粒子输入版本"""
    
    def __init__(self, num_particles: int, action_dim: int, hidden_dim: int = 512):
        super(DDPGActorParticle, self).__init__()
        self.action_dim = action_dim
        self.num_particles = num_particles
        
        # 粒子特征提取层
        self.particle_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 全局特征聚合
        particle_feature_dim = num_particles * 64
        self.net = nn.Sequential(
            nn.Linear(particle_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor输出层
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, 1.0)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        
        # 输出层使用较小的初始化
        torch.nn.init.uniform_(self.action_head.weight, -3e-3, 3e-3)
        if self.action_head.bias is not None:
            torch.nn.init.uniform_(self.action_head.bias, -3e-3, 3e-3)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            obs: shape (batch_size, num_particles, 3)
        Returns:
            action: shape (batch_size, action_dim), 范围[0, 1]
        """
        batch_size = obs.shape[0]
        
        # 处理每个粒子的特征
        particles_flat = obs.reshape(-1, 3)
        particle_features = self.particle_encoder(particles_flat)
        particle_features = particle_features.reshape(batch_size, self.num_particles, 64)
        global_features = particle_features.reshape(batch_size, -1)
        
        # 全局特征编码
        features = self.net(global_features)
        
        # 输出动作，使用sigmoid确保在[0,1]范围内
        action = torch.sigmoid(self.action_head(features))
        
        return action


class DDPGCriticParticle(nn.Module):
    """DDPG算法的Critic网络 - 粒子输入版本"""
    
    def __init__(self, num_particles: int, action_dim: int, hidden_dim: int = 512):
        super(DDPGCriticParticle, self).__init__()
        self.num_particles = num_particles
        
        # 粒子特征提取层
        self.particle_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 状态-动作特征融合网络
        particle_feature_dim = num_particles * 64
        self.net = nn.Sequential(
            nn.Linear(particle_feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, 1.0)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            obs: shape (batch_size, num_particles, 3)
            action: shape (batch_size, action_dim)
        Returns:
            q_value: shape (batch_size, 1)
        """
        batch_size = obs.shape[0]
        
        # 处理每个粒子的特征
        particles_flat = obs.reshape(-1, 3)
        particle_features = self.particle_encoder(particles_flat)
        particle_features = particle_features.reshape(batch_size, self.num_particles, 64)
        global_features = particle_features.reshape(batch_size, -1)
        
        # 状态-动作特征融合
        state_action = torch.cat([global_features, action], dim=-1)
        q_value = self.net(state_action)
        
        return q_value


class DDPGActorDepth(nn.Module):
    """DDPG算法的Actor网络 - 深度图输入版本"""
    
    def __init__(self, obs_shape: Tuple[int, int, int], action_dim: int, hidden_dim: int = 512):
        super(DDPGActorDepth, self).__init__()
        
        # 卷积特征提取
        self.conv_net = nn.Sequential(
            nn.Conv2d(obs_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        conv_output_size = self._get_conv_output_size(obs_shape)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor输出层
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # 输出层初始化
        torch.nn.init.uniform_(self.action_head.weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.action_head.bias, -3e-3, 3e-3)
    
    def _get_conv_output_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
            output = self.conv_net(dummy_input)
            return output.shape[1]
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            obs: shape (batch_size, height, width, channels) 或 (batch_size, channels, height, width)
        Returns:
            action: shape (batch_size, action_dim), 范围[0, 1]
        """
        # 确保输入格式为 (batch_size, channels, height, width)
        if obs.dim() == 4 and obs.shape[-1] in [1, 3]:
            obs = obs.permute(0, 3, 1, 2)
        
        conv_features = self.conv_net(obs)
        features = self.fc(conv_features)
        action = torch.sigmoid(self.action_head(features))
        
        return action


class DDPGCriticDepth(nn.Module):
    """DDPG算法的Critic网络 - 深度图输入版本"""
    
    def __init__(self, obs_shape: Tuple[int, int, int], action_dim: int, hidden_dim: int = 512):
        super(DDPGCriticDepth, self).__init__()
        
        # 卷积特征提取
        self.conv_net = nn.Sequential(
            nn.Conv2d(obs_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        conv_output_size = self._get_conv_output_size(obs_shape)
        
        # 状态-动作特征融合网络
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def _get_conv_output_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
            output = self.conv_net(dummy_input)
            return output.shape[1]
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            obs: shape (batch_size, height, width, channels) 或 (batch_size, channels, height, width)
            action: shape (batch_size, action_dim)
        Returns:
            q_value: shape (batch_size, 1)
        """
        # 确保输入格式为 (batch_size, channels, height, width)
        if obs.dim() == 4 and obs.shape[-1] in [1, 3]:
            obs = obs.permute(0, 3, 1, 2)
        
        conv_features = self.conv_net(obs)
        state_action = torch.cat([conv_features, action], dim=-1)
        q_value = self.fc(state_action)
        
        return q_value


class OUNoise:
    """Ornstein-Uhlenbeck噪声，用于DDPG的动作探索"""
    
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        
    def reset(self):
        """重置噪声状态"""
        self.state = np.ones(self.action_dim) * self.mu
        
    def sample(self) -> np.ndarray:
        """采样噪声"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class DDPGAgent:
    """DDPG算法实现"""
    
    def __init__(self, 
                 num_particles: int = None,
                 obs_shape: Tuple[int, int, int] = None,
                 action_dim: int = 4,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 hidden_dim: int = 512,
                 buffer_capacity: int = 1000000,
                 batch_size: int = 256,
                 noise_std: float = 0.2,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim
        
        # 根据输入类型选择网络
        if num_particles is not None:
            # 粒子位置输入
            self.actor = DDPGActorParticle(num_particles, action_dim, hidden_dim).to(device)
            self.target_actor = DDPGActorParticle(num_particles, action_dim, hidden_dim).to(device)
            self.critic = DDPGCriticParticle(num_particles, action_dim, hidden_dim).to(device)
            self.target_critic = DDPGCriticParticle(num_particles, action_dim, hidden_dim).to(device)
        elif obs_shape is not None:
            # 深度图输入
            self.actor = DDPGActorDepth(obs_shape, action_dim, hidden_dim).to(device)
            self.target_actor = DDPGActorDepth(obs_shape, action_dim, hidden_dim).to(device)
            self.critic = DDPGCriticDepth(obs_shape, action_dim, hidden_dim).to(device)
            self.target_critic = DDPGCriticDepth(obs_shape, action_dim, hidden_dim).to(device)
        else:
            raise ValueError("Must specify either num_particles or obs_shape")
        
        # 初始化目标网络
        self._hard_update(self.target_actor, self.actor)
        self._hard_update(self.target_critic, self.critic)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            obs_dim_or_shape=num_particles or obs_shape,
            action_dim=action_dim
        )
        
        # OU噪声
        self.noise = OUNoise(action_dim, sigma=noise_std)
        
        # 打印网络参数量
        self.print_network_parameters()
        
    def select_action(self, state: np.ndarray, add_noise: bool = True, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]
            
            if add_noise and not deterministic:
                noise = self.noise.sample()
                action = np.clip(action + noise, 0.0, 1.0)
            
            return action
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float, 
                        next_state: np.ndarray, done: bool):
        """存储经验到回放缓冲区"""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """更新网络参数"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # 从回放缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        # 更新Critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + (self.gamma * target_q * ~dones)
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        new_actions = self.actor(states)
        actor_loss = -self.critic(states, new_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': current_q.mean().item()
        }
    
    def reset_noise(self):
        """重置噪声"""
        self.noise.reset()
    
    def print_network_parameters(self):
        """打印DDPG网络参数量"""
        actor_params = sum(p.numel() for p in self.actor.parameters())
        target_actor_params = sum(p.numel() for p in self.target_actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        target_critic_params = sum(p.numel() for p in self.target_critic.parameters())
        
        total_params = actor_params + target_actor_params + critic_params + target_critic_params
        
        print(f"=== DDPG网络参数统计 ===")
        print(f"Actor网络: {actor_params:,} 参数")
        print(f"Target Actor网络: {target_actor_params:,} 参数")
        print(f"Critic网络: {critic_params:,} 参数")
        print(f"Target Critic网络: {target_critic_params:,} 参数")
        print(f"总参数量: {total_params:,}")
        print(f"参数大小: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")
        print("=" * 30)
    
    def _soft_update(self, target, source):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def _hard_update(self, target, source):
        """硬更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"DDPG模型已保存至: {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"DDPG模型已从 {filepath} 加载")

# DQN算法实现
class DQNNetwork(nn.Module):
    """DQN网络 - 输入粒子坐标，为每个粒子输出Q值"""
    
    def __init__(self, num_particles: int, hidden_dim: int = 512):
        super(DQNNetwork, self).__init__()
        self.num_particles = num_particles
        
        # 粒子特征提取层 - 处理每个粒子的3D坐标
        self.particle_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 全局特征聚合
        particle_feature_dim = num_particles * 64
        self.global_encoder = nn.Sequential(
            nn.Linear(particle_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 为pick和place分别输出Q值
        self.pick_q_head = nn.Linear(hidden_dim, num_particles)  # 为每个粒子输出pick的Q值
        self.place_q_head = nn.Linear(hidden_dim, num_particles)  # 为每个粒子输出place的Q值
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, 1.0)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        
        # 输出层使用较小的初始化
        torch.nn.init.uniform_(self.pick_q_head.weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.place_q_head.weight, -3e-3, 3e-3)
        if self.pick_q_head.bias is not None:
            torch.nn.init.uniform_(self.pick_q_head.bias, -3e-3, 3e-3)
        if self.place_q_head.bias is not None:
            torch.nn.init.uniform_(self.place_q_head.bias, -3e-3, 3e-3)
        
    def forward(self, obs: torch.Tensor) -> tuple:
        """
        前向传播
        Args:
            obs: shape (batch_size, num_particles, 3)
        Returns:
            pick_q_values: shape (batch_size, num_particles) - 每个粒子作为pick点的Q值
            place_q_values: shape (batch_size, num_particles) - 每个粒子作为place点的Q值
        """
        batch_size = obs.shape[0]
        
        # 处理每个粒子的特征
        particles_flat = obs.reshape(-1, 3)  # (batch_size * num_particles, 3)
        particle_features = self.particle_encoder(particles_flat)  # (batch_size * num_particles, 64)
        particle_features = particle_features.reshape(batch_size, self.num_particles, 64)
        global_features = particle_features.reshape(batch_size, -1)  # (batch_size, num_particles * 64)
        
        # 全局特征编码
        features = self.global_encoder(global_features)  # (batch_size, hidden_dim)
        
        # 输出每个粒子的Q值
        pick_q_values = self.pick_q_head(features)  # (batch_size, num_particles)
        place_q_values = self.place_q_head(features)  # (batch_size, num_particles)
        
        return pick_q_values, place_q_values


class ReplayBufferDQN:
    """DQN专用的经验回放缓冲区"""
    
    def __init__(self, capacity: int, num_particles: int):
        self.capacity = capacity
        self.num_particles = num_particles
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """从缓冲区采样经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),      # (batch_size, num_particles, 3)
            np.array(actions),     # (batch_size, 2) - [pick_idx, place_idx]
            np.array(rewards),     # (batch_size,)
            np.array(next_states), # (batch_size, num_particles, 3)
            np.array(dones)        # (batch_size,)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN算法实现 - 专为离散动作空间设计"""
    
    def __init__(self, 
                 num_particles: int,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 10000,
                 hidden_dim: int = 512,
                 buffer_capacity: int = 100000,
                 batch_size: int = 256,
                 target_update_frequency: int = 1000,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.num_particles = num_particles
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # 网络初始化
        self.q_network = DQNNetwork(num_particles, hidden_dim).to(device)
        self.target_network = DQNNetwork(num_particles, hidden_dim).to(device)
        
        # 初始化目标网络
        self._hard_update()
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBufferDQN(buffer_capacity, num_particles)
        
        # 训练统计
        self.total_steps = 0
        self.update_count = 0
        
        # 打印网络参数量
        self.print_network_parameters()
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        选择动作 - epsilon-greedy策略
        Args:
            state: shape (num_particles, 3)
            deterministic: 是否使用确定性策略（用于评估）
        Returns:
            action: shape (2,) - [pick_idx, place_idx]
        """
        # 计算当前epsilon
        if deterministic:
            epsilon = 0.0
        else:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                     np.exp(-1. * self.total_steps / self.epsilon_decay)
        
        if np.random.random() < epsilon:
            # 随机选择动作
            pick_idx = np.random.randint(0, self.num_particles)
            place_idx = np.random.randint(0, self.num_particles)
            return np.array([pick_idx, place_idx], dtype=np.float32)
        else:
            # 使用Q网络选择动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                pick_q_values, place_q_values = self.q_network(state_tensor)
                
                pick_idx = torch.argmax(pick_q_values, dim=1).cpu().numpy()[0]
                place_idx = torch.argmax(place_q_values, dim=1).cpu().numpy()[0]
                
                return np.array([pick_idx, place_idx], dtype=np.float32)
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, 
                        next_state: np.ndarray, done: bool):
        """存储经验到回放缓冲区"""
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1
    
    def update(self) -> Dict[str, float]:
        """更新Q网络"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # 从回放缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)  # 转换为LongTensor用于索引
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 当前Q值
        pick_q_values, place_q_values = self.q_network(states)
        
        # 选择对应动作的Q值
        pick_actions = actions[:, 0]  # pick动作索引
        place_actions = actions[:, 1]  # place动作索引
        
        current_pick_q = pick_q_values.gather(1, pick_actions.unsqueeze(1)).squeeze(1)
        current_place_q = place_q_values.gather(1, place_actions.unsqueeze(1)).squeeze(1)
        
        # 目标Q值 (Double DQN)
        with torch.no_grad():
            # 使用主网络选择动作
            next_pick_q, next_place_q = self.q_network(next_states)
            next_pick_actions = torch.argmax(next_pick_q, dim=1)
            next_place_actions = torch.argmax(next_place_q, dim=1)
            
            # 使用目标网络评估动作
            target_next_pick_q, target_next_place_q = self.target_network(next_states)
            target_next_pick_q = target_next_pick_q.gather(1, next_pick_actions.unsqueeze(1)).squeeze(1)
            target_next_place_q = target_next_place_q.gather(1, next_place_actions.unsqueeze(1)).squeeze(1)
            
            # 组合pick和place的Q值（取平均）
            target_next_q = (target_next_pick_q + target_next_place_q) / 2.0
            target_q = rewards + (self.gamma * target_next_q * ~dones)
        
        # 组合当前pick和place的Q值（取平均）
        current_q = (current_pick_q + current_place_q) / 2.0
        
        # 计算损失
        loss = F.mse_loss(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self._hard_update()
        
        # 计算当前epsilon
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                 np.exp(-1. * self.total_steps / self.epsilon_decay)
        
        return {
            'q_loss': loss.item(),
            'q_value_mean': current_q.mean().item(),
            'epsilon': epsilon,
            'pick_q_mean': current_pick_q.mean().item(),
            'place_q_mean': current_place_q.mean().item(),
            'target_q_mean': target_q.mean().item()
        }
    
    def print_network_parameters(self):
        """打印DQN网络参数量"""
        q_network_params = sum(p.numel() for p in self.q_network.parameters())
        target_network_params = sum(p.numel() for p in self.target_network.parameters())
        
        total_params = q_network_params + target_network_params
        
        print(f"=== DQN网络参数统计 ===")
        print(f"Q网络: {q_network_params:,} 参数")
        print(f"目标网络: {target_network_params:,} 参数")
        print(f"总参数量: {total_params:,}")
        print(f"参数大小: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")
        print("=" * 30)
    
    def _hard_update(self):
        """硬更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count
        }, filepath)
        print(f"DQN模型已保存至: {filepath}")
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.update_count = checkpoint.get('update_count', 0)
        print(f"DQN模型已从 {filepath} 加载")