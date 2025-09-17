import os
import time
from dataclasses import dataclass
import random
import cv2
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
import imageio
from tqdm import tqdm
import pickle

from daxbench.core.engine.cloth_simulator import ClothState
from daxbench.core.engine.usdrender.mesh_usd import create_usd_cloth_scene
from daxbench.core.envs.basic.cloth_env import ClothEnv
from daxbench.core.utils.util import get_expert_start_end_cloth
from sklearn.cluster import KMeans

from expert.fold_direction_langchain import *

from icecream import ic as print

my_path = os.path.dirname(os.path.abspath(__file__))


def spatial_sampling(particles: np.ndarray, target_num: int, method: str = "kmeans", **kwargs) -> np.ndarray:
    """
    基于空间分布的粒子采样算法
    
    Args:
        particles: 输入粒子坐标，形状为 (num_particles, 3)
        target_num: 目标采样数量
        method: 采样方法 ("kmeans", "grid", "farthest_point", "farthest_point_jax", "farthest_point_approx", "farthest_point_hybrid", "farthest_point_fast")
        **kwargs: 额外参数
            - subsample_ratio: 用于farthest_point_approx方法的预采样比例 (default: 0.5)
    
    Returns:
        采样后的粒子坐标，形状为 (target_num, 3)
    """
    start_time = time.time()
    # 如果数据量很大，打印调试信息
    if len(particles) > 5000 or target_num > 100:
        print(f"Large sampling: {len(particles)} particles -> {target_num}, method: {method}")
    
    if len(particles) <= target_num:
        # 如果粒子数不足，用最后一个粒子填充
        if len(particles) < target_num:
            padding_needed = target_num - len(particles)
            last_particle = particles[-1:, :]
            padding = np.repeat(last_particle, padding_needed, axis=0)
            return np.concatenate([particles, padding], axis=0)
        return particles
    particles = np.array(particles)
    
    if method == "kmeans":
        return _kmeans_sampling(particles, target_num)
    elif method == "grid":
        return _grid_sampling(particles, target_num)
    elif method == "farthest_point":
        return _farthest_point_sampling(particles, target_num)
    elif method == "farthest_point_jax":
        # JAX版本，支持GPU加速 - 优化的调用方式
        # 减少不必要的类型转换，直接在JAX中处理
        particles_jax = jnp.array(particles)
        result = _farthest_point_sampling_jax(particles_jax, target_num)
        return np.asarray(result)  # 使用asarray而非array，更高效
    elif method == "farthest_point_approx":
        # 近似版本，速度更快
        subsample_ratio = kwargs.get('subsample_ratio', 0.5)
        return _farthest_point_sampling_approximate(particles, target_num, subsample_ratio)
    elif method == "farthest_point_hybrid":
        # 混合策略：小数据用JAX，大数据用优化的近似方法
        if len(particles) <= 1000 and target_num <= 50:
            particles_jax = jnp.array(particles)
            result = _farthest_point_sampling_jax(particles_jax, target_num)
            return np.asarray(result)
        else:
            # 对于大数据，使用改进的近似方法
            return _farthest_point_sampling_approximate(particles, target_num, 0.3)
    elif method == "farthest_point_fast":
        # 超快速版本：专为训练时高频调用优化
        return _farthest_point_sampling_fast(particles, target_num)
    else:
        raise ValueError(f"Unsupported sampling method: {method}. "
                        f"Supported methods: 'kmeans', 'grid', 'farthest_point', 'farthest_point_jax', 'farthest_point_approx', 'farthest_point_hybrid', 'farthest_point_fast'")

def _kmeans_sampling(particles: np.ndarray, target_num: int) -> np.ndarray:
    """
    使用K-means聚类进行空间采样
    选择每个簇中最接近簇心的点作为代表
    """
    try:
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=target_num, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(particles)
        cluster_centers = kmeans.cluster_centers_
        
        sampled_particles = []
        for i in range(target_num):
            # 找到属于第i个簇的所有粒子
            cluster_mask = cluster_labels == i
            cluster_particles = particles[cluster_mask]
            
            if len(cluster_particles) > 0:
                # 找到最接近簇心的粒子
                distances = np.linalg.norm(cluster_particles - cluster_centers[i], axis=1)
                closest_idx = np.argmin(distances)
                sampled_particles.append(cluster_particles[closest_idx])
            else:
                # 如果某个簇为空，使用对应的簇心
                sampled_particles.append(cluster_centers[i])
        
        return np.array(sampled_particles)
    except Exception as e:
        # 如果K-means失败，回退到网格采样
        print(f"K-means采样失败，回退到网格采样: {e}")
        return _grid_sampling(particles, target_num)

def _grid_sampling(particles: np.ndarray, target_num: int) -> np.ndarray:
    """
    基于空间网格的采样方法
    将空间划分为网格，从每个网格中选择一个代表点
    """
    # 计算粒子的边界
    min_coords = np.min(particles, axis=0)
    max_coords = np.max(particles, axis=0)
    
    # 计算每个维度的网格数
    # 尝试创建一个近似立方体的网格
    total_volume = np.prod(max_coords - min_coords + 1e-6)
    grid_size = max(1, int(np.ceil(target_num ** (1/3))))
    
    # 创建网格
    x_bins = np.linspace(min_coords[0], max_coords[0], grid_size + 1)
    y_bins = np.linspace(min_coords[1], max_coords[1], grid_size + 1)
    z_bins = np.linspace(min_coords[2], max_coords[2], grid_size + 1)
    
    # 将粒子分配到网格
    x_indices = np.digitize(particles[:, 0], x_bins) - 1
    y_indices = np.digitize(particles[:, 1], y_bins) - 1
    z_indices = np.digitize(particles[:, 2], z_bins) - 1
    
    # 确保索引在有效范围内
    x_indices = np.clip(x_indices, 0, grid_size - 1)
    y_indices = np.clip(y_indices, 0, grid_size - 1)
    z_indices = np.clip(z_indices, 0, grid_size - 1)
    
    # 为每个网格选择一个代表点
    sampled_particles = []
    grid_dict = {}
    
    # 将粒子按网格分组
    for i, (x_idx, y_idx, z_idx) in enumerate(zip(x_indices, y_indices, z_indices)):
        grid_key = (x_idx, y_idx, z_idx)
        if grid_key not in grid_dict:
            grid_dict[grid_key] = []
        grid_dict[grid_key].append(i)
    
    # 从每个非空网格中选择一个粒子（选择最接近网格中心的）
    for grid_key, particle_indices in grid_dict.items():
        if len(sampled_particles) >= target_num:
            break
            
        x_idx, y_idx, z_idx = grid_key
        # 计算网格中心
        grid_center = np.array([
            (x_bins[x_idx] + x_bins[x_idx + 1]) / 2,
            (y_bins[y_idx] + y_bins[y_idx + 1]) / 2,
            (z_bins[z_idx] + z_bins[z_idx + 1]) / 2
        ])
        
        # 找到最接近网格中心的粒子
        particle_indices = np.array(particle_indices)
        grid_particles = particles[particle_indices]
        distances = np.linalg.norm(grid_particles - grid_center, axis=1)
        closest_idx = np.argmin(distances)
        sampled_particles.append(grid_particles[closest_idx])
    
    # 如果采样数量不足，添加剩余的粒子
    if len(sampled_particles) < target_num:
        # 找到未被采样的粒子
        sampled_set = set()
        for grid_key, particle_indices in grid_dict.items():
            if len(sampled_set) < len(sampled_particles):
                x_idx, y_idx, z_idx = grid_key
                grid_center = np.array([
                    (x_bins[x_idx] + x_bins[x_idx + 1]) / 2,
                    (y_bins[y_idx] + y_bins[y_idx + 1]) / 2,
                    (z_bins[z_idx] + z_bins[z_idx + 1]) / 2
                ])
                grid_particles = particles[particle_indices]
                distances = np.linalg.norm(grid_particles - grid_center, axis=1)
                closest_idx = np.argmin(distances)
                sampled_set.add(particle_indices[closest_idx])
        
        # 添加更多粒子直到达到目标数量
        remaining_indices = [i for i in range(len(particles)) if i not in sampled_set]
        additional_needed = target_num - len(sampled_particles)
        
        if len(remaining_indices) >= additional_needed:
            # 从剩余粒子中均匀采样
            step = len(remaining_indices) // additional_needed
            for i in range(additional_needed):
                idx = remaining_indices[i * step]
                sampled_particles.append(particles[idx])
        else:
            # 如果剩余粒子不足，用最后一个粒子填充
            sampled_particles.extend(particles[remaining_indices])
            while len(sampled_particles) < target_num:
                sampled_particles.append(sampled_particles[-1])
    
    return np.array(sampled_particles[:target_num])

def _farthest_point_sampling(particles: np.ndarray, target_num: int) -> np.ndarray:
    """
    最远点采样算法 - 优化版本
    逐步选择与已选点距离最远的点，使用向量化操作加速
    """
    if len(particles) <= target_num:
        return particles
    
    n_points = len(particles)
    sampled_indices = []
    
    # 选择第一个点（选择最接近中心的点）
    center = np.mean(particles, axis=0)
    distances_to_center = np.linalg.norm(particles - center, axis=1)
    first_idx = np.argmin(distances_to_center)
    sampled_indices.append(first_idx)
    
    # 初始化距离数组：每个点到已选点的最小距离
    min_distances = np.full(n_points, np.inf)
    
    # 逐步选择最远点
    for _ in range(target_num - 1):
        last_selected = sampled_indices[-1]
        
        # 向量化计算：更新所有点到最新选择点的距离
        distances_to_last = np.linalg.norm(particles - particles[last_selected], axis=1)
        min_distances = np.minimum(min_distances, distances_to_last)
        
        # 将已选择的点的距离设为0，避免重复选择
        min_distances[sampled_indices] = 0
        
        # 选择距离最大的点
        farthest_idx = np.argmax(min_distances)
        sampled_indices.append(farthest_idx)
    
    return particles[sampled_indices]

def _farthest_point_sampling_jax(particles, target_num: int):
    """
    JAX版本的最远点采样算法 - 高性能实现
    避免重复JIT编译和大内存分配，解决卡顿问题
    """
    # 在外部处理边界条件
    if particles.shape[0] <= target_num:
        return particles
    
    n_points = particles.shape[0]
    
    # 选择第一个点（最接近中心的点）
    center = jnp.mean(particles, axis=0)
    distances_to_center = jnp.linalg.norm(particles - center, axis=1)
    first_idx = jnp.argmin(distances_to_center)
    
    # 使用简化的实现，避免复杂的JIT操作
    sampled_indices = [int(first_idx)]
    min_distances = jnp.full(n_points, jnp.inf)
    
    # 简化的循环实现，避免重复JIT编译
    for _ in range(target_num - 1):
        last_selected = sampled_indices[-1]
        
        # JAX向量化计算距离
        distances_to_last = jnp.linalg.norm(particles - particles[last_selected], axis=1)
        min_distances = jnp.minimum(min_distances, distances_to_last)
        
        # 高效地将已选择的点的距离设为0
        # 使用简单的at操作，避免复杂的广播
        for idx in sampled_indices:
            min_distances = min_distances.at[idx].set(0.0)
        
        # 选择距离最大的点
        farthest_idx = int(jnp.argmax(min_distances))
        sampled_indices.append(farthest_idx)
    
    return particles[jnp.array(sampled_indices)]

def _farthest_point_sampling_fast(particles: np.ndarray, target_num: int) -> np.ndarray:
    """
    超快速最远点采样算法 - 专为训练时高频调用优化
    使用激进的近似策略，优先速度而非精度
    """
    if len(particles) <= target_num:
        # 如果粒子数不足，用最后一个粒子填充
        if len(particles) < target_num:
            padding_needed = target_num - len(particles)
            last_particle = particles[-1:, :]
            padding = np.repeat(last_particle, padding_needed, axis=0)
            return np.concatenate([particles, padding], axis=0)
        return particles
    
    # 对于大数据集，使用更激进的预采样
    if len(particles) > target_num * 5:
        # 先随机预采样到合理大小
        subsample_size = min(target_num * 3, len(particles) // 2)
        subsample_indices = np.random.choice(len(particles), subsample_size, replace=False)
        particles = particles[subsample_indices]
    
    # 使用简化的最远点采样
    n_points = len(particles)
    
    # 选择第一个点（随机选择以提高速度）
    first_idx = np.random.randint(0, n_points)
    sampled_indices = [first_idx]
    
    # 只计算部分点的距离，进一步提速
    for _ in range(target_num - 1):
        last_selected = sampled_indices[-1]
        
        # 计算距离
        distances = np.linalg.norm(particles - particles[last_selected], axis=1)
        
        # 将已选择的点的距离设为0
        for idx in sampled_indices:
            distances[idx] = 0.0
        
        # 选择距离最大的点
        farthest_idx = int(np.argmax(distances))
        sampled_indices.append(farthest_idx)
    
    return particles[sampled_indices]

def _farthest_point_sampling_approximate(particles: np.ndarray, target_num: int, subsample_ratio: float = 0.5) -> np.ndarray:
    """
    近似最远点采样算法 - 通过预采样减少计算量
    
    Args:
        particles: 输入粒子坐标
        target_num: 目标采样数量  
        subsample_ratio: 预采样比例，在0-1之间，越小速度越快但精度略低
    """
    if len(particles) <= target_num:
        return particles
    
    # 如果粒子数量很大，先进行预采样
    if len(particles) > target_num * 10 and subsample_ratio < 1.0:
        # 随机预采样，减少候选点数量
        subsample_size = max(target_num * 3, int(len(particles) * subsample_ratio))
        subsample_indices = np.random.choice(len(particles), subsample_size, replace=False)
        particles_subset = particles[subsample_indices]
        
        # 在子集上进行最远点采样
        sampled_subset = _farthest_point_sampling(particles_subset, target_num)
        return sampled_subset
    else:
        # 直接使用优化的最远点采样
        return _farthest_point_sampling(particles, target_num)


@dataclass
class DefaultConf:
    N = 80
    cell_size = 1.0 / N
    gravity = 0.7
    stiffness = 10000
    damping = 300
    dt = 1e-3
    max_v = 1
    small_num = 1e-8
    mu = 10  # friction
    seed = 1
    size = int(N / 5.0)
    mem_saving_level = 1
    # 0:fast but requires more memory, not recommended
    # 1:lesser memory, but faster
    # 2:much lesser memory but much slower
    batch_size = 10
    cloth_type = 'tshirt'
    task = "S_Corner_All_Middle"
    id_range = list(range(0, 1))
    goal_path = f"{my_path}/goals/{task}/goal.npy"
    use_substep_obs = True
    record_video = False
    sampling_method = 'farthest_point'
    num_sampled_particles = 15
    # N = 200
    # cell_size = 1.0 / N
    # gravity = 0.7
    # stiffness = 7000
    # damping = 2
    # dt = 0.5e-3
    # max_v = 2.
    # small_num = 1e-8
    # mu = 1.5  # friction
    # seed = 1
    # size = int(N / 5.0)
    # mem_saving_level = 2
    # # 0:fast but requires more memory, not recommended
    # # 1:lesser memory, but faster
    # # 2:much lesser memory but much slower
    # task = "fold_tshirt"
    # goal_path = f"{my_path}/goals/{task}/goal.npy"
    # use_substep_obs = True
    # record_video = False


FoldTshirtConfig = DefaultConf


class FoldEnv(ClothEnv):

    def __init__(self, conf=None, aux_reward=False, seed=1, reward_type="final_goal", subgoals_dir=None):
        conf = DefaultConf() if conf is None else conf
        task_steps = globals()[conf.task].steps()
        self.max_subgoal_steps = 3
        max_steps = self.max_subgoal_steps * task_steps


        self.batch_size = conf.batch_size
        self.cloth_type = conf.cloth_type
        self.reward_type = reward_type  # "final_goal", "subgoal", "combined", "final_goal_delta", "subgoal_delta"
        self.subgoals_dir = os.path.join('oracle', conf.task)
        self.sampling_method = conf.sampling_method
        self.num_sampled_particles = conf.num_sampled_particles
        
        # Always enable dual_arm for rendering
        init_start_time = time.time()
        self.chosen_ids = [conf.id_range[(i) % len(conf.id_range)] for i in range(self.batch_size)]
        print(self.chosen_ids)
        super().__init__(conf, self.batch_size, max_steps, aux_reward)
        print('init time: ', time.time() - init_start_time)
        self.observation_size = 1082
        self.episode_data = {'states': []}
        self.record_video = conf.record_video
        
        # 初始化子目标相关变量
        self.subgoals = []
        self.current_subgoal_idx = np.zeros(self.batch_size, dtype=int)
        self.current_subgoal_step = np.zeros(self.batch_size, dtype=int)

        # 初始化子目标状态变量，避免AttributeError
        self.subgoals_reached = None
        self.episode_success = None

        self.step_count = 0
        
        # 初始化距离跟踪变量，用于基于距离变化的奖励计算
        self.prev_final_goal_distance = None
        self.prev_subgoal_distance = None
        
        # 加载子目标（如果提供了路径）
        if self.subgoals_dir and self.reward_type in ["subgoal", "combined", "subgoal_delta", "binary_subgoal"]:
            self._load_subgoals()
        
        self.init_compile()
    
    def build_reset(self):
        """重写reset方法以处理vmap问题"""
        from jax import random
        init_state = self.simulator.reset_jax()

        def reset(key):
            key, _ = random.split(key)
            # new_x = init_state.x.at[..., [0, 2]].add(random.normal(key, (2,)) * 0.05)
            new_x = init_state.x
            state = init_state._replace(x=new_x)
            
            # 确保状态有正确的batch维度后再调用get_obs
            try:
                obs = self.get_obs(state)
                return obs, state
            except Exception as e:
                print(f"Warning: get_obs failed in reset: {e}")
                # 返回零观察作为fallback
                obs = jnp.zeros((self.batch_size, self.observation_size))
                return obs, state

        return reset
    
    def init_compile(self):
        obs, state = self.reset(self.simulator.key_global)
        
        # 重置gripper位置
        state = self.reset_gripper_positions(state)
        
        # 只使用6维动作进行编译测试
        actions = np.zeros((self.batch_size, 12))
        _, _, _, info = self.step_diff(actions, state)
        
        # 重置gripper位置
        info['state'] = self.reset_gripper_positions(info['state'])
        
        self.info = info

        if self.record_video:
            self.episode_data['states'].append(self.info['state_list'])

        # 处理关键点...
        if self.cloth_type == 'tshirt':
            for i in range(self.batch_size):
                if self.gt_keypoints_list[i] is None:
                    self.gt_keypoints_list[i] = []
                    continue
                self.gt_keypoints_list[i] = [self.gt_keypoints_list[i]['bottom_left'], self.gt_keypoints_list[i]['left_armpit'], self.gt_keypoints_list[i]['left_sleeve_bottom'], self.gt_keypoints_list[i]['left_sleeve_top'], self.gt_keypoints_list[i]['left_shoulder_top'], self.gt_keypoints_list[i]['left_collar'], self.gt_keypoints_list[i]['spine_top'], self.gt_keypoints_list[i]['right_collar'], self.gt_keypoints_list[i]['right_shoulder_top'], self.gt_keypoints_list[i]['right_sleeve_top'], self.gt_keypoints_list[i]['right_sleeve_bottom'], self.gt_keypoints_list[i]['right_armpit'], self.gt_keypoints_list[i]['bottom_right']]
        elif self.cloth_type == 'pant':
            for i in range(self.batch_size):
                if self.gt_keypoints_list[i] is None:
                    self.gt_keypoints_list[i] = []
                    continue
                self.gt_keypoints_list[i] = [self.gt_keypoints_list[i]['left_leg_right'], self.gt_keypoints_list[i]['left_leg_left'], self.gt_keypoints_list[i]['top_left'], self.gt_keypoints_list[i]['top_right'], self.gt_keypoints_list[i]['right_leg_right'], self.gt_keypoints_list[i]['right_leg_left'], self.gt_keypoints_list[i]['crotch']]
        elif self.cloth_type == 'rectangle' or self.cloth_type == 'square':
            for i in range(self.batch_size):
                if self.gt_keypoints_list[i] is None:
                    self.gt_keypoints_list[i] = []
                    continue
                self.gt_keypoints_list[i] = [self.gt_keypoints_list[i]['top_left'], self.gt_keypoints_list[i]['top_right'], self.gt_keypoints_list[i]['bottom_right'], self.gt_keypoints_list[i]['bottom_left']]
        else:
            raise NotImplementedError
        
        for i in range(len(self.gt_keypoints_list)):
            for j in range(len(self.gt_keypoints_list[i])):
                self.gt_keypoints_list[i][j] = np.array(self.gt_keypoints_list[i][j])
    
    def _load_subgoals(self):
        """加载子目标文件，每个batch ID对应自己的子目标序列"""
        # subgoals是一个列表，每个元素对应一个batch环境的子目标序列
        # subgoals[i] = [subgoal_0, subgoal_1, subgoal_2, ...] 对应第i个batch环境
        self.subgoals = []
        
        for batch_idx, chosen_id in enumerate(self.chosen_ids):
            # 每个ID的子目标路径: oracle/task/id/subgoals/
            id_subgoals_dir = os.path.join(self.subgoals_dir, str(chosen_id), 'subgoals')
            
            if not os.path.exists(id_subgoals_dir):
                print(f"Warning: Subgoals directory {id_subgoals_dir} does not exist for ID {chosen_id}!")
                self.subgoals.append([])  # 空的子目标序列
                continue
            
            # 加载该ID的所有子目标文件
            subgoal_files = sorted([f for f in os.listdir(id_subgoals_dir) if f.endswith('.pkl')])
            id_subgoals = []
            
            for filename in subgoal_files:
                filepath = os.path.join(id_subgoals_dir, filename)
                try:
                    with open(filepath, 'rb') as f:
                        subgoal_data = pickle.load(f)
                        id_subgoals.append(np.array(subgoal_data))
                        # print(f"Loaded subgoal {filename} for ID {chosen_id}: shape {subgoal_data.shape}")
                except Exception as e:
                    print(f"Error loading subgoal {filename} for ID {chosen_id}: {e}")
            
            self.subgoals.append(id_subgoals)
            # print(f"ID {chosen_id}: loaded {len(id_subgoals)} subgoals")
        
        # print(f"Total loaded subgoals for {len(self.chosen_ids)} IDs")
    
    def _calculate_subgoal_reward(self, current_state):
        """计算批量子目标奖励，每个batch环境使用自己的子目标序列"""
        if len(self.subgoals) == 0:
            return np.zeros(self.batch_size), np.ones(self.batch_size, dtype=bool)
            
        rewards = np.zeros(self.batch_size)
        subgoals_reached = np.zeros(self.batch_size, dtype=bool)
        self.distances = np.zeros(self.batch_size)
        # 批量处理每个环境的子目标
        for i in range(self.batch_size):
            # 获取第i个环境的子目标序列
            if i >= len(self.subgoals) or len(self.subgoals[i]) == 0:
                rewards[i] = 0.0
                subgoals_reached[i] = True
                continue
            
            # 检查当前子目标索引是否超出范围
            if self.current_subgoal_idx[i] >= len(self.subgoals[i]):
                print(f'In index {i}', f'current_subgoal_idx: {self.current_subgoal_idx[i]}', f'len(self.subgoals[i]): {len(self.subgoals[i])}')
                rewards[i] = 0.0
                subgoals_reached[i] = False
                continue
                
            # 获取第i个环境的当前子目标
            current_subgoal = self.subgoals[i][self.current_subgoal_idx[i]]
            
            # 计算当前状态与子目标的粒子距离
            # 计算当前状态与子目标的粒子距离
            current_particles = np.array(current_state.x[i])  # 第i个batch的粒子
            
            # 使用点对点欧氏距离（两者粒子数一致）
            current_particles_j = jnp.array(current_particles)
            current_subgoal_j = jnp.array(current_subgoal)
            # print(current_particles_j.shape, current_subgoal_j.shape)  
            # distance = jnp.mean(jnp.linalg.norm(current_particles_j - current_subgoal_j, axis=-1))
            #chamfer distance
            from daxbench.core.utils.util import calc_chamfer
        
            distance = calc_chamfer(current_particles_j[None, ...], current_subgoal_j)
            # print(distance)
            # print(distance)
            # 奖励函数：距离越小奖励越大
            subgoal_threshold = 0.012

            rewards[i] = np.exp(-distance * 30) if distance < subgoal_threshold*2 else -1
            # print(rewards[i])
            
            self.distances[i] = distance
            # 检查是否达到子目标（距离阈值）
            if distance < subgoal_threshold:
                print(f"subgoals_reached[i]: {subgoals_reached[i]}", f"distance: {distance}")
            # else:
            #     print(f"distance: {distance}")
            subgoals_reached[i] = distance < subgoal_threshold  # 可调整阈值
        # print(self.distances)
        return rewards, subgoals_reached
    
    def _calculate_final_goal_delta_reward(self, current_state):
        """计算基于final goal距离变化的奖励"""
        from daxbench.core.utils.util import calc_chamfer
        
        # 计算当前状态与final goal的距离
        current_distance = calc_chamfer(current_state.x, self.goal)
        
        # 如果是第一步，初始化上一步距离
        if self.prev_final_goal_distance is None:
            self.prev_final_goal_distance = current_distance.copy()
            # 第一步返回小的正奖励
            return np.ones(self.batch_size) * 0.1
        
        # 计算距离变化：上一步距离 - 当前距离（越小越好，所以这样计算）
        distance_delta = self.prev_final_goal_distance - current_distance
        
        # 更新上一步距离
        self.prev_final_goal_distance = current_distance.copy()
        
        # 将距离变化转换为奖励，使用tanh函数将其限制在合理范围内
        # 距离减少时为正奖励，距离增加时为负奖励
        rewards = np.tanh(distance_delta * 5) * 2.0  # 乘以50放大变化，乘以2调整奖励范围到[-2, 2]
        
        return rewards
    
    def _calculate_subgoal_delta_reward(self, current_state):
        """计算基于subgoal距离变化的奖励"""
        if len(self.subgoals) == 0:
            return np.zeros(self.batch_size), np.ones(self.batch_size, dtype=bool)
        
        self.distances = np.zeros(self.batch_size)
        current_distances = np.zeros(self.batch_size)
        rewards = np.zeros(self.batch_size)
        subgoals_reached = np.zeros(self.batch_size, dtype=bool)
        
        # 计算每个环境当前与其子目标的距离
        for i in range(self.batch_size):
            if i >= len(self.subgoals) or len(self.subgoals[i]) == 0:
                current_distances[i] = 0.0
                subgoals_reached[i] = True
                continue
                
            if self.current_subgoal_idx[i] >= len(self.subgoals[i]):
                current_distances[i] = 0.0
                subgoals_reached[i] = False
                continue
                
            # 获取当前子目标
            current_subgoal = self.subgoals[i][self.current_subgoal_idx[i]]
            current_particles = np.array(current_state.x[i])
            
            # 计算距离
            current_particles_j = jnp.array(current_particles)
            current_subgoal_j = jnp.array(current_subgoal)
            # distance = jnp.mean(jnp.linalg.norm(current_particles_j - current_subgoal_j, axis=-1))
            from daxbench.core.utils.util import calc_chamfer
        
            distance = calc_chamfer(current_particles_j[None, ...], current_subgoal_j)
            current_distances[i] = float(distance)
            self.distances[i] = distance
            # 检查是否达到子目标
            subgoal_threshold = 0.012
            subgoals_reached[i] = distance < subgoal_threshold
        
        # 如果是第一步，初始化上一步距离
        if self.prev_subgoal_distance is None:
            self.prev_subgoal_distance = current_distances.copy()
            # 第一步返回小的正奖励
            return np.ones(self.batch_size) * 0.1, subgoals_reached
        
        # 计算距离变化
        distance_deltas = self.prev_subgoal_distance - current_distances
        
        # 更新上一步距离
        self.prev_subgoal_distance = current_distances.copy()
        
        # 将距离变化转换为奖励
        rewards = np.tanh(distance_deltas * 5) * 2.0
        
        return rewards, subgoals_reached
    
    def _update_subgoal_states(self, subgoals_reached):
        """更新子目标状态，每个环境独立处理"""
        episode_success = np.zeros(self.batch_size, dtype=bool)
        subgoal_advanced = np.zeros(self.batch_size, dtype=bool)
        subgoal_timeout = np.zeros(self.batch_size, dtype=bool)
        
        self.current_subgoal_step += 1
        
        for i in range(self.batch_size):
            if subgoals_reached[i]:
                # print(f"subgoals_reached[i]: {subgoals_reached[i]}")
                # 达到当前子目标，进入下一个
                self.current_subgoal_idx[i] += 1
                self.current_subgoal_step[i] = 0
                
                # 检查该环境是否完成所有子目标
                if (i >= len(self.subgoals) or 
                    len(self.subgoals[i]) == 0 or 
                    self.current_subgoal_idx[i] >= len(self.subgoals[i])):
                    # 所有子目标完成
                    episode_success[i] = True
                else:
                    subgoal_advanced[i] = True
                    
            elif self.current_subgoal_step[i] >= self.max_subgoal_steps:
                # 子目标步数超限，重置当前子目标
                self.current_subgoal_step[i] = 0
                subgoal_timeout[i] = True
        
        return episode_success, subgoal_advanced, subgoal_timeout
  
    
    
    def reset_env(self):
        self.step_count = 0
        self.episode_data = {'states': []}
        obs, state = self.reset(self.simulator.key_global)
        actions = np.zeros((self.batch_size, 12))
        _, _, _, info = self.step_diff(actions, state)
        self.info = info
        
        # 重置子目标状态
        if self.reward_type in ["subgoal", "combined", "subgoal_delta", "binary_subgoal"]:
            self.reset_subgoal_states()
        
        # 重置距离跟踪变量
        if self.reward_type in ["final_goal_delta", "subgoal_delta"]:
            self.prev_final_goal_distance = None
            self.prev_subgoal_distance = None

        if self.record_video:
            self.episode_data['states'].append(self.info['state_list'])

    def create_cloth_mask(self, conf):
        masks = []
        self.gt_keypoints_list = []
        for mid in self.chosen_ids:
            img_path, gt_keypoints = self._resolve_mask_path_by_id(mid)
            mask = self._build_mask_from_image(img_path, conf)
            masks.append(mask)
            self.gt_keypoints_list.append(gt_keypoints)
        print('Mask loaded')
        if self.batch_size == 1:
            return masks[0]
        else:
            return jnp.stack(masks, axis=0)
    
    def _resolve_mask_path_by_id(self, mask_id):
        base_dir = os.path.join('cloth', f'polyfold-{self.cloth_type}-tasks', 'train', 'mask')
        cand = [f"{mask_id}.png"]

        gt_keypoints_path = os.path.join(base_dir.replace('mask', 'gt_keypoints'), f"{mask_id}.pkl")
        if not os.path.exists(gt_keypoints_path):
            print(f"GT keypoints {gt_keypoints_path} not found")
            gt_keypoints = None
        else:
            with open(gt_keypoints_path, 'rb') as f:
                gt_keypoints = pickle.load(f)

        for name in cand:
            p = os.path.join(base_dir, name)
            if os.path.exists(p):
                return p, gt_keypoints
            else:
                print(f"Mask {name} not found in {base_dir}")

        return os.path.join(my_path, "others", random.choice(["t-shirt.jpg", "t-shirt1.jpg", "t-shirt2.jpg", "t-shirt3.jpg"]))

    def _build_mask_from_image(self, img_path, conf):
        img = cv2.imread(img_path)
        if img is None:
            img = cv2.imread(f"{my_path}/others/t-shirt.jpg")
        size = conf.N // 2
        h_size = size // 2
        # img = cv2.resize(img, (size, size))
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # mask = (img.sum(-1) < 100).astype(np.int32)
        # cloth_mask = jnp.zeros((conf.N, conf.N))
        # cloth_mask = cloth_mask.at[conf.N // 2 - h_size:conf.N // 2 + h_size,
        #                            conf.N // 2 - h_size:conf.N // 2 + h_size].set(mask)
        img = cv2.resize(img, (conf.N, conf.N))
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        mask = (img.sum(-1) < 100).astype(np.int32)
        cloth_mask = jnp.zeros((conf.N, conf.N))
        cloth_mask = cloth_mask.at[0:conf.N, 0:conf.N].set(mask)

        return cloth_mask
    
    def reset_gripper_positions(self, state):
        """重置gripper到左上角和右上角位置"""
        # 左上角gripper (red) - 坐标系中左边和前面
        left_top_pos = jnp.array([0.1, 0.15, 0.1, 0.01])
        # 右上角gripper (blue) - 坐标系中右边和前面  
        right_top_pos = jnp.array([0.9, 0.15, 0.1, 0.01])
        
        # 处理批量数据
        if state.primitive0.ndim > 1:  # 如果有batch维度
            left_pos_batch = jnp.tile(left_top_pos, (self.batch_size, 1))
            right_pos_batch = jnp.tile(right_top_pos, (self.batch_size, 1))
        else:
            left_pos_batch = left_top_pos
            right_pos_batch = right_top_pos
            
        state = state._replace(
            primitive0=left_pos_batch,
            primitive1=right_pos_batch
        )
        
        return state
    
    def step_fold(self, actions, visualize=False, visualize_path=None):
        self.step_count += 1
        state_before = self.info['state']
        # rgbs_before, depths_before, mask_images = self.render_all(state_before, need_mask=True, visualize=False)
        # actions_test = actions.copy()
        # for i in range(actions_test.shape[0]):
        #     actions_test[i, 0] = self.gt_keypoints_list[i][1][0] /400
        #     actions_test[i, 2] = self.gt_keypoints_list[i][1][1] /400
        #     actions_test[i, 3] = (self.gt_keypoints_list[i][1][0] + self.gt_keypoints_list[i][3][0]) / 2 / 400
        #     actions_test[i, 5] = (self.gt_keypoints_list[i][1][1] + self.gt_keypoints_list[i][3][1]) / 2 / 400
        #     actions_test[i, 6] = self.gt_keypoints_list[i][1][0] / 400
        #     actions_test[i, 8] = self.gt_keypoints_list[i][1][1] / 400
        #     actions_test[i, 9] = (self.gt_keypoints_list[i][1][0] + self.gt_keypoints_list[i][3][0]) / 2 / 400
        #     actions_test[i, 11] = (self.gt_keypoints_list[i][1][1] + self.gt_keypoints_list[i][3][1]) / 2 / 400
        # actions = actions_test
 
        corrected_actions = ClothEnv.check_and_correct_dual_arm_pick_points(
            actions, state_before, None, grid_size=self.conf.N
        )

        actions = corrected_actions

        obs, base_reward, done, info = self.step_diff(actions, state_before)
        
        # 每步操作后重置gripper位置
        info['state'] = self.reset_gripper_positions(info['state'])
        
        # 计算自定义奖励
        reward = self._calculate_custom_reward(info['state'], base_reward)
        
        # 更新子目标信息到info中
        if self.reward_type in ["subgoal", "combined", "subgoal_delta", "binary_subgoal"]:
            subgoal_info = self._get_subgoal_info()
            info.update(subgoal_info)
        
        self.info = info

        if self.record_video:
            self.episode_data['states'].append(self.info['state_list'])


        if visualize and visualize_path is not None:
            start = time.time()
            for i in range(0, self.batch_size):
                if not (i%5 == 0 and (self.subgoals_reached is not None and self.subgoals_reached[i])):
                    continue
                rgb_before, _, = self.render(state_before, idx=i, visualize=False)
                save_path = os.path.join(visualize_path, f"{i}_{self.chosen_ids[i]}",f'{self.step_count}')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                rgb_pnp = rgb_before.copy()
                rgb_pc = rgb_before.copy()
                valid_mask = (
                    (state_before.x[i][:, 2] != 0) &      # z坐标不为0
                    (state_before.x[i][:, 0] != 0) &      # x坐标不为0  
                    (state_before.x[i][:, 0] > 0.1) &     # x坐标大于0.1
                    (state_before.x[i][:, 0] < 0.9) &      # x坐标小于0.9
                    (state_before.x[i][:, 2] > 0.1) &      # y坐标大于0.1
                    (state_before.x[i][:, 2] < 0.9)       # y坐标小于0.9
                )
                valid_particles = state_before.x[i][valid_mask]
                x_sampled = spatial_sampling(valid_particles, self.num_sampled_particles, method=self.sampling_method)
  
                for j in range(x_sampled.shape[0]):
                    cv2.circle(rgb_pc, (int(x_sampled[j, 0]*rgb_pnp.shape[1]), int(x_sampled[j, 2]*rgb_pnp.shape[0])), 5, (255, 0, 255), -1)
                    cv2.circle(rgb_pc, (int(x_sampled[j, 0]*rgb_pc.shape[1]), int(x_sampled[j, 2]*rgb_pc.shape[0])), 5, (255, 0, 255), -1)
                cv2.imwrite(os.path.join(save_path, f'rgb_before.png'), rgb_pc)
                cv2.circle(rgb_pnp, (int(actions[i, 0]*rgb_pnp.shape[1]), int(actions[i, 2]*rgb_pnp.shape[0])), 5, (0, 0, 255), -1)
                cv2.circle(rgb_pnp, (int(actions[i, 3]*rgb_pnp.shape[1]), int(actions[i, 5]*rgb_pnp.shape[0])), 5, (255, 0, 0), -1)
                if actions.shape[1] == 12:
                    cv2.circle(rgb_pnp, (int(actions[i, 6]*rgb_pnp.shape[1]), int(actions[i, 8]*rgb_pnp.shape[0])), 5, (0, 0, 255), -1)
                    cv2.circle(rgb_pnp, (int(actions[i, 9]*rgb_pnp.shape[1]), int(actions[i, 11]*rgb_pnp.shape[0])), 5, (255, 0, 0), -1)


                cv2.imwrite(os.path.join(save_path, f'rgb_pnp.png'), rgb_pnp)
                with open(os.path.join(save_path, f'info.txt'), 'w') as f:
                    f.write('pick1: (' + f'{actions[i, 0]*rgb_pnp.shape[1]}' + ', ' + f'{actions[i, 2]*rgb_pnp.shape[0]}' + ')' + '\n' + 'place1: (' + f'{actions[i, 3]*rgb_pnp.shape[1]}' + ', ' + f'{actions[i, 5]*rgb_pnp.shape[0]}' + ')' + '\n' + 'pick2: (' + f'{actions[i, 6]*rgb_pnp.shape[1]}' + ', ' + f'{actions[i, 8]*rgb_pnp.shape[0]}' + ')' + '\n' + 'place2: (' + f'{actions[i, 9]*rgb_pnp.shape[1]}' + ', ' + f'{actions[i, 11]*rgb_pnp.shape[0]}' + ')' + '\n' + 'reward_type: ' + f'{self.reward_type}' + '\n' + 'reward: ' + f'{reward[i]}' + '\n' + 'subgoal_idx: ' + f'{self.current_subgoal_idx[i]}' + '\n' + 'subgoal_step: ' + f'{self.current_subgoal_step[i]}' + '\n' + 'total_subgoals: ' + f'{len(self.subgoals[i])}' + '\n' + 'max_subgoal_steps: ' + f'{self.max_subgoal_steps}' + '\n' + 'distance: ' + f'{self.distances[i]}' + '\n' + 'subgoal_reached: ' + f'{self.subgoals_reached[i]}')

                rgb_pc_goal = rgb_before.copy()
               
                if self.current_subgoal_idx[i] < len(self.subgoals[i]):
                    cur_subgoal = self.subgoals[i][self.current_subgoal_idx[i]]
                    cur_subgoal_sampled = spatial_sampling(cur_subgoal, self.num_sampled_particles, method=self.sampling_method)
                    for j in range(cur_subgoal_sampled.shape[0]):
                        cv2.circle(rgb_pc_goal, (int(cur_subgoal_sampled[j, 0]*rgb_pc_goal.shape[1]), int(cur_subgoal_sampled[j, 2]*rgb_pc_goal.shape[0])), 5, (255, 0, 255), -1)

                    cv2.imwrite(os.path.join(save_path, f'rgb_pc_goal.png'), rgb_pc_goal)

                rgb_after, _= self.render(self.info['state'], idx=i, visualize=False)
                rgb_pc_goal_obs = rgb_after.copy()

                if self.current_subgoal_idx[i] < len(self.subgoals[i]):
                    cur_subgoal = self.subgoals[i][self.current_subgoal_idx[i]]
                    cur_subgoal_sampled = spatial_sampling(cur_subgoal, self.num_sampled_particles, method=self.sampling_method)
                    for j in range(cur_subgoal_sampled.shape[0]):
                        cv2.circle(rgb_pc_goal_obs, (int(cur_subgoal_sampled[j, 0]*rgb_pc_goal_obs.shape[1]), int(cur_subgoal_sampled[j, 2]*rgb_pc_goal_obs.shape[0])), 5, (255, 0, 255), -1)
                    cur_obs_sampled = spatial_sampling(self.info['state'].x[i], self.num_sampled_particles, method=self.sampling_method)
                    for j in range(cur_obs_sampled.shape[0]):
                        cv2.circle(rgb_pc_goal_obs, (int(cur_obs_sampled[j, 0]*rgb_pc_goal_obs.shape[1]), int(cur_obs_sampled[j, 2]*rgb_pc_goal_obs.shape[0])), 5, (120, 0, 120), -1)
                    cv2.imwrite(os.path.join(save_path, f'rgb_pc_goal_obs.png'), rgb_pc_goal_obs)

                cv2.imwrite(os.path.join(save_path, f'rgb_after.png'), rgb_after)
            end = time.time()
            # print(f"Time taken for one round of visualization: {end - start} seconds")


        # settle_steps = 30  # 增加30步让布料下沉
        # no_action = jnp.zeros((self.batch_size, 12))  # 无动作
        # for _ in range(settle_steps):
        #     info['state'], _ = self.simulator.step_jax(info['state'], no_action)
        #     print('settling...')
        return obs, reward, done, info
    
    def _calculate_custom_reward(self, current_state, base_reward):
        """根据reward_type计算自定义奖励"""
        if self.reward_type == "final_goal":
            # 使用原始的final goal reward
            return base_reward
            
        elif self.reward_type == "subgoal":
            # 只使用子目标奖励
            if len(self.subgoals) == 0:
                return base_reward
            
            subgoal_rewards, subgoals_reached = self._calculate_subgoal_reward(current_state)
            episode_success, subgoal_advanced, subgoal_timeout = self._update_subgoal_states(subgoals_reached)

            self.subgoals_reached = subgoals_reached
            self.episode_success = episode_success
            # 计算最终奖励
            rewards = subgoal_rewards.copy()
            
            # 达成子目标的额外奖励
            rewards += subgoals_reached * 10.0
            
            # 完成所有子目标的大奖励
            # rewards += episode_success * 100.0
            
            # 超时惩罚
            # rewards -= subgoal_timeout * 1.0
            
            return rewards
            
        elif self.reward_type == "combined":
            # 结合final goal和subgoal奖励
            final_reward = base_reward
            
            if len(self.subgoals) > 0:
                subgoal_rewards, subgoals_reached = self._calculate_subgoal_reward(current_state)
                episode_success, subgoal_advanced, subgoal_timeout = self._update_subgoal_states(subgoals_reached)
                self.subgoals_reached = subgoals_reached
                self.episode_success = episode_success
                # 组合奖励：0.3 * final_goal + 0.7 * subgoal
                combined_rewards = 0.3 * final_reward + 0.7 * subgoal_rewards
                
                # 达成子目标的额外奖励
                combined_rewards += subgoals_reached * 5.0
                
                # 完成所有子目标的大奖励
                combined_rewards += episode_success * 50.0
                
                # 超时惩罚
                combined_rewards -= subgoal_timeout * 2.0
                
                return combined_rewards
            else:
                return final_reward
                
        elif self.reward_type == "final_goal_delta":
            # 基于final goal距离变化的奖励
            return self._calculate_final_goal_delta_reward(current_state)
            
        elif self.reward_type == "subgoal_delta":
            # 基于subgoal距离变化的奖励
            if len(self.subgoals) == 0:
                return base_reward
            
            delta_rewards, subgoals_reached = self._calculate_subgoal_delta_reward(current_state)
            episode_success, subgoal_advanced, subgoal_timeout = self._update_subgoal_states(subgoals_reached)
            self.subgoals_reached = subgoals_reached
            self.episode_success = episode_success
            # 计算最终奖励
            rewards = delta_rewards.copy()
            
            # 达成子目标的额外奖励
            rewards += subgoals_reached * 10.0
            
            # 完成所有子目标的大奖励
            rewards += episode_success * 100.0
            
            # 超时惩罚
            rewards -= subgoal_timeout * 5.0
            
            return rewards
            
        elif self.reward_type == "binary_subgoal":            # 基于子目标的0-1奖励：失败时为0，每达成一个子目标+1，最终成功再+1
            if len(self.subgoals) == 0:
                # 如果没有子目标，使用简单的0-1奖励
                return np.ones(self.batch_size) if np.any(base_reward > 0) else np.zeros(self.batch_size)
            
            subgoal_rewards, subgoals_reached = self._calculate_subgoal_reward(current_state)
            episode_success, subgoal_advanced, subgoal_timeout = self._update_subgoal_states(subgoals_reached)
            print(111)
            self.subgoals_reached = subgoals_reached
            self.episode_success = episode_success
            
            # 初始化为0的奖励
            rewards = np.zeros(self.batch_size)
            
            # 每达成一个子目标+1
            rewards += subgoals_reached.astype(float) * 1.0
            
            # 完成所有子目标（最终成功）再+1
            # rewards += episode_success.astype(float) * 1.0
            
            return rewards
        
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}. "
                           f"Supported types: 'final_goal', 'subgoal', 'combined', 'final_goal_delta', 'subgoal_delta', 'binary_subgoal'")
    
    def _get_subgoal_info(self):
        """获取子目标相关信息，每个环境可能有不同的子目标数量"""
        # 计算每个环境的子目标总数
        total_subgoals_per_env = []
        for i in range(self.batch_size):
            if i < len(self.subgoals):
                total_subgoals_per_env.append(len(self.subgoals[i]))
            else:
                total_subgoals_per_env.append(0)
        
        return {
            'subgoal_idx': self.current_subgoal_idx.copy(),
            'subgoal_step': self.current_subgoal_step.copy(),
            'total_subgoals': total_subgoals_per_env,
            'subgoal_reached': self.subgoals_reached.copy() if self.subgoals_reached is not None else None,
            'episode_success': self.episode_success.copy() if self.episode_success is not None else None,
            'max_subgoal_steps': self.max_subgoal_steps,
            'chosen_ids': self.chosen_ids.copy()
        }
    
    def reset_subgoal_states(self):
        """重置子目标状态"""
        self.current_subgoal_idx = np.zeros(self.batch_size, dtype=int)
        self.current_subgoal_step = np.zeros(self.batch_size, dtype=int)
        
        # 重置子目标状态变量
        self.subgoals_reached = None
        self.episode_success = None
        
        # 如果使用基于距离变化的奖励，也重置距离跟踪变量
        if self.reward_type in ["final_goal_delta", "subgoal_delta"]:
            self.prev_final_goal_distance = None
            self.prev_subgoal_distance = None
    
    def on_episode_end(self):
        pass

    def get_state(self):
        return self.info['state']

    def get_episode_data(self):
        return self.episode_data


    @staticmethod
    @vmap
    @jax.jit
    def get_obs(state: ClothState, obs_type=ClothEnv.PARTICLE):

        if obs_type == ClothEnv.DEPTH:
            pixel_size = 0.003125
            bounds = jnp.array([[0, 1], [0, 1], [0, 1]])
            points = state.x + jnp.array([[0, 0.01, 0]])
            width = 320
            height = 640
            iz = jnp.argsort(points[:, 1])
            points = points[iz]
            px = jnp.floor((points[:, 0] - bounds[0, 0]) / pixel_size).astype(int)
            py = jnp.floor((points[:, 2] - bounds[1, 0]) / pixel_size).astype(int)
            px = jnp.clip(px, 0, width - 1)
            py = jnp.clip(py, 0, height - 1)

            heightmap = jnp.zeros((320, 640), dtype=jnp.float32)
            heightmap.at[py, px].set(points[:, 1] - bounds[2, 0])
            heightmap = jnp.expand_dims(heightmap, axis=-1)
            obs = heightmap

        elif obs_type == ClothEnv.PARTICLE:

            # sample x (N,3) every 10 points
            x = state.x[::10, :]
            obs = jnp.concatenate(
                [
                    x.flatten(),
                    # v.flatten(),
                    state.primitive0,
                    state.primitive1,
                ],
                axis=-1,
            )
        else:
            raise NotImplementedError

        return obs


    def render_video(self, episode_data, output_dir="./videos", 
                                       batch_idx=None, format="gif", fps=20, filename_prefix="episode"):
        """
        Generate video or GIF from episode_data
        Args:
            episode_data: Dictionary containing 'states' key with list of states
            output_dir: Output directory path
            batch_idx: Specific batch index, if None generates videos for all batches
            format: Output format, "gif" or "mp4"
            fps: Frame rate
            filename_prefix: Filename prefix
            
        Returns:
            List of generated file paths
        """
        from pathlib import Path
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        states_list = episode_data['states']
        if not states_list:
            print("Warning: No state data in episode_data")
            return []
        
        # Get batch_size
        first_state = states_list[0]
        if hasattr(first_state, 'x'):
            batch_size = first_state.x.shape[1]
        else:
            batch_size = len(first_state) if isinstance(first_state, (list, tuple)) else 1
        
        generated_files = []
        
        if batch_idx is not None:
            if batch_idx >= batch_size:
                raise ValueError(f"batch_idx {batch_idx} is out of range [0, {batch_size-1}]")
            batch_indices = [batch_idx]
        else:
            batch_indices = list(range(batch_size))
        
        print(f"Starting video generation, total {len(states_list)} frames, processing batches: {batch_indices}")
        
        for b_idx in tqdm(batch_indices, desc=f"Rendering batches (total: {len(batch_indices)})", 
                         unit="batch", 
                         bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
            
            rgb_frames = []
            
            for step_idx, state in enumerate(states_list):
                for time_idx in range(state.x.shape[0]):
                    current_state = state._replace(
                        x=state.x[time_idx],  # shape: (batch_size, particles, 3)
                        primitive0=state.primitive0[time_idx] if hasattr(state, 'primitive0') and len(state.primitive0.shape) > 2 else state.primitive0,
                        primitive1=state.primitive1[time_idx] if hasattr(state, 'primitive1') and len(state.primitive1.shape) > 2 else state.primitive1,
                    )
                    # 同步切片包含时间维度的几何/掩码字段，避免 vmap 轴不一致
                    # 依据规则：若该字段第0维长度与 state.x 的第0维（时间步数）相同，则按 time_idx 切片
                    ts_len = state.x.shape[0]
                    # idx_i_b
                    if hasattr(state, 'idx_i_b') and state.idx_i_b is not None:
                        if hasattr(state.idx_i_b, 'shape') and len(state.idx_i_b.shape) >= 2 and state.idx_i_b.shape[0] == ts_len:
                            current_state = current_state._replace(idx_i_b=state.idx_i_b[time_idx])
                        else:
                            current_state = current_state._replace(idx_i_b=state.idx_i_b)
                    # idx_j_b
                    if hasattr(state, 'idx_j_b') and state.idx_j_b is not None:
                        if hasattr(state.idx_j_b, 'shape') and len(state.idx_j_b.shape) >= 2 and state.idx_j_b.shape[0] == ts_len:
                            current_state = current_state._replace(idx_j_b=state.idx_j_b[time_idx])
                        else:
                            current_state = current_state._replace(idx_j_b=state.idx_j_b)
                    # j_x_b
                    if hasattr(state, 'j_x_b') and state.j_x_b is not None:
                        if hasattr(state.j_x_b, 'shape') and len(state.j_x_b.shape) >= 2 and state.j_x_b.shape[0] == ts_len:
                            current_state = current_state._replace(j_x_b=state.j_x_b[time_idx])
                        else:
                            current_state = current_state._replace(j_x_b=state.j_x_b)
                    # j_y_b
                    if hasattr(state, 'j_y_b') and state.j_y_b is not None:
                        if hasattr(state.j_y_b, 'shape') and len(state.j_y_b.shape) >= 2 and state.j_y_b.shape[0] == ts_len:
                            current_state = current_state._replace(j_y_b=state.j_y_b[time_idx])
                        else:
                            current_state = current_state._replace(j_y_b=state.j_y_b)
                    # i_x_b
                    if hasattr(state, 'i_x_b') and state.i_x_b is not None:
                        if hasattr(state.i_x_b, 'shape') and len(state.i_x_b.shape) >= 2 and state.i_x_b.shape[0] == ts_len:
                            current_state = current_state._replace(i_x_b=state.i_x_b[time_idx])
                        else:
                            current_state = current_state._replace(i_x_b=state.i_x_b)
                    # i_y_b
                    if hasattr(state, 'i_y_b') and state.i_y_b is not None:
                        if hasattr(state.i_y_b, 'shape') and len(state.i_y_b.shape) >= 2 and state.i_y_b.shape[0] == ts_len:
                            current_state = current_state._replace(i_y_b=state.i_y_b[time_idx])
                        else:
                            current_state = current_state._replace(i_y_b=state.i_y_b)
                    # original_length_b
                    if hasattr(state, 'original_length_b') and state.original_length_b is not None:
                        if hasattr(state.original_length_b, 'shape') and len(state.original_length_b.shape) >= 3 and state.original_length_b.shape[0] == ts_len:
                            current_state = current_state._replace(original_length_b=state.original_length_b[time_idx])
                        else:
                            current_state = current_state._replace(original_length_b=state.original_length_b)
                    # ori_len_is_not_0_b
                    if hasattr(state, 'ori_len_is_not_0_b') and state.ori_len_is_not_0_b is not None:
                        if hasattr(state.ori_len_is_not_0_b, 'shape') and len(state.ori_len_is_not_0_b.shape) >= 3 and state.ori_len_is_not_0_b.shape[0] == ts_len:
                            current_state = current_state._replace(ori_len_is_not_0_b=state.ori_len_is_not_0_b[time_idx])
                        else:
                            current_state = current_state._replace(ori_len_is_not_0_b=state.ori_len_is_not_0_b)
                    # cloth_mask_b
                    if hasattr(state, 'cloth_mask_b') and state.cloth_mask_b is not None:
                        if hasattr(state.cloth_mask_b, 'shape') and len(state.cloth_mask_b.shape) >= 3 and state.cloth_mask_b.shape[0] == ts_len:
                            current_state = current_state._replace(cloth_mask_b=state.cloth_mask_b[time_idx])
                        else:
                            current_state = current_state._replace(cloth_mask_b=state.cloth_mask_b)

                    rgb_frame, _ = self.render(current_state, visualize=False, idx=b_idx)

                
                    if rgb_frame.dtype != np.uint8:
                        rgb_frame = (rgb_frame * 255).astype(np.uint8)
                    
                    rgb_frames.append(rgb_frame)

            
            if not rgb_frames:
                print(f"Warning: No frames generated for batch {b_idx}")
                continue
            
            # 输出路径：output_dir/<id>/<id>.<format>
            id_str = str(self.chosen_ids[b_idx]) if hasattr(self, 'chosen_ids') and b_idx < len(self.chosen_ids) else str(b_idx)
            id_dir = os.path.join(output_dir, id_str)
            Path(id_dir).mkdir(parents=True, exist_ok=True)

            filename = f"{id_str}.{format}"
            filepath = os.path.join(id_dir, filename)
            
            try:
                if format.lower() == "gif":
                    imageio.mimsave(filepath, rgb_frames, fps=fps)
                elif format.lower() == "mp4":
                    with imageio.get_writer(filepath, fps=fps, codec='libx264') as writer:
                        for frame in rgb_frames:
                            writer.append_data(frame)
                else:
                    raise ValueError(f"Unsupported format: {format}, please use 'gif' or 'mp4'")
                
                generated_files.append(filepath)
            except Exception as e:
                print(f"Error saving file: {e}")
                continue
        
        print(f"Video generation completed! Generated {len(generated_files)} files")
        return generated_files


if __name__ == "__main__":
    conf = DefaultConf()
    conf.batch_size = 10
    conf.id_range = list(range(0, 10))

    # conf.id_range = list(range(0, 50))
    # conf.id_range = list(range(50, 100))
    conf.cloth_type = 'square'
    conf.record_video = True
    env = FoldEnv(conf=conf, seed=1)    
    # env.collect_goal()
    # env.collect_expert_demo(10)
    print("time start")
    start_time = time.time()
    iter_num = 1
    
    for i in range(iter_num):
        state = env.get_state()
        
        # actions = get_expert_start_end_cloth(env.get_x_grid(state), env.cloth_mask)

        # actions = env.get_random_fold_action(state)
        # actions = env.get_random_dual_arm_fold_action(state)
        actions = np.zeros((env.batch_size, 6))

        obs, reward, done, info = env.step_fold(actions)
        state = env.get_state()
        print(state.x.shape)
        print(state.x[0].max())
        print(state.x[0].min())
        with open('particle_pos.pkl', 'wb') as f:
            pickle.dump(np.array(state.x[0]), f)
    
    episode_data = env.get_episode_data()
        
    print('All time: ', time.time() - start_time)
    print('Average time: ', (time.time() - start_time) / iter_num)
    
    vstart = time.time()
    env.render_video(episode_data, output_dir="./videos", batch_idx=None, format="gif", fps=20, filename_prefix="episode")
    print('Video time: ', time.time() - vstart)
    print('Average video time: ', (time.time() - vstart) / env.batch_size)