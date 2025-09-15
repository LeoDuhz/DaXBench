import glob
import math
import os
import pickle
import cv2
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from gym.spaces import Box
from jax import random, vmap

from daxbench.core.engine.cloth_simulator import ClothSimulator, ClothState
from daxbench.core.engine.pyrender.py_render import MeshPyRenderer
from daxbench.core.utils.util import calc_chamfer, get_expert_start_end_cloth, get_projection

from icecream import ic as print

my_path = os.path.dirname(os.path.abspath(__file__))


class ClothEnv:
    PARTICLE = "PARTICLE"
    DEPTH = "DEPTH"
    RGB = "RGB"

    def __init__(self, conf, batch_size, max_steps, aux_reward=False):

        assert conf
        cloth_mask = self.create_cloth_mask(conf)
        collision_func = self.get_collision_func()
        simulator = ClothSimulator(conf, batch_size, collision_func, cloth_mask)

        self.conf = conf
        self.aux_reward = aux_reward
        self.simulator = simulator
        self.cloth_mask = simulator.cloth_mask
        self.max_steps = max_steps
        self.batch_size = simulator.batch_size
        self.cur_step = 0
        self.action_size = 6
        self.seed(conf.seed)

        assert conf.goal_path
        self.goal_path = conf.goal_path

        # 兼容多mask：用批内最大有效点数估计观测维度
        if hasattr(cloth_mask, "ndim") and cloth_mask.ndim == 3:
            num_p = int(jnp.max(jnp.sum(cloth_mask.astype(jnp.int32), axis=(1, 2))))
            idx_base_i, idx_base_j = jnp.nonzero(cloth_mask[0])
        else:
            num_p = int(self.cloth_mask.astype(jnp.int32).sum())
            idx_base_i, idx_base_j = jnp.nonzero(self.cloth_mask)

        self.observation_size = num_p * 6 + 8
        self.cloth_state_shape = (num_p, 6)
        self.observation_space = Box(
            low=-1.0, high=1.0, shape=(num_p * 6 + 8,), dtype=np.float32
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.spec = None

        self.idx_i, self.idx_j = idx_base_i, idx_base_j
        self.renderer = MeshPyRenderer(top_down_view=True)
        self.step_diff = self.build_step_diff()
        self.step_diff = jax.jit(self.step_diff)
        self.reset = self.build_reset()

        if not os.path.exists(conf.goal_path):
            print("**************** Warning: goal file does not exist!")
            self.goal = jnp.zeros((1, 3))
        else:
            goal_map = np.load(conf.goal_path)
            self.goal = jnp.array(goal_map)

    def seed(self, seed):
        self.simulator.key_global = random.PRNGKey(seed)
        np.random.seed(seed)

    def state_to_depth(self, state):
        pixel_size = 0.003125
        z_offset = 0.01
        width = 320
        height = 320
        bounds = jnp.array([[0, 1], [0, 1], [0, 1]])
        points = state.x + jnp.array([[0, z_offset, 0]])
        points = points[0]

        iz = jnp.argsort(points[..., 1])
        points = points[iz]
        px = jnp.floor((points[:, 0] - bounds[0, 0]) / pixel_size).astype(int)
        py = jnp.floor((points[:, 2] - bounds[1, 0]) / pixel_size).astype(int)
        px = jnp.clip(px, 0, width - 1)
        py = jnp.clip(py, 0, height - 1)

        heightmap = jnp.zeros((width, height), dtype=jnp.float32)
        heightmap = heightmap.at[py, px].set(points[:, 1])
        heightmap = jnp.expand_dims(heightmap, axis=-1)
        heightmap = np.array(heightmap)

        return np.array(heightmap)

    @staticmethod
    @vmap
    @jax.jit
    def get_obs(state: ClothState, obs_type=PARTICLE):

        if obs_type == ClothEnv.DEPTH:
            pixel_size = 0.003125
            z_offset = 0.01
            bounds = jnp.array([[0, 1], [0, 1], [0, 1]])
            points = state.x + jnp.array([[0, z_offset, 0]])
            points = points[0]
            width = 320
            height = 320
            iz = jnp.argsort(points[..., 1])
            points = points[iz]
            px = jnp.floor((points[:, 0] - bounds[0, 0]) / pixel_size).astype(int)
            py = jnp.floor((points[:, 2] - bounds[1, 0]) / pixel_size).astype(int)
            px = jnp.clip(px, 0, width - 1)
            py = jnp.clip(py, 0, height - 1)

            heightmap = jnp.zeros((width, height), dtype=jnp.float32)
            heightmap = heightmap.at[py, px].set(points[:, 1])
            heightmap = jnp.expand_dims(heightmap, axis=-1)
            obs = heightmap

        elif obs_type == ClothEnv.PARTICLE:
            obs = jnp.concatenate(
                [
                    state.x.flatten(),
                    # state.v.flatten(),
                    state.primitive0,
                    state.primitive1,
                ],
                axis=-1,
            )
        else:
            raise NotImplementedError

        return obs

    @staticmethod
    @partial(vmap, in_axes=(0, 0), out_axes=1)
    def get_pnp_actions(actions, state: ClothState):
        pick, place = actions[:3], actions[3:]

        pick = pick.at[1].set(0)
        place = place.at[1].set(0)

        # 修复：下移阶段应该强吸(suction=0)，而不是不吸(suction=1)
        act_down = pick - state.primitive0[:3]
        act_down = jnp.zeros(4).at[:3].set(act_down)  # suction = 0 (强吸)
        act_down = act_down[None, ...].repeat(5, axis=0)
        act_down = act_down.at[..., :3].set(act_down[..., :3] / 5)
        sub_actions = act_down

        # 修复：举起阶段应该强吸(suction=0)
        lift_height = 0.15  # 稍微降低举起高度，避免过度
        act_up = jnp.array([0, lift_height, 0, 0])  # suction = 0 (强吸)
        act_up = act_up[None, ...].repeat(12, axis=0)  # 减少步数
        act_up = act_up.at[..., :3].set(act_up[..., :3] / 12)
        sub_actions = jnp.concatenate([sub_actions, act_up], axis=0)

        # 修复：移动阶段应该强吸(suction=0)
        act_move = place - pick
        act_move = act_move.at[1].set(0)
        act_move = jnp.zeros(4).at[:3].set(act_move)  # suction = 0 (强吸)
        act_move = act_move[None, ...].repeat(25, axis=0)  # 减少步数
        act_move = act_move.at[..., :3].set(act_move[..., :3] / 25)
        sub_actions = jnp.concatenate([sub_actions, act_move], axis=0)

        # 修复：增加释放时长并添加辅助脱离动作
        release_height = 0.05  # 释放时稍微向上移动，帮助布料脱离
        act_release = jnp.array([0, release_height, 0, 1])  # suction = 1 (释放)
        act_release = act_release[None, ...].repeat(15, axis=0)  # 增加释放时长
        act_release = act_release.at[..., :3].set(act_release[..., :3] / 15)  # 渐进释放
        sub_actions = jnp.concatenate([sub_actions, act_release], axis=0)


        dummy_actions = jnp.zeros_like(sub_actions)
        sub_actions = jnp.concatenate([sub_actions, dummy_actions], axis=1)

        return sub_actions

    @staticmethod
    @partial(vmap, in_axes=(0, 0), out_axes=1)
    def get_dual_arm_pnp_actions(actions, state: ClothState):
        arm1_pick, arm1_place = actions[:3], actions[3:6]
        arm2_pick, arm2_place = actions[6:9], actions[9:12]
        arm1_pick = arm1_pick.at[1].set(0)
        arm1_place = arm1_place.at[1].set(0)
        arm2_pick = arm2_pick.at[1].set(0)
        arm2_place = arm2_place.at[1].set(0)
        # ARM 1 actions - 修复吸盘强度
        act1_down = arm1_pick - state.primitive0[:3]
        act1_down = jnp.zeros(4).at[:3].set(act1_down)  # suction = 0 (强吸)
        act1_down = act1_down[None, ...].repeat(5, axis=0) / 5


        lift_height = 0.1
        act1_up = jnp.array([0, lift_height, 0, 0])[None, ...].repeat(12, axis=0) / 12  # suction = 0
        
        act1_move = jnp.zeros(4).at[:3].set((arm1_place - arm1_pick).at[1].set(0))  # suction = 0
        act1_move = act1_move[None, ...].repeat(25, axis=0) / 25

        # 修复：增加释放时长并添加辅助脱离动作
        release_height = 0.05  # 释放时稍微向上移动，帮助布料脱离
        act1_release = jnp.array([0, release_height, 0, 1])[None, ...].repeat(15, axis=0) / 15  # suction = 1 (释放)
        arm1_actions = jnp.concatenate([act1_down, act1_up, act1_move, act1_release], axis=0)

        # ARM 2 actions - 应用相同修复
        act2_down = arm2_pick - state.primitive1[:3]
        act2_down = jnp.zeros(4).at[:3].set(act2_down)  # suction = 0
        act2_down = act2_down[None, ...].repeat(5, axis=0) / 5

        act2_up = jnp.array([0, lift_height, 0, 0])[None, ...].repeat(12, axis=0) / 12
        
        act2_move = jnp.zeros(4).at[:3].set((arm2_place - arm2_pick).at[1].set(0))
        act2_move = act2_move[None, ...].repeat(25, axis=0) / 25

        # 修复：增加释放时长并添加辅助脱离动作
        act2_release = jnp.array([0, release_height, 0, 1])[None, ...].repeat(15, axis=0) / 15  # suction = 1 (释放)
        arm2_actions = jnp.concatenate([act2_down, act2_up, act2_move, act2_release], axis=0)

        sub_actions = jnp.concatenate([arm1_actions, arm2_actions], axis=1)
        return sub_actions

    def get_x_grid(self, state):
        return self.simulator.get_x_grid(state.x, state.idx_i_b, state.idx_j_b)

    def build_reset(self):
        init_state = self.simulator.reset_jax()

        def reset(key):
            key, _ = random.split(key)
            new_x = init_state.x.at[..., [0, 2]].add(random.normal(key, (2,)) * 0.05)
            state = init_state._replace(x=new_x)
            return self.get_obs(state), state

        return reset

    def step_with_render(self, actions, state: ClothState, visualize=True):
        obs, reward, done, info = self.step_diff(actions, state)
        actions = self.get_pnp_actions(actions, state)
        img_list = []
        for action in actions:
            state, _ = self.simulator.step_jax(state, action)
            rgb, depth = self.render(state, visualize)
            img_list.append(rgb)

        info["img_list"] = img_list
        return obs, reward, done, info

    def build_step_diff(self):
        get_obs_list = jax.vmap(self.get_obs)

        def step_diff(actions, state: ClothState):
            old_chamfer_distance = calc_chamfer(state.x, self.goal)
            
            if actions.shape[-1] == 12:  # Dual arm actions
                arm1_pickup = actions[..., :3]
                arm2_pickup = actions[..., 6:9]
                arm1_place = actions[..., 3:6]
                arm2_place = actions[..., 9:12]
                
                # 计算pick点之间的距离
                pick_distance = jnp.sqrt(jnp.sum((arm1_pickup - arm2_pickup) ** 2, axis=-1))
                # 计算place点之间的距离
                place_distance = jnp.sqrt(jnp.sum((arm1_place - arm2_place) ** 2, axis=-1))
                
                # 距离阈值，可以根据需要调整
                distance_threshold = 0.05
                
                # 如果pick点和place点都很近，退化为单臂
                use_single_arm = (pick_distance < distance_threshold) & (place_distance < distance_threshold)
                
                # 根据条件选择使用单臂还是双臂
                def single_arm_path():
                    # 使用第一个机械臂的动作，第二个机械臂保持在静止位置
                    single_actions = jnp.concatenate([
                        arm1_pickup, arm1_place,  # arm1 pick and place
                        jnp.array([-0.8, 0.1, -0.8])[None, ...].repeat(actions.shape[0], axis=0),  # arm2 rest position
                        jnp.array([-0.8, 0.1, -0.8])[None, ...].repeat(actions.shape[0], axis=0)   # arm2 rest position
                    ], axis=-1)
                    return self.get_dual_arm_pnp_actions(single_actions, state)
                
                def dual_arm_path():
                    return self.get_dual_arm_pnp_actions(actions, state)
                
                actions = jax.lax.cond(
                    jnp.all(use_single_arm),  # 如果所有样本都使用单臂
                    single_arm_path,
                    dual_arm_path
                )
                
                particle_num = state.x.shape[-2]
                arm1_pickup_expanded = arm1_pickup[..., None, :].repeat(particle_num, -2)
                arm2_pickup_expanded = arm2_pickup[..., None, :].repeat(particle_num, -2)
                
                contact_distance1 = jnp.sqrt(jnp.sum((arm1_pickup_expanded - state.x) ** 2, -1)).min(-1)
                contact_distance2 = jnp.sqrt(jnp.sum((arm2_pickup_expanded - state.x) ** 2, -1)).min(-1)
                contact_distance = jnp.minimum(contact_distance1, contact_distance2)
                
            else:  # Single arm actions (6-dim) 
                static_arm_position = jnp.array([-0.8, 0.1, -0.8])  # rest position: top-right corner
                
                # [arm1_pick(3), arm1_place(3), arm2_pick(3), arm2_place(3)]
                expanded_actions = jnp.concatenate([
                    actions[..., :6],  
                    static_arm_position[None, ...].repeat(actions.shape[0], axis=0),  
                    static_arm_position[None, ...].repeat(actions.shape[0], axis=0)  
                ], axis=-1)
                
                arm1_pickup = expanded_actions[..., :3]
                particle_num = state.x.shape[-2]
                arm1_pickup_expanded = arm1_pickup[..., None, :].repeat(particle_num, -2)
                contact_distance = jnp.sqrt(jnp.sum((arm1_pickup_expanded - state.x) ** 2, -1)).min(-1)
                
                # 使用统一的双臂动作处理函数
                actions = self.get_pnp_actions(expanded_actions, state)
                
                
            
            state, state_list = jax.lax.scan(self.simulator.step_jax, state, actions, length=actions.shape[0])
            state = state._replace(cur_step=state.cur_step + 1)
            obs = self.get_obs(state)

            if self.conf.use_substep_obs:
                obs_list = get_obs_list(state_list)
            else:
                obs_list = obs
                
            reward, done, info = 0, state.cur_step >= self.max_steps, {
                "state": state, "obs_list": obs_list, "state_list": state_list
            }
            
            chamfer_distance = calc_chamfer(state.x, self.goal)
            reward = math.e ** (-chamfer_distance * 10)
            if self.aux_reward:
                reward += math.e ** (-contact_distance)
            
            real_reward = old_chamfer_distance - chamfer_distance + 0.1 * contact_distance
            info['real_reward'] = real_reward
            reward *= 0.99 ** state.cur_step
            return obs, reward, done, info

        return step_diff

    def render(self, state: ClothState, visualize=True, idx=0):
        assert idx < state.x.shape[0]
        indices = self.simulator.get_indices_for_batch(idx)
        return self.renderer.render(
            self.get_x_grid(state)[idx],
            indices,
            state.primitive0[idx],
            ps1=state.primitive1[idx],
            visualize=visualize
        )

    def render_all(self, state: ClothState, need_mask=False, visualize=False):
        """
        render all states in the batch
        
        Args:
            state: ClothState object, contains batch_size states
            visualize: whether to display the rendering window (False on headless servers)
            
        Returns:
            tuple: (rgb_images, depth_images)
                - rgb_images: numpy array of shape (batch_size, height, width, 3)
                - depth_images: numpy array of shape (batch_size, height, width)
        """
        
        x_grids = self.get_x_grid(state)  # shape: (batch_size, N, N, 3)
        batch_size = x_grids.shape[0]
        
        rgb_images = []
        depth_images = []
        
        for i in range(batch_size):
            indices = self.simulator.get_indices_for_batch(i)
            rgb, depth = self.renderer.render(
                x_grids[i],
                indices,
                state.primitive0[i],
                ps1=state.primitive1[i],
                visualize=visualize
            )
            rgb_images.append(rgb)
            depth_images.append(depth)
        
        rgb_images = np.array(rgb_images)  # shape: (batch_size, height, width, 3)
        depth_images = np.array(depth_images)  # shape: (batch_size, height, width)

        mask_images = None
        if need_mask:
            mask_images = self.create_mask_from_depth(depth_images)

        if need_mask:
            return rgb_images, depth_images, mask_images
        else:
            return rgb_images, depth_images

    def create_mask_from_depth(self, depths_before):
        batch_size = len(depths_before)
        masks = []
        
        for i in range(batch_size):
            depth_img = depths_before[i]
            
            if depth_img.max() > depth_img.min():
                depth_normalized = ((depth_img - depth_img.min()) / (depth_img.max() - depth_img.min()) * 255).astype(np.uint8)
            else:
                # 如果深度图没有变化，设为全零
                depth_normalized = np.zeros_like(depth_img, dtype=np.uint8)
            
            
            # 创建mask: 1-250范围内为前景(0/黑色)，其他为背景(255/白色)
            mask = np.ones_like(depth_normalized, dtype=np.uint8) * 255  # 初始化为白色背景
            
            # 前景条件：深度值在1-250之间
            foreground_condition = (depth_normalized > 30) & (depth_normalized < 250)
            mask[foreground_condition] = 0  # 前景设为黑色
            
            masks.append(mask)
        
        return np.array(masks)  # (batch_size, height, width)
    

    def create_cloth_mask(self, conf):
        raise NotImplementedError

    def get_collision_func(self):
        def collision_func(x, v, idx_i, idx_j):
            return v

        return collision_func

    def collect_goal(self):
        assert self.batch_size == 1
        while True:
            self.simulator.key_global, _ = random.split(self.simulator.key_global)
            obs, state = self.reset(self.simulator.key_global)
            valid_episode = True
            while True:
                self.render(state)
                actions = get_expert_start_end_cloth(self.get_x_grid(state), self.cloth_mask)

                # click on the same place to terminate
                if jnp.linalg.norm(actions[0, :3] - actions[0, 3:]) < 1e-3:
                    break

                # click on two far away points to terminate
                if jnp.linalg.norm(actions[0, :3] - actions[0, 3:]) > 0.8:
                    valid_episode = False
                    break

                # obs, reward, _, info = env.step_diff(actions, state)
                obs, reward, _, info = self.step_with_render(actions, state)
                state = info['state']
                print("reward", reward)

            if valid_episode:
                os.makedirs(f"{my_path}/goals/{self.conf.task}", exist_ok=True)
                np.save(f"{my_path}/goals/{self.conf.task}/goal.npy", state.x[0])
                print("Goal saved in", f"{my_path}/goals/{self.conf.task}/goal.npy")
                exit(0)

    def collect_expert_demo(self, num_demo=10):
        assert self.batch_size == 1

        # visualize goal
        goal_state = np.load(self.conf.goal_path)
        goal_state = goal_state[None, ...].repeat(self.batch_size, axis=0)
        goal_grid = self.simulator.get_x_grid(goal_state)
        goal_map = get_projection(goal_grid, self.cloth_mask, size=512)
        # cv2.imshow("goal_map", goal_map[0])
        # cv2.waitKey(10)

        # get number of existing demo files
        num_existing_demo = len(glob.glob(f"{my_path}/expert_demo/{self.conf.task}/*.pkl"))
        i = num_existing_demo
        while i < num_demo:
            self.simulator.key_global, _ = random.split(self.simulator.key_global)
            obs, state = self.reset(self.simulator.key_global)
            demo = {"obs": [], "action": [], "state": []}
            valid_episode = True
            while True:
                self.render(state)
                actions = get_expert_start_end_cloth(self.get_x_grid(state), self.cloth_mask, goal_map)

                # click on the same place to terminate
                if jnp.linalg.norm(actions[0, :3] - actions[0, 3:]) < 1e-3:
                    break

                # click on two far away points to terminate
                if jnp.linalg.norm(actions[0, :3] - actions[0, 3:]) > 0.8:
                    valid_episode = False
                    break

                demo['state'].append(state)
                demo['action'].append(actions)
                demo['obs'].append(obs)

                obs, reward, _, info = self.step_diff(actions, state)
                # obs, reward, _, info = self.step_with_render(actions, state)
                state = info['state']
                print(state.cur_step, "reward", reward)

            if valid_episode:
                os.makedirs(f"{my_path}/expert_demo/{self.conf.task}", exist_ok=True)
                with open(f"{my_path}/expert_demo/{self.conf.task}/demo_{i}.pkl", "wb") as f:
                    pickle.dump(demo, f)
                    i += 1

        exit(0)

    @staticmethod
    def get_random_fold_action(state: ClothState):
        num_particle = state.x.shape[1]
        batch_size = state.x.shape[0]
        batch_idx = jnp.arange(batch_size)

        st_point = np.random.randint(0, num_particle, size=(batch_size,))
        ed_point = np.random.randint(0, num_particle, size=(batch_size,))

        actions = jnp.concatenate((state.x[batch_idx, st_point], state.x[batch_idx, ed_point]), axis=-1)
        return actions
    
    @staticmethod
    def ensure_pick_on_cloth(pick_point, cloth_particles, cloth_mask, grid_size):
        """
        确保pick点在布料上，如果不在则找到最近的布料点
        
        Args:
            pick_point: 3D pick点坐标 (x, y, z)
            cloth_particles: 所有布料粒子坐标 (N, 3)
            cloth_mask: 布料mask (grid_size, grid_size)
            grid_size: 网格大小
            
        Returns:
            corrected_pick_point: 修正后的pick点坐标
        """
        # 将pick点转换为网格坐标
        pick_2d = jnp.clip(pick_point[:2] * grid_size, 0, grid_size - 1).astype(jnp.int32)
        
        # 检查是否在布料上
        is_on_cloth = cloth_mask[pick_2d[1], pick_2d[0]]  # 注意：mask是(i,j)格式
        
        def use_original_pick():
            return pick_point
            
        def find_nearest_cloth():
            # 计算pick点到所有布料粒子的距离
            distances = jnp.sqrt(jnp.sum((cloth_particles - pick_point) ** 2, axis=-1))
            # 找到最近的粒子索引
            nearest_idx = jnp.argmin(distances)
            return cloth_particles[nearest_idx]
        
        # 如果pick点在布料上，使用原点；否则找到最近的布料点
        corrected_pick = jax.lax.cond(is_on_cloth, use_original_pick, find_nearest_cloth)
        return corrected_pick
    
    @staticmethod
    def generate_cloth_mask_from_state(state: ClothState, grid_size: int = 64):
        """
        从当前状态生成cloth mask
        
        Args:
            state: ClothState对象
            grid_size: 网格大小
            
        Returns:
            cloth_masks: (batch_size, grid_size, grid_size)的布料mask数组
        """
        batch_size = state.x.shape[0]
        cloth_masks = []
        
        for b in range(batch_size):
            # 提取当前batch的布料粒子坐标
            cloth_particles = state.x[b]  # (num_particles, 3)
            
            # 将粒子坐标转换为网格坐标
            coords_2d = jnp.clip(cloth_particles[:, [0, 2]] * grid_size, 0, grid_size - 1).astype(jnp.int32)
            
            # 创建mask
            cloth_mask = jnp.zeros((grid_size, grid_size), dtype=jnp.bool_)
            cloth_mask = cloth_mask.at[coords_2d[:, 1], coords_2d[:, 0]].set(True)
            
            cloth_masks.append(cloth_mask)
        
        return jnp.stack(cloth_masks)  # (batch_size, grid_size, grid_size)
    
    @staticmethod
    def get_x_grid_static(x, idx_i_b, idx_j_b, grid_size):
        """
        静态版本的get_x_grid，用于生成cloth mask
        """
        # 简化的x_grid生成逻辑
        x_grid = jnp.zeros((grid_size, grid_size, 3))
        x_grid = x_grid.at[idx_i_b[0], idx_j_b[0]].set(x[0])
        return x_grid[None, ...]  # 添加batch维度
    
    @staticmethod
    def check_and_correct_dual_arm_pick_points(actions, states, cloth_masks, grid_size):
        """
        检查并修正双臂action中的pick点
        
        Args:
            actions: (batch_size, 12) 的action数组
            cloth_masks: (batch_size, grid_size, grid_size) 的布料mask数组
            grid_size: 网格大小
            
        Returns:
            corrected_actions: 修正后的action数组
        """
        # 确保输入是JAX数组
        actions = jnp.array(actions)
        cloth_masks = jnp.array(cloth_masks)
        batch_size = actions.shape[0]
        
        # 提取pick点 (batch_size, 3)
        arm1_pick = actions[:, :3]  # x, z, y 格式
        arm1_place = actions[:, 3:6]
        arm2_pick = actions[:, 6:9]  # x, z, y 格式
        arm2_place = actions[:, 9:12]
        corrected_actions = []
        
        for i in range(batch_size):
            # 使用第i个batch对应的mask修正第一个pick点
            arm1_pick_corrected = ClothEnv.correct_single_pick_point(
                arm1_pick[i], states.x[i], None, grid_size
            )
            arm1_place_corrected = ClothEnv.correct_single_pick_point(
                arm1_place[i], states.x[i], None, grid_size
            )
            # 使用第i个batch对应的mask修正第二个pick点
            arm2_pick_corrected = ClothEnv.correct_single_pick_point(
                arm2_pick[i], states.x[i], None, grid_size
            )
            arm2_place_corrected = ClothEnv.correct_single_pick_point(
                arm2_place[i], states.x[i], None, grid_size
            )
            # 重新组合action
            # corrected_action = jnp.concatenate([
            #     arm1_pick_corrected,  # 0:3
            #     actions[i, 3:6],      # place1 3:6
            #     arm2_pick_corrected,  # 6:9
            #     actions[i, 9:12]      # place2 9:12
            # ])
            corrected_action = jnp.concatenate([
                arm1_pick_corrected,  # 0:3
                arm1_place_corrected,      # place1 3:6
                arm2_pick_corrected,  # 6:9
                arm2_place_corrected      # place2 9:12
            ])
            
            corrected_actions.append(corrected_action)
        
        return jnp.stack(corrected_actions)
    
    @staticmethod
    def correct_single_pick_point(pick_point, state_x, cloth_mask, grid_size):
        """
        修正单个pick点
        
        Args:
            pick_point: (3,) 的pick点坐标 [x, z, y]
            cloth_mask: (grid_size, grid_size) 的布料mask
            grid_size: 网格大小
            
        Returns:
            corrected_pick_point: 修正后的pick点
        """
        # 确保输入都是JAX数组
        pick_point = jnp.array(pick_point)
        #resize the cloth mask to grid_size, grid_size
        
        state_x = jnp.array(state_x)
        distances = jnp.sqrt(jnp.sum((state_x - pick_point) ** 2, axis=-1))
        nearest_idx = jnp.argmin(distances)
        min_distance = distances[nearest_idx]
        if min_distance > 0.1:
            print(pick_point)
            print(state_x[nearest_idx])
            print(f'Min distance: {min_distance}, pick point is too far from cloth, use original pick point')
            return pick_point
        # elif min_distance > 0.001:
        #     print(pick_point, state_x[nearest_idx])
        nearest_point = state_x[nearest_idx]
        return nearest_point

    @staticmethod
    def get_random_dual_arm_fold_action(state: ClothState):
        num_particle = state.x.shape[1]
        batch_size = state.x.shape[0]
        batch_idx = jnp.arange(batch_size)

        arm1_pick_idx = np.random.randint(0, num_particle, size=(batch_size,))
        arm1_place_idx = np.random.randint(0, num_particle, size=(batch_size,))
        arm2_pick_idx = np.random.randint(0, num_particle, size=(batch_size,))
        arm2_place_idx = np.random.randint(0, num_particle, size=(batch_size,))

        actions = jnp.concatenate((
            state.x[batch_idx, arm1_pick_idx],
            state.x[batch_idx, arm1_place_idx],
            state.x[batch_idx, arm2_pick_idx],
            state.x[batch_idx, arm2_place_idx]
        ), axis=-1)
        return actions