import os
import time
from dataclasses import dataclass

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
import imageio
from tqdm import tqdm

from daxbench.core.engine.cloth_simulator import ClothState
from daxbench.core.engine.usdrender.mesh_usd import create_usd_cloth_scene
from daxbench.core.envs.basic.cloth_env import ClothEnv
from daxbench.core.utils.util import get_expert_start_end_cloth

from icecream import ic as print

my_path = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DefaultConf:
    N = 200
    cell_size = 1.0 / N
    gravity = 0.5
    stiffness = 5000
    damping = 2
    dt = 0.5e-3
    max_v = 2.
    small_num = 1e-8
    mu = 0.9  # friction
    seed = 1
    size = int(N / 5.0)
    mem_saving_level = 2
    # 0:fast but requires more memory, not recommended
    # 1:lesser memory, but faster
    # 2:much lesser memory but much slower
    task = "fold_tshirt"
    goal_path = f"{my_path}/goals/{task}/goal.npy"
    use_substep_obs = True


FoldTshirtConfig = DefaultConf


class FoldEnv(ClothEnv):

    def __init__(self, batch_size, conf=None, aux_reward=False, seed=1, record_video=False):
        conf = DefaultConf() if conf is None else conf
        max_steps = 5
        super().__init__(conf, batch_size, max_steps, aux_reward)
        self.observation_size = 1082
        self.episode_data = {
            'states': [],
        }
        
        self.record_video = record_video

        self.init_compile()
    
    def init_compile(self):
        obs, state = self.reset(self.simulator.key_global)
        actions = np.zeros((self.batch_size, 6))
        _, _, _, info = self.step_diff(actions, state)
        self.info = info

        if self.record_video:
            self.episode_data['states'].append(self.info['state_list'])

    def create_cloth_mask(self, conf):  
        img = cv2.imread(f"{my_path}/others/t-shirt.jpg")

        size = conf.N // 2
        h_size = size // 2

        img = cv2.resize(img, (size, size))
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # cv2.imshow("img", img)
        # cv2.waitKey(20)

        mask = (img.sum(-1) < 100).astype(np.int32)
        # cv2.imshow("mask", mask.astype(np.float32))
        # cv2.waitKey(20)

        cloth_mask = jnp.zeros((conf.N, conf.N))
        cloth_mask = cloth_mask.at[conf.N // 2 - h_size:conf.N // 2 + h_size,
                     conf.N // 2 - h_size:conf.N // 2 + h_size].set(mask)

        return cloth_mask
    
    def step_fold(self, actions):
        state_before = self.info['state']
        rgbs_before, depths_before = self.render_all(state_before, False)

        obs, reward, done, info = self.step_diff(actions, state_before)
        # obs, reward, done, info = env.step_with_render(actions, state)

        self.info = info

        if self.record_video:
            self.episode_data['states'].append(self.info['state_list'])

        return obs, reward, done, info
    
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
                    rgb_frame, _ = self.render(current_state, visualize=False, idx=b_idx)

                
                    if rgb_frame.dtype != np.uint8:
                        rgb_frame = (rgb_frame * 255).astype(np.uint8)
                    
                    rgb_frames.append(rgb_frame)

            
            if not rgb_frames:
                print(f"Warning: No frames generated for batch {b_idx}")
                continue
            
            filename = f"{filename_prefix}_batch_{b_idx}.{format}"
            filepath = os.path.join(output_dir, filename)
            
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
    env = FoldEnv(batch_size=8, seed=1, record_video=True)
    # env.collect_goal()
    # env.collect_expert_demo(10)
    print("time start")
    start_time = time.time()
    iter_num = 3
    for i in range(iter_num):
        # actions = get_expert_start_end_cloth(env.get_x_grid(state), env.cloth_mask)
        state = env.get_state()
        actions = env.get_random_fold_action(state)
        print(actions.shape)
        print(actions)
        # actions = np.zeros((env.batch_size, 6))

        obs, reward, done, info = env.step_fold(actions)
    
    episode_data = env.get_episode_data()
        
    print('All time: ', time.time() - start_time)
    print('Average time: ', (time.time() - start_time) / iter_num)

    vstart = time.time()
    env.render_video(episode_data, output_dir="./videos", batch_idx=None, format="gif", fps=20, filename_prefix="episode")
    print('Video time: ', time.time() - vstart)
    print('Average video time: ', (time.time() - vstart) / env.batch_size)