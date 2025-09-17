import time
import pickle
import numpy as np
import argparse
from daxbench.core.envs.fold_env import FoldEnv, DefaultConf
import os
from expert.fold_direction_langchain import *
import cv2

def transform_to_array(data_list):
  result_list = []
  for item in data_list:
      for sub_item in item:
          result_list.extend([sub_item[0], 0, sub_item[1]])
  
  return np.array(result_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='S_Corner_All_Middle',
                       help='任务名称')
    parser.add_argument('--only_render_first_image', action='store_true', default=False,
                       help='只渲染第一张图片')
    args = parser.parse_args()
    oracle_base_dir = 'oracle'
    task = args.task
    only_render_first_image = args.only_render_first_image
    task_dir = os.path.join(oracle_base_dir, task)
    task_func = globals()[task]()
    conf = DefaultConf()
    # if task.startswith('T') or task.startswith('P'):
    #     conf.N = 150
    
    conf.id_range = list(range(0, 50))
    conf.batch_size = len(conf.id_range)

    if only_render_first_image:
        conf.id_range = list(range(0, 100))
        conf.batch_size = len(conf.id_range)

    # conf.id_range = list(range(0, 50))
    # conf.id_range = list(range(50, 100))
    if task.startswith('S'):
        conf.cloth_type = 'square'
    elif task.startswith('T'):
        conf.cloth_type = 'tshirt'
    elif task.startswith('P'):
        conf.cloth_type = 'pant'
    elif task.startswith('R'):
        conf.cloth_type = 'rectangle'
    conf.task = task
    conf.record_video = True
    env = FoldEnv(conf=conf, seed=1)    

    iter_num = task_func.steps()
    for i in range(iter_num):
        state_before = env.get_state()
        rgbs_before, _, _ = env.render_all(state_before, need_mask=True, visualize=False)

        if only_render_first_image:
            for batch_idx in range(conf.batch_size):
                if not os.path.exists(os.path.join('first_image', f'{conf.cloth_type}')):
                    os.makedirs(os.path.join('first_image', f'{conf.cloth_type}'))
                cv2.imwrite(os.path.join('first_image', f'{conf.cloth_type}', f"rgbs_{env.chosen_ids[batch_idx]}.png"), rgbs_before[batch_idx])
            exit()
        
        for batch_idx in range(conf.batch_size):
            if not os.path.exists(os.path.join(task_dir, f"{batch_idx}", f"{i}")):
                os.makedirs(os.path.join(task_dir, f"{batch_idx}", f"{i}"))
            cv2.imwrite(os.path.join(task_dir, f"{batch_idx}", f"{i}", f"rgb_before.png"), rgbs_before[batch_idx])
        actions = np.zeros((conf.batch_size, 12))
        for batch_idx in range(conf.batch_size):
            actions[batch_idx] = transform_to_array(task_func.oracle_fold(*env.gt_keypoints_list[batch_idx], step=i))
            actions[batch_idx] = actions[batch_idx] / 400
        obs, reward, done, info = env.step_fold(actions)
        states = env.get_state()
        rgbs_after, _, _ = env.render_all(states, need_mask=True, visualize=False)
        for batch_idx in range(conf.batch_size):
            if not os.path.exists(os.path.join(task_dir, f"{batch_idx}", f"{i}")):
                os.makedirs(os.path.join(task_dir, f"{batch_idx}", f"{i}"))
            cv2.imwrite(os.path.join(task_dir, f"{batch_idx}", f"{i}", f"rgb_after.png"), rgbs_after[batch_idx])
        for batch_idx in range(conf.batch_size):
            if not os.path.exists(os.path.join(task_dir, f"{batch_idx}", "subgoals")):
                os.makedirs(os.path.join(task_dir, f"{batch_idx}", "subgoals"))
            with open(os.path.join(task_dir, f"{batch_idx}", "subgoals", f"{i}.pkl"), "wb") as f:
                pickle.dump(states.x[batch_idx], f)


    states = env.get_state()
    assert states.x.shape[0] == conf.batch_size
    for batch_idx in range(conf.batch_size):
        if not os.path.exists(os.path.join(task_dir, f"{batch_idx}")):
            os.makedirs(os.path.join(task_dir, f"{batch_idx}"))
        with open(os.path.join(task_dir, f"{batch_idx}", "goal.pkl"), "wb") as f:
            pickle.dump(states.x[batch_idx], f)
    

    episode_data = env.get_episode_data()

    env.render_video(episode_data, output_dir=task_dir, batch_idx=None, format="gif", fps=20, filename_prefix="episode")
