import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import cv2
import trimesh
from dataclasses import dataclass
from functools import partial
from jax import vmap

# 添加路径
sys.path.append('/root/DaXBench')

# 导入基础模块
from daxbench.core.envs.basic.cloth_env import ClothEnv, ClothState

# 配置类
@dataclass
class CustomClothConf:
    N = 200  # 增加分辨率来减少锯齿，可以设置为64, 128, 256
    size = 8
    gravity = 9.8
    stiffness = 1000.0
    damping = 0.999
    dt = 5e-4
    max_v = 10.0
    small_num = 1e-5
    mu = 0.5
    seed = 1
    mem_saving_level = 2
    
    obj_file_path = None
    task = "custom_cloth"
    use_substep_obs = False
    
    # 新增缩放参数
    cloth_scale = 0.5  # 衣服缩放因子，1.0为原始大小，0.5为一半大小，2.0为两倍大小
    cloth_center_x = 0.5  # 衣服在X轴的中心位置 (0.0-1.0)
    cloth_center_y = 0.5  # 衣服在Y轴的中心位置 (0.0-1.0)
    
    def __post_init__(self):
        my_path = os.path.dirname(os.path.abspath(__file__))
        self.goal_path = f"{my_path}/daxbench/core/envs/goals/{self.task}/goal.npy"

def analyze_mesh_orientation(mesh):
    """
    分析mesh的朝向，确定正前方视角
    """
    vertices = mesh.vertices
    extents = mesh.extents
    
    print(f"Mesh extents: X={extents[0]:.4f}, Y={extents[1]:.4f}, Z={extents[2]:.4f}")
    print(f"Mesh bounds min: {vertices.min(axis=0)}")
    print(f"Mesh bounds max: {vertices.max(axis=0)}")
    
    # 对于T-shirt，通常：
    # X轴：左右方向（宽度）
    # Y轴：上下方向（高度） 
    # Z轴：前后方向（厚度）
    
    # 所以正前方视角应该是看X-Y平面（忽略Z轴的深度）
    return [0, 1]  # X和Y轴

def obj_to_front_view_mask(obj_file_path, image_size=128):
    """
    从正前方视角创建T-shirt mask
    """
    mesh = trimesh.load(obj_file_path)
    vertices = mesh.vertices
    faces = mesh.faces
    
    # 分析mesh朝向
    projection_axes = analyze_mesh_orientation(mesh)
    print(f"使用投影轴: {projection_axes} (正前方视角)")
    
    # 提取正前方视角的2D坐标 (X-Y平面)
    vertices_2d = vertices[:, projection_axes]
    
    # 归一化到图像尺寸
    min_coords = vertices_2d.min(axis=0)
    max_coords = vertices_2d.max(axis=0)
    
    print(f"2D投影范围: X=[{min_coords[0]:.4f}, {max_coords[0]:.4f}], Y=[{min_coords[1]:.4f}, {max_coords[1]:.4f}]")
    
    # 计算缩放，保持长宽比
    margin = 5
    available_size = image_size - 2 * margin
    ranges = max_coords - min_coords
    scale = available_size / ranges.max()
    
    # 缩放坐标
    vertices_scaled = (vertices_2d - min_coords) * scale
    
    # 居中
    scaled_size = ranges * scale
    center_offset = (available_size - scaled_size) / 2 + margin
    vertices_scaled += center_offset
    
    print(f"缩放后范围: {vertices_scaled.min(axis=0)} 到 {vertices_scaled.max(axis=0)}")
    
    # 创建高分辨率图像进行精确渲染
    high_res = image_size * 4
    vertices_high_res = vertices_scaled * 4
    
    # 使用面片信息渲染
    img = np.zeros((high_res, high_res), dtype=np.uint8)
    
    # 渲染所有三角面片
    for face in faces:
        # 获取三角形的2D投影坐标
        triangle_2d = vertices_high_res[face]
        
        # 转换为整数坐标
        pts = triangle_2d.astype(np.int32)
        
        # 确保坐标在有效范围内
        pts = np.clip(pts, 0, high_res - 1)
        
        # 检查三角形是否有效（面积大于0）
        if cv2.contourArea(pts) > 1:
            cv2.fillPoly(img, [pts], 255)
    
    # 降采样回原始尺寸
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    
    # 二值化
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    
    # 后处理：清理噪点
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return img


def obj_to_smooth_mask(obj_file_path, image_size=128, anti_aliasing=True, cloth_scale=1.0, center_x=0.5, center_y=0.5):
    """
    生成平滑的无锯齿mask，支持缩放和位置控制
    
    Parameters:
    - cloth_scale: 缩放因子 (1.0=原始大小, 0.5=一半大小, 2.0=两倍大小)
    - center_x: X轴中心位置 (0.0-1.0)
    - center_y: Y轴中心位置 (0.0-1.0)
    """
    mesh = trimesh.load(obj_file_path)
    vertices = mesh.vertices
    faces = mesh.faces
    
    # 分析mesh朝向
    projection_axes = analyze_mesh_orientation(mesh)
    print(f"使用投影轴: {projection_axes} (正前方视角)")
    print(f"缩放因子: {cloth_scale}, 中心位置: ({center_x}, {center_y})")
    
    # 提取正前方视角的2D坐标 (X-Y平面)
    vertices_2d = vertices[:, projection_axes]
    
    # 归一化到图像尺寸
    min_coords = vertices_2d.min(axis=0)
    max_coords = vertices_2d.max(axis=0)
    
    print(f"2D投影范围: X=[{min_coords[0]:.4f}, {max_coords[0]:.4f}], Y=[{min_coords[1]:.4f}, {max_coords[1]:.4f}]")
    
    # 计算缩放，保持长宽比，并应用用户指定的缩放因子
    margin = 8
    available_size = image_size - 2 * margin
    ranges = max_coords - min_coords
    
    # 基础缩放（保持长宽比）
    base_scale = available_size / ranges.max()
    
    # 应用用户缩放因子
    final_scale = base_scale * cloth_scale
    
    # 缩放坐标
    vertices_scaled = (vertices_2d - min_coords) * final_scale
    
    # 计算缩放后的尺寸
    scaled_size = ranges * final_scale
    
    # 用户指定的中心位置（0.0-1.0范围转换为像素坐标）
    target_center_x = center_x * image_size
    target_center_y = center_y * image_size
    
    # 计算偏移量以将衣服放置到指定位置
    offset_x = target_center_x - scaled_size[0] / 2
    offset_y = target_center_y - scaled_size[1] / 2
    
    vertices_scaled[:, 0] += offset_x
    vertices_scaled[:, 1] += offset_y
    
    print(f"最终缩放: {final_scale:.4f}, 偏移: ({offset_x:.2f}, {offset_y:.2f})")
    print(f"缩放后范围: {vertices_scaled.min(axis=0)} 到 {vertices_scaled.max(axis=0)}")
    
    if anti_aliasing:
        # 使用更高分辨率进行抗锯齿
        super_res = image_size * 8  # 8倍超采样
        vertices_super_res = vertices_scaled * 8
        
        # 创建超高分辨率图像
        img = np.zeros((super_res, super_res), dtype=np.float32)
        
        # 渲染所有三角面片
        for face in faces:
            triangle_2d = vertices_super_res[face]
            pts = triangle_2d.astype(np.int32)
            pts = np.clip(pts, 0, super_res - 1)
            
            if cv2.contourArea(pts) > 1:
                cv2.fillPoly(img, [pts], 1.0)
        
        # 使用高质量降采样
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        
        # 平滑边缘
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        # 转换为二值图像，但保持边缘平滑
        img = (img * 255).astype(np.uint8)
        
    else:
        # 标准方法
        img = obj_to_front_view_mask(obj_file_path, image_size)
    
    return img

def create_high_quality_cloth_mask(obj_file_path, grid_size=128, cloth_scale=1.0, center_x=0.5, center_y=0.5):
    """
    创建高质量的布料mask，减少锯齿，支持缩放和位置控制
    
    Parameters:
    - cloth_scale: 缩放因子 (1.0=原始大小, 0.5=一半大小, 2.0=两倍大小)
    - center_x: X轴中心位置 (0.0-1.0)
    - center_y: Y轴中心位置 (0.0-1.0)
    """
    # 方法1: 超采样抗锯齿，带缩放控制
    mask_smooth = obj_to_smooth_mask(obj_file_path, grid_size, anti_aliasing=True, 
                                   cloth_scale=cloth_scale, center_x=center_x, center_y=center_y)
    
    # 方法2: 形态学平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 方法3: 边缘平滑
    mask_smooth = cv2.medianBlur(mask_smooth, 3)
    
    # 最终二值化，但使用更宽松的阈值来保持边缘
    _, mask_smooth = cv2.threshold(mask_smooth, 100, 255, cv2.THRESH_BINARY)
    
    return mask_smooth

# 自定义布料环境类
class CustomClothEnv(ClothEnv):
    
    def __init__(self, batch_size, obj_file_path, conf=None, aux_reward=False, seed=1):
        if conf is None:
            conf = CustomClothConf()
        conf.obj_file_path = obj_file_path
        conf.seed = seed
        self.conf = conf
        
        max_steps = 10
        super().__init__(conf, batch_size, max_steps, aux_reward)
        
    def create_cloth_mask(self, conf):
        """从OBJ文件创建cloth mask"""
        return self.obj_to_cloth_mask(conf.obj_file_path, conf.N)
    
    def obj_to_cloth_mask(self, obj_file_path, grid_size):
        """将OBJ文件转换为cloth mask - 使用改进的高质量方法，支持缩放控制"""
        # 使用高质量的mask生成方法，并应用配置中的缩放参数
        mask_img = create_high_quality_cloth_mask(
            obj_file_path, 
            grid_size, 
            cloth_scale=self.conf.cloth_scale,
            center_x=self.conf.cloth_center_x,
            center_y=self.conf.cloth_center_y
        )
        
        # 转换为布料mask
        cloth_mask = (mask_img > 127).astype(np.float32)
        
        return jnp.array(cloth_mask)

# 主函数
def main():
    print("开始支持缩放的布料仿真示例...")
    
    # OBJ文件路径
    obj_file_path = "/root/DaXBench/0005.obj"
    
    # 检查文件是否存在
    if not os.path.exists(obj_file_path):
        print(f"错误：找不到OBJ文件 {obj_file_path}")
        return
    
    # try:
    #     # 展示缩放效果
    #     print("展示不同缩放和位置效果...")
    #     preview_cloth_scaling_effects(
    #         obj_file_path, 
    #         "cloth_scaling_effects.png"
    #     )
    #     print("缩放效果已保存到 cloth_scaling_effects_*.png")
        
    # except Exception as e:
    #     print(f"预览缩放效果时出错: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return
    
    # 创建目标文件夹
    os.makedirs("daxbench/core/envs/goals/custom_cloth", exist_ok=True)
    
    # 创建一个简单的目标文件
    dummy_goal = np.random.random((100, 3))
    np.save("daxbench/core/envs/goals/custom_cloth/goal.npy", dummy_goal)
    
    # 测试不同的缩放设置
    test_configs = [
        {"scale": 0.7, "center_x": 0.5, "center_y": 0.5, "name": "小尺寸_居中"},
        # {"scale": 1.2, "center_x": 0.3, "center_y": 0.3, "name": "大尺寸_左上"},
        # {"scale": 1.0, "center_x": 0.7, "center_y": 0.7, "name": "正常尺寸_右下"},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n=== 测试配置 {i+1}: {config['name']} ===")
        
        # 创建自定义配置
        conf = CustomClothConf()
        conf.cloth_scale = config["scale"]
        conf.cloth_center_x = config["center_x"]
        conf.cloth_center_y = config["center_y"]
        
        # 创建环境
        print(f"创建环境 - 缩放: {config['scale']}, 位置: ({config['center_x']}, {config['center_y']})")
        env = CustomClothEnv(
            batch_size=1,
            obj_file_path=obj_file_path,
            conf=conf,
            seed=42
        )
        
        # 重置环境
        key = random.PRNGKey(42 + i)
        env.simulator.key_global, key = random.split(key)
        obs, state = env.reset(key)
        
        print(f"观察维度: {obs.shape}")
        print(f"布料粒子数量: {int(env.cloth_mask.sum())}")

        for i in range(100):
            # actions = get_expert_start_end_cloth(env.get_x_grid(state), env.cloth_mask)
            actions = env.get_random_fold_action(state)
            # actions = np.zeros((env.batch_size, 6))
            print(actions)
            obs, reward, done, info = env.step_diff(actions, state)
            # obs, reward, done, info = env.step_with_render(actions, state)
            print(222)
            state = info['state']
            rgb, depth = env.render(state, False)
            print(rgb.shape)
            print(depth.shape)
            cv2.imwrite(f'rgb/rgb_{i}.png', rgb)
            cv2.imwrite(f'depth/depth_{i}.png', depth)
            
    #         # 保存并显示cloth mask
    #         import matplotlib.pyplot as plt
    #         plt.figure(figsize=(6, 6))
    #         plt.imshow(np.array(env.cloth_mask), cmap='gray')
    #         plt.title(f'{config["name"]}\n缩放: {config["scale"]}, 位置: ({config["center_x"]}, {config["center_y"]})')
    #         plt.axis('off')
            
    #         # 保存mask
    #         mask_filename = f'cloth_mask_{config["name"]}.png'
    #         plt.savefig(mask_filename, dpi=150, bbox_inches='tight')
    #         print(f"Cloth mask已保存到: {mask_filename}")
            
    #         # 渲染一帧
    #         try:
    #             rgb, depth = env.render(state, visualize=False)
    #             cv2.imwrite(f'render_{config["name"]}.png', rgb)
    #             print(f"渲染图已保存到: render_{config['name']}.png")
    #         except Exception as e:
    #             print(f"渲染时出错: {e}")
            
    #         plt.close()  # 关闭图形避免内存积累
        
    #     print("\n=== 所有测试配置完成！ ===")
    #     print("生成的文件:")
    #     print("- cloth_scaling_effects.png (缩放效果对比)")
    #     print("- cloth_mask_*.png (不同配置的mask)")
    #     print("- render_*.png (渲染效果)")
        
    # except Exception as e:
    #     print(f"运行环境时出错: {e}")
    #     import traceback
    #     traceback.print_exc()

if __name__ == "__main__":
    main()