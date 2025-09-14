import numpy as np
import trimesh
import cv2
from PIL import Image

def obj_to_image_mask(obj_file_path, image_size=128):
    """将OBJ文件转换为图像mask，类似T-shirt环境的方法"""
    mesh = trimesh.load(obj_file_path)
    
    # 获取2D投影
    vertices_2d = mesh.vertices[:, [0, 2]]  # X-Z 投影
    
    # 归一化到图像尺寸
    min_coords = vertices_2d.min(axis=0)
    max_coords = vertices_2d.max(axis=0)
    
    # 缩放到图像大小
    scale = (image_size - 20) / (max_coords - min_coords).max()  # 留边界
    vertices_scaled = (vertices_2d - min_coords) * scale
    
    # 居中
    center_offset = (image_size - (max_coords - min_coords) * scale) / 2
    vertices_scaled += center_offset
    
    # 创建图像
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # 在顶点位置标记
    coords = vertices_scaled.astype(int)
    coords = np.clip(coords, 0, image_size - 1)
    
    img[coords[:, 1], coords[:, 0]] = 255
    
    # 形态学操作生成连续mask
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=3)
    
    # 填充轮廓
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.fillPoly(img, [largest_contour], 255)
    
    return img

def preview_cloth_mask(obj_file_path, output_path=None):
    """预览生成的cloth mask"""
    mask_img = obj_to_image_mask(obj_file_path)
    
    if output_path:
        cv2.imwrite(output_path, mask_img)
    
    # # 显示预览
    # cv2.imshow("Cloth Mask Preview", mask_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return mask_img 