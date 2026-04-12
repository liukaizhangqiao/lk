
import cv2
import numpy as np
import os
from pathlib import Path
def mask_to_skeleton(image_path, output_path):
    """
    将单张黑白掩码图像转换为骨架图像
    :param image_path: 输入掩码图像路径
    :param output_path: 输出骨架图像路径
    """
    # 1. 读取黑白掩码图像（灰度模式）
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"警告：无法读取图像 {image_path}，跳过该文件")
        return

    # 2. 确保图像是二值化的（裂缝为白色255，背景为黑色0）
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 3. 骨架提取（核心操作）
    # 创建细化器，选择经典的Zhang-Suen算法（高效且适合线性结构）
    skeleton = cv2.ximgproc.thinning(binary_img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # 4. 保存结果
    cv2.imwrite(output_path, skeleton)
    print(f"已处理并保存：{output_path}")


def batch_process_skeleton(input_folder, output_folder):
    """
    批量处理文件夹内的所有图像，生成骨架图像
    :param input_folder: 输入图像文件夹路径
    :param output_folder: 输出骨架图像文件夹路径
    """
    # 创建输出文件夹（如果不存在）
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # 支持的图像格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # 遍历文件夹内的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件格式是否支持
        if filename.lower().endswith(supported_formats):
            # 构建输入输出路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 处理单张图像
            mask_to_skeleton(input_path, output_path)

    print("\n批量处理完成！")


# 调用示例
if __name__ == "__main__":
    # 配置输入输出文件夹路径
    input_folder = r"E:\bone_crack500\unet\pre_bone"  # 输入文件夹
    output_folder = r"E:\bone_crack500\unet\pre_bone_line"  # 输出文件夹

    # 执行批量处理
    batch_process_skeleton(input_folder, output_folder)