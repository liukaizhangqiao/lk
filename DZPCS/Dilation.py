import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # 进度条，需安装：pip install tqdm

# --------------------------
# 解决Matplotlib中文显示问题
# --------------------------
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False


def generate_marginal_mask_opencv(y_gt, kernel_size=5):
    """生成边际掩码+膨胀掩码"""
    if not np.all(np.isin(y_gt, [0, 1])):
        raise ValueError("输入掩码必须是0-1二值图！")

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    y_gt_uint8 = (y_gt * 255).astype(np.uint8)
    dilated_mask_uint8 = cv2.dilate(y_gt_uint8, kernel, iterations=1)
    dilated_mask = (dilated_mask_uint8 / 255).astype(np.uint8)

    M = dilated_mask - y_gt
    M = np.maximum(M, 0)
    return M.astype(np.uint8), dilated_mask.astype(np.uint8)


def save_mask_as_image(mask, save_path, is_binary=True):
    """保存掩码为图片"""
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    mask_save = (mask * 255).astype(np.uint8) if is_binary else mask.astype(np.uint8)
    cv2.imwrite(save_path, mask_save)


def batch_process_masks(input_dir, output_root, kernel_size=5):
    """
    批量处理目录下的所有掩码文件
    Args:
        input_dir: 原始掩码文件所在目录
        output_root: 输出文件根目录
        kernel_size: 膨胀核大小（论文指定5）
    """
    # 1. 获取目录下所有PNG掩码文件
    mask_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
    if not mask_files:
        raise FileNotFoundError(f"目录{input_dir}下未找到PNG掩码文件！")

    # 2. 遍历所有文件，批量处理
    for mask_file in tqdm(mask_files, desc="批量处理掩码"):
        # 2.1 读取单个掩码文件
        mask_path = os.path.join(input_dir, mask_file)
        y_gt_original = cv2.imread(mask_path, 0)  # 单通道灰度图
        if y_gt_original is None:
            print(f"⚠️  跳过无效文件：{mask_file}")
            continue

        # 2.2 转为0-1二值图（CrackTree200掩码：裂缝=255→1，背景=0→0）
        y_gt = (y_gt_original == 255).astype(np.uint8)

        # 2.3 生成边际掩码+膨胀掩码
        marginal_mask, dilated_mask = generate_marginal_mask_opencv(y_gt, kernel_size)

        # 2.4 定义输出路径（按类型分类保存）
        base_name = os.path.splitext(mask_file)[0]  # 文件名前缀
        # 原始掩码输出路径
        original_save_path = os.path.join(output_root, "original_masks", f"{base_name}_original.png")
        # 膨胀掩码输出路径
        dilated_save_path = os.path.join(output_root, "dilated_masks", f"{base_name}.png")
        # 边际掩码输出路径
        marginal_save_path = os.path.join(output_root, "marginal_masks", f"{base_name}_marginal.png")

        # 2.5 保存文件
        save_mask_as_image(y_gt, original_save_path)
        save_mask_as_image(dilated_mask, dilated_save_path)
        save_mask_as_image(marginal_mask, marginal_save_path)

    print(f"\n✅ 批量处理完成！所有文件已保存至：{output_root}")
    print(f"  - 原始掩码：{os.path.join(output_root, 'original_masks')}")
    print(f"  - 膨胀掩码：{os.path.join(output_root, 'dilated_masks')}")
    print(f"  - 边际掩码：{os.path.join(output_root, 'marginal_masks')}")


# --------------------------
# 运行批量处理（关键：用原始字符串r""或双反斜杠\\）
# --------------------------
if __name__ == "__main__":
    # 配置路径（修复转义错误：用r""原始字符串，或把\改为\\）
    INPUT_DIR = r"C:\Users\ASUS\Desktop\opencv_generate_mask\cracktree200"  # 你的掩码目录
    OUTPUT_ROOT = r"C:\Users\ASUS\Desktop\opencv_generate_mask\cracktree200_batch_output_3"  # 输出目录

    # 启动批量处理
    batch_process_masks(
        input_dir=INPUT_DIR,
        output_root=OUTPUT_ROOT,
        kernel_size=7
    )