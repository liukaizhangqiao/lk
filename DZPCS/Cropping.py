import os
import cv2
import numpy as np
from tqdm import tqdm  # 进度条库，提升体验

class Crack500Splitter:
    """
    模拟Crack500数据集的分割逻辑：滑动窗口裁剪大图为小图，并筛选有效子图
    """
    def __init__(self,
                 src_img_dir,        # 原始图像目录
                 src_label_dir,      # 原始标注图目录（二值图）
                 save_img_dir,       # 裁剪后图像保存目录
                 save_label_dir,     # 裁剪后标注图保存目录
                 patch_size=512,     # 子图尺寸（Crack500常用256×256）
                 stride=256,         # 滑动步长（重叠50%）
                 crack_thresh=5):    # 裂缝像素阈值：子图中裂缝像素数≥此值才保留
        self.src_img_dir = src_img_dir
        self.src_label_dir = src_label_dir
        self.save_img_dir = save_img_dir
        self.save_label_dir = save_label_dir
        self.patch_size = patch_size
        self.stride = stride
        self.crack_thresh = crack_thresh

        # 创建保存目录
        os.makedirs(self.save_img_dir, exist_ok=True)
        os.makedirs(self.save_label_dir, exist_ok=True)

    def _get_crack_pixel_count(self, label_patch):
        """
        计算标注子图中裂缝像素的数量（二值图中白色为裂缝，值为255）
        """
        # 转为二值图（确保只有0和255）
        _, binary_label = cv2.threshold(label_patch, 127, 255, cv2.THRESH_BINARY)
        # 统计白色像素数（裂缝）
        crack_pixels = np.sum(binary_label == 255)
        return crack_pixels

    def split_single_image(self, img_name):
        """
        分割单张原始图像及其标注图
        """
        # 1. 读取原始图像和标注图
        img_path = os.path.join(self.src_img_dir, img_name)
        label_name = img_name.replace('.jpg', '.png')  # 假设标注图为png格式
        label_path = os.path.join(self.src_label_dir, label_name)

        if not os.path.exists(label_path):
            print(f"警告：{img_name} 无对应标注图，跳过")
            return

        img = cv2.imread(img_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # 灰度读取标注图

        if img is None or label is None:
            print(f"警告：{img_name} 读取失败，跳过")
            return

        h, w = img.shape[:2]
        patch_idx = 0  # 子图序号

        # 2. 滑动窗口遍历裁剪
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                # 裁剪图像子图
                img_patch = img[y:y+self.patch_size, x:x+self.patch_size]
                # 裁剪标注子图
                label_patch = label[y:y+self.patch_size, x:x+self.patch_size]

                # 3. 筛选有效子图（核心：保留含裂缝的子图，少量纯背景子图）
                crack_pixels = self._get_crack_pixel_count(label_patch)
                # 条件1：裂缝像素数≥阈值 → 保留（核心样本）
                # 条件2：纯背景但每100张子图保留1张 → 平衡正负样本
                if crack_pixels >= self.crack_thresh or (patch_idx % 100 == 0 and crack_pixels == 0):
                    # 构造保存文件名
                    save_name = f"{img_name.split('.')[0]}_patch{patch_idx}.png"
                    save_label_name = f"{label_name.split('.')[0]}_patch{patch_idx}.png"

                    # 保存子图
                    cv2.imwrite(os.path.join(self.save_img_dir, save_name), img_patch)
                    cv2.imwrite(os.path.join(self.save_label_dir, save_label_name), label_patch)

                patch_idx += 1

    def split_all_images(self):
        """
        批量分割所有原始图像
        """
        img_list = [f for f in os.listdir(self.src_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        for img_name in tqdm(img_list, desc="分割进度"):
            self.split_single_image(img_name)
        print(f"分割完成！裁剪后的图像保存在：{self.save_img_dir}")
        print(f"裁剪后的标注保存在：{self.save_label_dir}")

if __name__ == "__main__":
    # ===================== 配置参数 =====================
    # 请根据你的实际路径修改！
    SRC_IMG_DIR = r"C:\Users\ASUS\Desktop\opencv_generate_mask\llm_crack500"    # 原始500张图像目录
    SRC_LABEL_DIR = r"C:\Users\ASUS\Desktop\opencv_generate_mask\llm_crack500_mask"  # 原始标注图目录
    SAVE_IMG_DIR = r"C:\Users\ASUS\Desktop\opencv_generate_mask\llm_crack500_clip"      # 裁剪后图像保存目录
    SAVE_LABEL_DIR = r"C:\Users\ASUS\Desktop\opencv_generate_mask\llm_crack500_mask_clip"    # 裁剪后标注保存目录

    # ===================== 执行分割 =====================
    splitter = Crack500Splitter(
        src_img_dir=SRC_IMG_DIR,
        src_label_dir=SRC_LABEL_DIR,
        save_img_dir=SAVE_IMG_DIR,
        save_label_dir=SAVE_LABEL_DIR,
        patch_size=512,    # 【修改1】改为512×512
        stride=256,        # 【修改2】步长改为256（保持50%重叠）
        crack_thresh=5     # 裂缝像素阈值，可根据需求调整
    )
    splitter.split_all_images()