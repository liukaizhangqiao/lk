import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# -------------------------- 核心配置参数（重点调整） --------------------------
img_defect_path = r"E:\deepcrack18\003.png"  # 有缺陷图路径
img_normal_path = r"E:\deepcrack18\003_.png"  # 无缺陷原图路径
save_mask_path = r"E:\deepcrack18\003_mask.png" # 掩码保存路径

# 新增：所有图像的保存根路径（可自定义）
save_root_dir = r"E:\deepcrack18\result2_images"
# 确保保存目录存在
os.makedirs(save_root_dir, exist_ok=True)

# 形态学操作参数
morph_close_kernel_size = (5, 5)   # 闭运算核（填充孔洞）
morph_close_iterations = 2         # 闭运算迭代次数
morph_open_iterations = 1          # 开运算迭代次数
morph_erode_kernel_size = (3,3)   # 腐蚀核（收缩过大的边界）
morph_erode_iterations =1 # 腐蚀迭代次数（建议1-2）

connectivity = 8  # 连通域分析的邻域（8邻域）
area_threshold_mode = "median"     # 面积阈值模式："median"/"manual"
manual_area_threshold =15 # 手动阈值（像素）
manual_diff_threshold =9 # 手动差分阈值（替代Otsu，适合小占比裂缝）
use_manual_threshold = True        # 是否使用手动阈值（True=手动，False=Otsu）

# 新增：差分图预处理参数（控制阈值前收缩程度）
blur_kernel_size = (3,3)  # 高斯模糊核（3x3保留细节，5x5弱化噪点）
shrink_kernel = (3,3)     # 收缩核（3x3轻微收缩，5x5明显收缩）

# -----------------------------------------------------------------------------------

# 检查文件是否存在
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件！路径：{file_path}")
        return False
    return True

# 校验输入文件
if not check_file_exists(img_defect_path) or not check_file_exists(img_normal_path):
    exit(1)

# 读取图像（灰度模式）
img_defect = cv2.imread(img_defect_path, cv2.IMREAD_GRAYSCALE)
img_normal = cv2.imread(img_normal_path, cv2.IMREAD_GRAYSCALE)

# 检查图像读取是否成功
def check_img_read(img, img_name):
    if img is None:
        print(f"❌ 错误：{img_name} 读取失败！可能是文件损坏或格式不支持")
        return False
    return True

if not check_img_read(img_defect, "有缺陷图") or not check_img_read(img_normal, "无缺陷原图"):
    exit(1)

# 确保两张图像尺寸一致（按原图尺寸缩放缺陷图）
if img_defect.shape != img_normal.shape:
    print(f"⚠️  提示：两张图尺寸不一致！")
    print(f"   无缺陷图尺寸：{img_normal.shape} | 有缺陷图尺寸：{img_defect.shape}")
    img_defect = cv2.resize(img_defect, (img_normal.shape[1], img_normal.shape[0]),interpolation=cv2.INTER_LINEAR)
    print(f"✅ 已将有缺陷图缩放到与原图一致，新尺寸：{img_defect.shape}")

# 计算像素差分图（绝对值差）
diff_original = cv2.absdiff(img_normal, img_defect)  # 保留原始差分图（用于对比）
diff_processed = diff_original.copy()                # 预处理用的差分图

# -------------------------- 新增：差分图预处理（阈值前收缩白色区域） --------------------------
# 1. 轻量高斯模糊，弱化边缘细碎噪点
diff_processed = cv2.GaussianBlur(diff_processed, blur_kernel_size, 0)

# 2. 自定义收缩函数：收缩白色区域宽度
def shrink_diff_image(diff_img, kernel_size=(3,3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # 开运算收缩：保留核心白色区域，去除边缘扩展部分
    shrinked_diff = cv2.morphologyEx(diff_img, cv2.MORPH_OPEN, kernel)
    return shrinked_diff

# 执行收缩
diff_processed = shrink_diff_image(diff_processed, kernel_size=shrink_kernel)
print(f"✅ 已在阈值前收缩差分图，模糊核{blur_kernel_size}，收缩核{shrink_kernel}")
# -----------------------------------------------------------------------------------

# 阈值分割：基于预处理后的差分图
if use_manual_threshold:
    _, mask = cv2.threshold(diff_processed, manual_diff_threshold, 255, cv2.THRESH_BINARY)
    print(f"📌 使用手动差分阈值：{manual_diff_threshold}")
else:
    _, mask = cv2.threshold(diff_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"📌 Otsu自动计算的阈值：{_}")

# -------------------------- 形态学操作优化 --------------------------
# 1. 闭运算：填充缺陷内部的小黑点/孔洞
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_close_kernel_size)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=morph_close_iterations)

# 2. 腐蚀操作：收缩过大的裂缝边界（核心优化点）
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_erode_kernel_size)
mask = cv2.erode(mask, erode_kernel, iterations=morph_erode_iterations)
print(f"✅ 已执行腐蚀操作：核尺寸{morph_erode_kernel_size}，迭代{morph_erode_iterations}次")

# 3. 可选：轻量开运算去除小噪点
if morph_open_iterations > 0:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=morph_open_iterations)

# -------------------------- 连通域分析：过滤小噪点 --------------------------
# 寻找所有连通域（包含统计信息）
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
foreground_areas = stats[1:, cv2.CC_STAT_AREA]  # 提取前景连通域面积（跳过背景）

if len(foreground_areas) == 0:
    filtered_mask = np.zeros_like(mask)
    print("⚠️  提示：未检测到前景连通域，掩码为全黑")
else:
    # 确定面积阈值
    if area_threshold_mode == "median":
        area_threshold = np.median(foreground_areas)
        print(f"📌 连通域面积中位数：{area_threshold:.1f} 像素（作为过滤阈值）")
    elif area_threshold_mode == "manual":
        area_threshold = manual_area_threshold
        print(f"📌 手动设置的过滤阈值：{area_threshold} 像素")
    else:
        area_threshold = np.median(foreground_areas)
        print(f"❌ 错误：无效的阈值模式，默认使用中位数：{area_threshold:.1f}")

    # 创建过滤后的掩码（初始全黑）
    filtered_mask = np.zeros_like(mask)
    # 遍历所有前景连通域，保留达标区域并填充内部孔洞
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= area_threshold:
            # 提取单个连通域
            single_component = (labels == label).astype(np.uint8) * 255

            # 修复：正确的孔洞填充逻辑
            non_zero_pts = cv2.findNonZero(single_component)
            if non_zero_pts is not None:
                start_pt = tuple(non_zero_pts[0][0])
                filled_component = single_component.copy()
                h, w = single_component.shape
                mask_flood = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(filled_component, mask_flood, start_pt, 255)
                single_component = cv2.bitwise_or(single_component, filled_component)

            # 将填充后的连通域加入最终掩码
            filtered_mask = cv2.bitwise_or(filtered_mask, single_component)

    # 最终轻量闭运算，确保无残留小孔洞
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                     iterations=1)
    # 打印过滤信息
    filtered_num = np.sum(foreground_areas >= area_threshold)
    print(f"📊 连通域过滤：原始{len(foreground_areas)}个 → 保留{filtered_num}个")

# 保存最终掩码
cv2.imwrite(save_mask_path, filtered_mask)
print(f"✅ 最终缺陷掩码已保存到：{save_mask_path}")

# -------------------------- 新增：生成缺陷叠加图（用于保存和展示） --------------------------
img_defect_color = cv2.cvtColor(img_defect, cv2.COLOR_GRAY2BGR)
img_defect_color[filtered_mask == 255] = [0, 0, 255]  # 红色标注缺陷

# -------------------------- 新增：保存所有单独的图像 --------------------------
# 1. 保存有缺陷图
defect_save_path = os.path.join(save_root_dir, "01_defective_image.png")
cv2.imwrite(defect_save_path, img_defect)
print(f"✅ 有缺陷图已保存：{defect_save_path}")

# 2. 保存无缺陷原图
normal_save_path = os.path.join(save_root_dir, "02_normal_image.png")
cv2.imwrite(normal_save_path, img_normal)
print(f"✅ 无缺陷原图已保存：{normal_save_path}")

# 3. 保存原始差分图
diff_original_save_path = os.path.join(save_root_dir, "03_original_difference.png")
cv2.imwrite(diff_original_save_path, diff_original)
print(f"✅ 原始差分图已保存：{diff_original_save_path}")

# 4. 保存预处理后差分图
diff_processed_save_path = os.path.join(save_root_dir, "04_processed_difference.png")
cv2.imwrite(diff_processed_save_path, diff_processed)
print(f"✅ 预处理后差分图已保存：{diff_processed_save_path}")

# 5. 保存最终掩码（复用原有路径，也可自定义）
print(f"✅ 最终掩码已保存：{save_mask_path}")

# 6. 保存缺陷叠加图
overlay_save_path = os.path.join(save_root_dir, "05_defect_overlay.png")
cv2.imwrite(overlay_save_path, img_defect_color)
print(f"✅ 缺陷叠加图已保存：{overlay_save_path}")

# -------------------------- 可视化对比（原有逻辑不变） --------------------------
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.figure(figsize=(24, 5))  # 加宽画布，容纳6个子图

# 子图1：有缺陷图
plt.subplot(1, 6, 1)
plt.imshow(img_defect, cmap="gray")
plt.title("Defective Image", fontsize=10)
plt.axis("off")

# 子图2：无缺陷原图
plt.subplot(1, 6, 2)
plt.imshow(img_normal, cmap="gray")
plt.title("Normal Image", fontsize=10)
plt.axis("off")

# 子图3：原始差分图（预处理前）
plt.subplot(1, 6, 3)
plt.imshow(diff_original, cmap="gray")
plt.title("Original Difference", fontsize=10)
plt.axis("off")

# 子图4：预处理后差分图（阈值前）✅ 新增
plt.subplot(1, 6, 4)
plt.imshow(diff_processed, cmap="gray")
plt.title("Processed Difference (Shrinked)", fontsize=10)
plt.axis("off")

# 子图5：最终掩码
plt.subplot(1, 6, 5)
plt.imshow(filtered_mask, cmap="gray")
plt.title("Final Mask (Optimized)", fontsize=10)
plt.axis("off")

# 子图6：缺陷叠加对比
plt.subplot(1, 6, 6)
plt.imshow(cv2.cvtColor(img_defect_color, cv2.COLOR_BGR2RGB))
plt.title("Defect Overlay (Red)", fontsize=10)
plt.axis("off")

plt.tight_layout()

# -------------------------- 新增：保存拼接后的完整对比图 --------------------------
comparison_save_path = os.path.join(save_root_dir, "06_full_comparison.png")
plt.savefig(comparison_save_path, dpi=150, bbox_inches='tight')
print(f"✅ 拼接后的完整对比图已保存：{comparison_save_path}")

plt.show()

print("🎉 流程执行完成！所有图像已保存至：", save_root_dir)