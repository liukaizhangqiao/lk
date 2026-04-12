import numpy as np
import cv2
import os
import csv
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from glob import glob
import tkinter as tk
from tkinter import filedialog


def select_folder(title="选择文件夹"):
    """交互式选择文件夹"""
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_path if folder_path else None


def select_file(title="选择文件", filetypes=[("CSV文件", "*.csv")]):
    """交互式选择文件"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return file_path if file_path else None


def load_dice_iou_from_csv(csv_path, filename_col="image_name", dice_col="dice_score", iou_col="iou_score"):
    """
    从CSV读取Dice和IoU值，构建{文件名: (dice, iou)}的字典
    :param csv_path: Dice/IoU的CSV文件路径
    :param filename_col: CSV中存储文件名的列名
    :param dice_col: CSV中存储Dice值的列名
    :param iou_col: CSV中存储IoU值的列名
    :return: 映射字典
    """
    dice_iou_dict = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # 检查列名是否存在
            if filename_col not in reader.fieldnames or dice_col not in reader.fieldnames or iou_col not in reader.fieldnames:
                print(f"❌ CSV列名错误！需包含：{filename_col}, {dice_col}, {iou_col}")
                print(f"   当前CSV列名：{reader.fieldnames}")
                return dice_iou_dict

            for row in reader:
                # 提取文件名（去除后缀）和数值
                file_name = row[filename_col].strip()
                # 去除文件名后缀（如111212-1.png → 111212-1）
                if '.' in file_name:
                    file_name = os.path.splitext(file_name)[0]
                # 转换数值（容错处理）
                try:
                    dice = float(row[dice_col].strip())
                    iou = float(row[iou_col].strip())
                    # 限制数值范围在0~1
                    dice = np.clip(dice, 0.0, 1.0)
                    iou = np.clip(iou, 0.0, 1.0)
                    dice_iou_dict[file_name] = (dice, iou)
                except ValueError:
                    print(f"⚠️ {file_name}的Dice/IoU值非数字，设为0.0")
                    dice_iou_dict[file_name] = (0.0, 0.0)
        print(f"✅ 成功读取{len(dice_iou_dict)}条Dice/IoU记录")
    except Exception as e:
        print(f"❌ 读取Dice/IoU CSV失败：{str(e)}")
    return dice_iou_dict


def load_mask_from_file(file_path, threshold=127):
    """读取掩码并转为二值化数组"""
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取文件: {file_path}")
    return (mask > threshold).astype(np.uint8)


def count_skeleton_pixels(skeleton_mask):
    """统计骨架像素数（纵向延伸长度）"""
    return np.sum(skeleton_mask == 1)


def count_binary_region_pixels(binary_mask):
    """统计二值化区域总像素数"""
    return np.sum(binary_mask == 1)


def calculate_thin_target_width(binary_pixel_count, skeleton_pixel_count):
    """计算细薄目标平均横向宽度：二值总像素 / 骨架像素"""
    if skeleton_pixel_count == 0:
        return 0.0
    return round(binary_pixel_count / skeleton_pixel_count, 4)


def extract_skeleton_points(mask):
    """提取骨架点坐标"""
    y_coords, x_coords = np.nonzero(mask)
    return np.column_stack((x_coords, y_coords))


# -------------------------- 论文LineAcc指标核心函数 --------------------------
def compute_euclidean_distance(mask):
    """计算欧式距离（L2）- 论文pos指标核心"""
    if np.sum(mask) == 0:
        return np.full(mask.shape, np.inf)
    return distance_transform_edt(1 - mask)


def compute_lineacc_pos(true_skeleton_mask, pred_skeleton_mask, sigma=1.0, eps=1e-3):
    """计算论文LineAcc-pos（位置匹配度）- 欧式距离版"""
    # 双向欧式距离图
    dist_pred_to_gt = compute_euclidean_distance(true_skeleton_mask)
    dist_gt_to_pred = compute_euclidean_distance(pred_skeleton_mask)

    # 提取骨架像素并高斯加权
    pred_skeleton = pred_skeleton_mask
    gt_skeleton = true_skeleton_mask
    pred_weight = pred_skeleton * np.exp(-np.square(dist_pred_to_gt) / (2 * sigma ** 2))
    gt_weight = gt_skeleton * np.exp(-np.square(dist_gt_to_pred) / (2 * sigma ** 2))

    # 双向归一化
    pred_sum = np.sum(pred_skeleton) + eps
    gt_sum = np.sum(gt_skeleton) + eps
    term1 = np.sum(pred_weight) / pred_sum
    term2 = np.sum(gt_weight) / gt_sum

    return round(term1 * term2, 4)


def compute_lineacc_length(true_skeleton_mask, pred_skeleton_mask, eps=1e-3):
    """计算论文LineAcc-length（长度匹配度）"""
    true_len = count_skeleton_pixels(true_skeleton_mask) + eps
    pred_len = count_skeleton_pixels(pred_skeleton_mask) + eps
    length_ratio = pred_len / true_len
    return round(np.exp(-np.abs(length_ratio - 1)), 4)


def compute_lineacc_width(true_binary_mask, pred_binary_mask, true_skeleton_mask, pred_skeleton_mask, eps=1e-3):
    """计算论文LineAcc-width（宽度匹配度）"""
    sum_true = count_binary_region_pixels(true_binary_mask) + eps
    sum_pred = count_binary_region_pixels(pred_binary_mask) + eps
    sum_true_skel = count_skeleton_pixels(true_skeleton_mask) + eps
    sum_pred_skel = count_skeleton_pixels(pred_skeleton_mask) + eps

    # 平均宽度比值 = (真实总像素/真实骨架) / (预测总像素/预测骨架)
    width_ratio = (sum_true / sum_true_skel) / (sum_pred / sum_pred_skel)
    return round(np.exp(-np.abs(width_ratio - 1)), 4)


def compute_lineacc_combined(pos, width, length, dice_score, iou_score, normalize=True):
    """计算论文LineAcc-combined（综合评分）- 加权融合"""
    # 论文权重：pos=2, width/length/dice/iou=0.5
    combined = 2 * pos + 0.5 * width + 0.5 * length + 0.5 * dice_score + 0.5 * iou_score
    if normalize:
        # 方案1：除以理论最大值4（推荐，和论文一致）
        combined = combined / 4.0
        # 方案2：如果要和论文完全对齐，可改为除以实际最大值（需统计所有样本后再归一化）
    return round(combined, 4)

# -------------------------- 原有骨架匹配评分（保留兼容） --------------------------
def gaussian_weight(distance, sigma=5.0):
    """高斯衰减加权"""
    return np.exp(-(distance ** 2) / (2 * sigma ** 2))


def skeleton_matching_score(true_skeleton_mask, pred_skeleton_mask, sigma=5.0):
    """计算双向距离+高斯加权的匹配评分（原有逻辑，保留）"""
    true_pts = extract_skeleton_points(true_skeleton_mask)
    pred_pts = extract_skeleton_points(pred_skeleton_mask)

    true_skeleton_pix = count_skeleton_pixels(true_skeleton_mask)
    pred_skeleton_pix = count_skeleton_pixels(pred_skeleton_mask)
    if true_skeleton_pix == 0 and pred_skeleton_pix == 0:
        return 1.0, {"msg": "双骨架为空"}
    if true_skeleton_pix == 0 or pred_skeleton_pix == 0:
        return 0.0, {"msg": "单骨架为空"}

    true_tree = KDTree(true_pts)
    pred_tree = KDTree(pred_pts)
    pred2true_dist = true_tree.query(pred_pts)[0].mean()
    true2pred_dist = pred_tree.query(true_pts)[0].mean()
    bidir_dist = (pred2true_dist + true2pred_dist) / 2.0

    h, w = true_skeleton_mask.shape
    normalize_base = np.sqrt(h ** 2 + w ** 2)
    gauss_w = gaussian_weight(bidir_dist, sigma)
    norm_raw = np.clip(bidir_dist / normalize_base, 0.0, 1.0)
    final_score = np.clip(1 - norm_raw * (1 - gauss_w), 0.0, 1.0)

    return round(final_score, 4), {
        "pred2true_dist": round(pred2true_dist, 4),
        "true2pred_dist": round(true2pred_dist, 4),
        "bidir_dist": round(bidir_dist, 4),
        "gauss_w": round(gauss_w, 4)
    }


def batch_evaluate_thin_target_metrics():
    # 1. 先选择Dice/IoU的CSV文件
    print("📌 请选择【Dice/IoU数据】CSV文件")
    dice_iou_csv = select_file("Dice/IoU CSV文件")
    if not dice_iou_csv:
        print("❌ 未选择Dice/IoU CSV文件，程序退出")
        return
    # 读取Dice/IoU字典（适配你的CSV列名：image_name, dice_score, iou_score）
    dice_iou_dict = load_dice_iou_from_csv(
        csv_path=dice_iou_csv,
        filename_col="image_name",
        dice_col="dice_score",
        iou_col="iou_score"
    )
    if not dice_iou_dict:
        print("❌ 未读取到有效Dice/IoU数据，程序退出")
        return

    # 2. 选择各掩码文件夹
    print("\n📌 请选择【真实二值化掩码】文件夹（目标整体区域，无_binary）")
    true_binary_dir = select_folder("真实二值化掩码文件夹")
    if not true_binary_dir:
        print("❌ 未选择文件夹，程序退出")
        return

    print("\n📌 请选择【预测二值化掩码】文件夹（目标整体区域，有_binary）")
    pred_binary_dir = select_folder("预测二值化掩码文件夹")
    if not pred_binary_dir:
        print("❌ 未选择文件夹，程序退出")
        return

    print("\n📌 请选择【真实骨架掩码】文件夹（无_binary后缀）")
    true_skeleton_dir = select_folder("真实骨架掩码文件夹")
    if not true_skeleton_dir:
        print("❌ 未选择文件夹，程序退出")
        return

    print("\n📌 请选择【预测骨架掩码】文件夹（有_binary后缀）")
    pred_skeleton_dir = select_folder("预测骨架掩码文件夹")
    if not pred_skeleton_dir:
        print("❌ 未选择文件夹，程序退出")
        return

    IMG_EXT = "png"
    GAUSS_SIGMA = 5.0
    LINEACC_SIGMA = 5.0  # 论文pos指标的高斯sigma
    SAVE_CSV_PATH = "thin_target_evaluation_results_with_lineacc.csv"

    true_binary_pattern = os.path.join(true_binary_dir, f"*.{IMG_EXT}")
    true_binary_files = glob(true_binary_pattern)
    if not true_binary_files:
        print(f"❌ 在{true_binary_dir}中未找到{IMG_EXT}格式文件")
        return

    results = []
    print("\n🚀 开始评估细薄目标指标（含论文LineAcc完整指标）...")
    for true_binary_f in true_binary_files:
        true_binary_fname = os.path.basename(true_binary_f)
        true_base, _ = os.path.splitext(true_binary_fname)

        # 拼接文件路径（保留原有_binary后缀逻辑）
        true_skeleton_f = os.path.join(true_skeleton_dir, f"{true_base}.{IMG_EXT}")
        pred_binary_f = os.path.join(pred_binary_dir, f"{true_base}_binary.{IMG_EXT}")
        pred_skeleton_f = os.path.join(pred_skeleton_dir, f"{true_base}_binary.{IMG_EXT}")

        missing_files = []
        if not os.path.exists(true_skeleton_f):
            missing_files.append(f"真实骨架文件: {true_skeleton_f}")
        if not os.path.exists(pred_binary_f):
            missing_files.append(f"预测二值文件: {pred_binary_f}")
        if not os.path.exists(pred_skeleton_f):
            missing_files.append(f"预测骨架文件: {pred_skeleton_f}")
        if missing_files:
            print(f"⚠️ {true_binary_fname} 缺失文件：{', '.join(missing_files)}")
            continue

        try:
            # 加载掩码
            true_binary_mask = load_mask_from_file(true_binary_f)
            true_skeleton_mask = load_mask_from_file(true_skeleton_f)
            pred_binary_mask = load_mask_from_file(pred_binary_f)
            pred_skeleton_mask = load_mask_from_file(pred_skeleton_f)

            # 原有指标计算
            true_skeleton_pix = count_skeleton_pixels(true_skeleton_mask)
            pred_skeleton_pix = count_skeleton_pixels(pred_skeleton_mask)
            true_binary_pix = count_binary_region_pixels(true_binary_mask)
            pred_binary_pix = count_binary_region_pixels(pred_binary_mask)
            true_target_width = calculate_thin_target_width(true_binary_pix, true_skeleton_pix)
            pred_target_width = calculate_thin_target_width(pred_binary_pix, pred_skeleton_pix)
            match_score, dist_details = skeleton_matching_score(true_skeleton_mask, pred_skeleton_mask, GAUSS_SIGMA)

            # 论文LineAcc指标计算
            # 1. 从字典自动获取Dice和IoU
            if true_base in dice_iou_dict:
                dice_score, iou_score = dice_iou_dict[true_base]
                print(f"🔍 {true_base} - 自动读取Dice: {dice_score:.4f}, IoU: {iou_score:.4f}")
            else:
                print(f"⚠️ {true_base} 未找到对应的Dice/IoU值，设为0.0")
                dice_score, iou_score = 0.0, 0.0

            # 2. 计算LineAcc各子指标
            lineacc_pos = compute_lineacc_pos(true_skeleton_mask, pred_skeleton_mask, LINEACC_SIGMA)
            lineacc_length = compute_lineacc_length(true_skeleton_mask, pred_skeleton_mask)
            lineacc_width = compute_lineacc_width(true_binary_mask, pred_binary_mask, true_skeleton_mask,
                                                  pred_skeleton_mask)
            lineacc_combined = compute_lineacc_combined(lineacc_pos, lineacc_width, lineacc_length, dice_score,
                                                        iou_score)

            # 整合所有结果（原有+论文指标）
            single_result = {
                # 基础信息
                "文件名": true_base,
                # 原有指标
                "真实二值区域总像素数": true_binary_pix,
                "真实骨架像素数（纵向延伸长度）": true_skeleton_pix,
                "真实目标平均横向宽度": true_target_width,
                "预测二值区域总像素数": pred_binary_pix,
                "预测骨架像素数（纵向延伸长度）": pred_skeleton_pix,
                "预测目标平均横向宽度": pred_target_width,
                "宽度绝对偏差": round(abs(true_target_width - pred_target_width), 4),
                "宽度相对偏差(%)": round(
                    abs(true_target_width - pred_target_width) / true_target_width * 100 if true_target_width != 0 else 0,
                    2),
                "预测→真实平均距离(像素)": dist_details["pred2true_dist"],
                "真实→预测平均距离(像素)": dist_details["true2pred_dist"],
                "双向平均距离(像素)": dist_details["bidir_dist"],
                "高斯权重": dist_details["gauss_w"],
                "原有骨架匹配评分(0~1)": match_score,
                # 论文LineAcc指标（核心新增）
                "LineAcc-pos（位置匹配度）": lineacc_pos,
                "LineAcc-width（宽度匹配度）": lineacc_width,
                "LineAcc-length（长度匹配度）": lineacc_length,
                "输入Dice值": round(dice_score, 4),
                "输入IoU值": round(iou_score, 4),
                "LineAcc-combined（综合评分）": lineacc_combined,
                # 备注
                "备注": dist_details.get("msg", "正常评估")
            }
            results.append(single_result)

            # 打印进度
            print(
                f"✅ {true_base} | "
                f"LineAcc-pos：{lineacc_pos} | "
                f"LineAcc-width：{lineacc_width} | "
                f"LineAcc-length：{lineacc_length} | "
                f"LineAcc-combined：{lineacc_combined}"
            )

        except Exception as e:
            print(f"❌ 处理{true_base}失败：{str(e)}")

    # 保存结果
    if results:
        with open(SAVE_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n📁 评估结果已保存至：{os.path.abspath(SAVE_CSV_PATH)}")

        # 计算汇总指标
        total_files = len(results)
        # 原有指标汇总
        avg_true_skeleton_pix = np.mean([r["真实骨架像素数（纵向延伸长度）"] for r in results])
        avg_true_target_width = np.mean([r["真实目标平均横向宽度"] for r in results])
        avg_pred_skeleton_pix = np.mean([r["预测骨架像素数（纵向延伸长度）"] for r in results])
        avg_pred_target_width = np.mean([r["预测目标平均横向宽度"] for r in results])
        avg_width_abs_error = np.mean([r["宽度绝对偏差"] for r in results])
        avg_width_rel_error = np.mean([r["宽度相对偏差(%)"] for r in results])
        avg_match_score = np.mean([r["原有骨架匹配评分(0~1)"] for r in results])
        avg_bidir_dist = np.mean([r["双向平均距离(像素)"] for r in results])
        # 论文LineAcc指标汇总
        avg_lineacc_pos = np.mean([r["LineAcc-pos（位置匹配度）"] for r in results])
        avg_lineacc_width = np.mean([r["LineAcc-width（宽度匹配度）"] for r in results])
        avg_lineacc_length = np.mean([r["LineAcc-length（长度匹配度）"] for r in results])
        avg_lineacc_combined = np.mean([r["LineAcc-combined（综合评分）"] for r in results])

        # 打印汇总
        print("\n" + "=" * 150)
        print(f"📊 细薄目标核心指标汇总（含论文LineAcc完整指标）")
        print(f"有效评估文件数：{total_files}")
        print(f"├─ 原有指标汇总：")
        print(
            f"│  ├─ 纵向延伸长度（骨架像素数）：真实均值 {avg_true_skeleton_pix:.2f} | 预测均值 {avg_pred_skeleton_pix:.2f}")
        print(f"│  ├─ 平均横向宽度：真实均值 {avg_true_target_width:.4f} | 预测均值 {avg_pred_target_width:.4f}")
        print(f"│  ├─ 宽度绝对偏差均值 {avg_width_abs_error:.4f} | 相对偏差均值 {avg_width_rel_error:.2f}%")
        print(f"│  ├─ 原有骨架匹配评分均值 {avg_match_score:.4f} | 双向平均距离均值 {avg_bidir_dist:.2f} 像素")
        print(f"├─ 论文LineAcc指标汇总：")
        print(f"│  ├─ LineAcc-pos（位置）均值：{avg_lineacc_pos:.4f}")
        print(f"│  ├─ LineAcc-width（宽度）均值：{avg_lineacc_width:.4f}")
        print(f"│  ├─ LineAcc-length（长度）均值：{avg_lineacc_length:.4f}")
        print(f"│  └─ LineAcc-combined（综合）均值：{avg_lineacc_combined:.4f}")
        print("=" * 150)
    else:
        print("\n⚠️  未完成任何文件的评估！")


if __name__ == "__main__":
    print("🎯 细薄目标核心指标评估工具（集成论文LineAcc完整指标）")
    print("🔍 核心逻辑：")
    print("   - 保留原有纵向长度/横向宽度/骨架匹配评分逻辑")
    print("   - 新增论文LineAcc指标：pos(位置)/width(宽度)/length(长度)/combined(综合)")
    print("   - LineAcc-combined加权规则：pos×2 + width×0.5 + length×0.5 + Dice×0.5 + IoU×0.5")
    print("   - 欧式距离计算pos指标，适配图像旋转场景")
    print("   - 预测文件需带_binary后缀，真实文件无后缀")
    print("   - 自动从CSV读取Dice/IoU值，无需手动输入")
    print("-" * 80)
    batch_evaluate_thin_target_metrics()