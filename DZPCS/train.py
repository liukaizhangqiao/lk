import logging
import os
import sys
import csv  # 新增：导入csv模块
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm

# 新增：导入数据扩充库
import albumentations as A
import segmentation_models_pytorch as smp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d:%m:%Y %H:%M:%S",
)

# ----------------------------
# Set the device to GPU if available
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")
if device == "cpu":
    os.system("export OMP_NUM_THREADS=64")
    torch.set_num_threads(os.cpu_count())

# ----------------------------
# Download the CamVid dataset, if needed
# ----------------------------
# Change this to your desired directory
main_dir = "/home/user/Desktop/alldata/"

# --由llm生成的数据集
# data_dir = os.path.join(main_dir, "llm_deepcrack")
data_dir = os.path.join(main_dir, "llm_crack500")
# data_dir = os.path.join(main_dir, "llm_cracktree200_9")


if not os.path.exists(data_dir):
    logging.info("Loading data...")
    os.system(f"git clone https://github.com/alexgkendall/SegNet-Tutorial {data_dir}")
    logging.info("Done!")

# Create a directory to store the output masks
output_dir = os.path.join(data_dir, "pavement_output_images")
os.makedirs(output_dir, exist_ok=True)

# 新增：创建单独保存预测掩码的文件夹
pred_mask_dir = os.path.join(output_dir, "predicted_masks")
os.makedirs(pred_mask_dir, exist_ok=True)

# ----------------------------
# Define the hyperparameters
# ----------------------------
epochs_max = 200  # Number of epochs to train the model
adam_lr = 1e-4  # 学习率
eta_min = 1e-5  # 最低学习率
batch_size = 4  # Batch size for training
input_image_reshape = (512, 512)  # Desired shape for the input images and masks
foreground_class = 255  # 二分类中设置为前景的像素

# 数据增强
train_augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
    A.SmallestMaxSize(max_size=512, p=0.5),  # 轻微调整尺寸，增加多样性
])

# ----------------------------
# 新增：单独保存掩码的函数
# ----------------------------
def save_predicted_mask(mask_array, save_path, threshold=0.5):
    """
    保存预测掩码为灰度图（0=背景，255=前景）
    :param mask_array: 模型输出的掩码数组（numpy格式）
    :param save_path: 保存路径
    :param threshold: 二值化阈值
    """
    # 二值化：大于阈值设为255（前景），否则0（背景）
    binary_mask = (mask_array > threshold).astype(np.uint8) * 255
    # 保存为灰度图
    cv2.imwrite(save_path, binary_mask)

# ----------------------------
# Define a custom dataset class for the pavement dataset
# ----------------------------
class Dataset(BaseDataset):

    def __init__(
            self,
            images_dir,
            masks_dir,
            input_image_reshape=(512, 512),
            foreground_class=255,
            augmentation=None,
    ):
        self.ids = os.listdir(images_dir)  # 到底指定数据集文件夹位置
        self.images_filepaths = [os.path.join(images_dir, image_id) for image_id in self.ids]  # 到达指定数据集文件夹，再拼接每一个图像文件地址
        self.masks_filepaths = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.input_image_reshape = input_image_reshape  # 保持默认属性
        self.foreground_class = foreground_class
        self.augmentation = augmentation

    def __getitem__(self, i):
        # Read the image
        image = cv2.imread(self.images_filepaths[i],
                           cv2.IMREAD_GRAYSCALE)  # 变量 image 现在存储了一个 NumPy 数组，它的形状是 (高度, 宽度)，以灰度形式
        image = np.expand_dims(image, axis=-1)  # 我们也通常把它处理成单通道的形式 (H, W, 1)，而不是 (H, W)，这样可以保持数据维度的一致性。
        image = cv2.resize(image, self.input_image_reshape)  # 将图像缩放到目标尺寸
        mask = cv2.imread(self.masks_filepaths[i],
                          0)  # 读取单张掩码（灰度图），并进行二值化映射（前景类→255，其他→0）， 0: 这个参数和 cv2.IMREAD_GRAYSCALE 是等价的
        # 检查 cv2.imread 是否成功读取了掩码文件。如果文件路径错误或文件损坏，imread 会返回 None
        if mask is None:
            raise ValueError(f"无法读取掩码: {self.masks_filepaths[i]}")

        # 使用 NumPy 的 where 函数对掩码进行二值化处理
        mask_remap = np.where(mask == self.foreground_class, 1, 0).astype(np.uint8)
        # 将处理后的掩码也缩放到目标尺寸
        mask_remap = cv2.resize(mask_remap, self.input_image_reshape, interpolation=cv2.INTER_NEAREST)

        # 应用数据增强（仅训练集）
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)  # 如果存在数据增强策略，则对图像和掩码同时执行增强操作。
            image, mask_remap = sample["image"], sample["mask"]

        # 安全检查：确保图像是 (H, W, C) 形状
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        # 转换为PyTorch张量并归一化
        image = torch.tensor(image).float().permute(2, 0, 1) / 255.0
        # 将处理好的 NumPy 掩码数组转换为 PyTorch 张量
        mask_remap = torch.tensor(mask_remap).long()

        # 返回图像、掩码，同时返回图像文件名（新增：用于CSV记录）
        return image, mask_remap, self.ids[i]

    def __len__(self):
        return len(self.ids)

# Define a class for the pavement model
class PavementModel(torch.nn.Module):
    def __init__(self, arch, encoder_name, in_channels=3, out_classes=1, **kwargs):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes, **kwargs,
        )

    def forward(self, image):
        # 对输入图像进行标准化处理
        image = (image - self.mean) / self.std
        # 返回模型推理得到的掩码张量
        mask = self.model(image)
        return mask

# 可视化
def visualize(output_dir, image_filename, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image, cmap='gray')
    plt.savefig(os.path.join(output_dir, image_filename))
    plt.close()

# 可视化数据增强
def visualize_augmented_samples(dataset, output_dir, num_samples=5, num_versions=5):
    """可视化数据增强后的样本，同时显示原始图像（不增强）用于对比"""
    aug_output_dir = os.path.join(output_dir, "augmented_samples_with_original")
    os.makedirs(aug_output_dir, exist_ok=True)

    for i in range(num_samples):
        # 1. 获取原始图像（关闭增强）
        original_aug = dataset.augmentation
        dataset.augmentation = None  # 不应用任何增强
        orig_image, orig_mask, _ = dataset[i]  # 新增：接收文件名（但此处不用）
        dataset.augmentation = original_aug  # 恢复增强策略

        # 原始图像格式转换（用于可视化）
        orig_img_np = orig_image.numpy().transpose(1, 2, 0).squeeze()  # 转为 (H, W)
        orig_mask_np = orig_mask.squeeze().numpy()  # 原始掩码 (H, W)

        # 2. 获取增强后的图像，对比显示
        for version in range(num_versions):
            # 读取增强后的图像和掩码（使用原增强策略）
            aug_image, aug_mask, _ = dataset[i]
            # 增强图像格式转换
            aug_img_np = aug_image.numpy().transpose(1, 2, 0).squeeze()  # (H, W)
            aug_mask_np = aug_mask.squeeze().numpy()  # 增强掩码 (H, W)

            # 保存对比图：原始（图像+掩码） vs 增强（图像+掩码）
            visualize(
                aug_output_dir,
                f"aug_sample_{i + 1}_version_{version + 1}_with_original.png",
                original_image=orig_img_np,
                original_mask=orig_mask_np,
                augmented_image=aug_img_np,
                augmented_mask=aug_mask_np
            )
    logging.info(f"含原始图像的增强对比图已保存到：{aug_output_dir}")

# Use multiple CPUs in parallel
def train_and_evaluate_one_epoch(
        model, train_dataloader, valid_dataloader, optimizer, scheduler, loss_fn, device
):
    # Set the model to training mode
    model.train()
    train_loss = 0
    # 每次读取一个批次的图像和对应掩码
    for batch in tqdm(train_dataloader, desc="Training", file=sys.stdout):
        images, masks, _ = batch  # 新增：接收文件名（训练时不用）
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    # 学习率自动调优器
    scheduler.step()
    avg_train_loss = train_loss / len(train_dataloader)

    # Set the model to evaluation mode
    model.eval()
    val_loss = 0
    with torch.inference_mode():
        for batch in tqdm(valid_dataloader, desc="Evaluating", file=sys.stdout):
            images, masks, _ = batch  # 新增：接收文件名（验证时不用）
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(valid_dataloader)
    return avg_train_loss, avg_val_loss

def train_model(
        model,
        train_dataloader,
        valid_dataloader,
        optimizer,
        scheduler,
        loss_fn,
        device,
        epochs,
):
    train_losses = []
    val_losses = []
    best_val_loss = np.inf  # 初始设为无穷大
    best_model_path = "DZPCS_U-Net_crack500.pth"  # 直接保存在当前文件夹

    for epoch in range(epochs):
        avg_train_loss, avg_val_loss = train_and_evaluate_one_epoch(
            model,
            train_dataloader,
            valid_dataloader,
            optimizer,
            scheduler,
            loss_fn,
            device,
        )
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        logging.info(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
        )

        # ===================== 核心：保存最好的模型 =====================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"✅ 找到更好的模型！已保存到当前文件夹：{best_model_path}")
        # ===============================================================

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    return history

def evaluate_model(model, output_dir, pred_mask_dir, test_dataloader, loss_fn, device):
    # Set the model to evaluation mode
    model.eval()
    test_loss = 0
    tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0

    # 新增：初始化CSV文件
    csv_filepath = os.path.join(output_dir, "per_image_metrics.csv")
    # 写入CSV表头
    with open(csv_filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "dice_score", "iou_score"])

    # 新增：存储单张图像的指标
    per_image_metrics = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating", file=sys.stdout)):
            images, masks, image_names = batch  # 新增：接收图像文件名
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, masks)
            test_loss += loss.item()

            # 计算预测掩码
            prob_mask = outputs.sigmoid().squeeze(1)
            pred_mask = (prob_mask > 0.5).long()

            # 遍历当前批次的每张图像
            for idx in range(len(images)):
                # 单张图像的真实掩码和预测掩码
                single_true_mask = masks[idx:idx + 1]  # 保持维度 (1, H, W)
                single_pred_mask = pred_mask[idx:idx + 1]

                # 计算单张图像的TP/FP/FN/TN
                tp, fp, fn, tn = smp.metrics.get_stats(
                    single_pred_mask, single_true_mask, mode="binary"
                )

                # 计算单张图像的Dice和IoU
                dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()

                # 存储指标
                image_name = image_names[idx]
                per_image_metrics.append({
                    "image_name": image_name,
                    "dice_score": round(dice, 4),
                    "iou_score": round(iou, 4)
                })

                # 可视化单张图像（保留原有逻辑）
                input_img = images[idx].cpu().numpy().transpose(1, 2, 0)
                output_mask = outputs[idx].squeeze().cpu().numpy()
                true_mask = masks[idx].cpu().numpy().squeeze()

                visualize(
                    output_dir,
                    f"output_{batch_idx}_{idx}.png",  # 修改命名避免重复
                    input_image=input_img,
                    output_mask=output_mask,
                    binary_mask=output_mask > 0.5,
                    true_mask=true_mask,
                )

                # ----------------------------
                # 核心修改：调整掩码文件名格式为「原始图像名_binary.后缀」
                # ----------------------------
                # 拆分文件名和后缀（处理如 001.png、crack_02.jpg 等格式）
                img_base, img_ext = os.path.splitext(image_name)
                # 构造新文件名：原始名称 + _binary + 原后缀
                mask_save_name = f"{img_base}_binary{img_ext}"
                mask_save_path = os.path.join(pred_mask_dir, mask_save_name)
                # 保存预测掩码
                save_predicted_mask(output_mask, mask_save_path, threshold=0.5)

                # 累加全局指标
                tp_total += tp.sum().item()
                fp_total += fp.sum().item()
                fn_total += fn.sum().item()
                tn_total += tn.sum().item()

    # 新增：将单张图像的指标写入CSV
    with open(csv_filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for metric in per_image_metrics:
            writer.writerow([metric["image_name"], metric["dice_score"], metric["iou_score"]])
    logging.info(f"单张图像的Dice/IoU已保存到: {csv_filepath}")
    logging.info(f"预测掩码已保存到: {pred_mask_dir}")

    # 计算全局指标
    test_loss_mean = test_loss / len(test_dataloader)
    tp_t = torch.tensor([tp_total])
    fp_t = torch.tensor([fp_total])
    fn_t = torch.tensor([fn_total])
    tn_t = torch.tensor([tn_total])

    precision = smp.metrics.precision(tp_t, fp_t, fn_t, tn_t, reduction="micro").item()
    recall = smp.metrics.recall(tp_t, fp_t, fn_t, tn_t, reduction="micro").item()
    f1 = smp.metrics.f1_score(tp_t, fp_t, fn_t, tn_t, reduction="micro").item()
    iou_score = smp.metrics.iou_score(tp_t, fp_t, fn_t, tn_t, reduction="micro").item()

    # 输出日志
    logging.info(f"Test Loss: {test_loss_mean:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-score: {f1:.4f}")
    logging.info(f"IoU Score: {iou_score:.4f}")

    return test_loss_mean, iou_score, precision, recall, f1

# ----------------------------
# Define the data directories and create the datasets
# ----------------------------
x_train_dir = os.path.join(data_dir, "pavement", "train")
y_train_dir = os.path.join(data_dir, "pavement", "trainannot")

x_val_dir = os.path.join(data_dir, "pavement", "val")
y_val_dir = os.path.join(data_dir, "pavement", "valannot")

x_test_dir = os.path.join(data_dir, "pavement", "test")
y_test_dir = os.path.join(data_dir, "pavement", "testannot")

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    input_image_reshape=input_image_reshape,
    foreground_class=foreground_class,
    augmentation=train_augmentation,
)
valid_dataset = Dataset(
    x_val_dir,
    y_val_dir,
    input_image_reshape=input_image_reshape,
    foreground_class=foreground_class,
)
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    input_image_reshape=input_image_reshape,
    foreground_class=foreground_class,
)

# 修正：读取数据集时接收文件名
image, mask, img_name = train_dataset[0]
logging.info(f"Image name: {img_name}")
logging.info(f"Unique values in mask: {np.unique(mask)}")
logging.info(f"Image shape: {image.shape}")
logging.info(f"Mask shape: {mask.shape}")

# ----------------------------
# Create the dataloaders using the datasets
# ----------------------------
logging.info(f"Train size: {len(train_dataset)}")
logging.info(f"Valid size: {len(valid_dataset)}")
logging.info(f"Test size: {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------
# 调用数据增强可视化函数
# ----------------------------
visualize_augmented_samples(train_dataset, output_dir, num_samples=5, num_versions=5)

# ----------------------------
# Lets look at some samples
# ----------------------------
# Visualize and save train sample
sample = train_dataset[0]
visualize(
    output_dir,
    "train_sample.png",
    train_image=sample[0].numpy().transpose(1, 2, 0),
    train_mask=sample[1].squeeze(),
)

# Visualize and save validation sample
sample = valid_dataset[0]
visualize(
    output_dir,
    "validation_sample.png",
    validation_image=sample[0].numpy().transpose(1, 2, 0),
    validation_mask=sample[1].squeeze(),
)

# Visualize and save test sample
sample = test_dataset[0]
visualize(
    output_dir,
    "test_sample.png",
    test_image=sample[0].numpy().transpose(1, 2, 0),
    test_mask=sample[1].squeeze(),
)

# ----------------------------
# Create and train the model
# ----------------------------
max_iter = epochs_max * len(train_dataloader)  # Total number of iterations

# 1---resnet34+unet
model = PavementModel("Unet", "resnet34", in_channels=3, out_classes=1)

# 2---resnet50+unet
# model = PavementModel("Unet", "resnet50", in_channels=3, out_classes=1)

# 3---resnet34+unet++
# model = PavementModel("unetplusplus", "resnet34", in_channels=3, out_classes=1)

# 4---resnet50+unet++
# model = PavementModel("unetplusplus", "resnet50", in_channels=3, out_classes=1)

# 5---resnet34+FPN
# model = PavementModel("FPN", "resnet34", in_channels=3, out_classes=1)

# 6---resnet34+deeplabv3
# model = PavementModel("deeplabv3", "resnet34", in_channels=3, out_classes=1)

# 7---EfficientNet-b3+FPN
# model = PavementModel("FPN", "efficientnet-b3", in_channels=3, out_classes=1)
# print(model)

# Training loop
model = model.to(device)

# Define the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

# Define the learning rate scheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=eta_min)

# Define the loss function
# 6 种损失
loss_fn = smp.losses.FocalLoss(smp.losses.BINARY_MODE)

# loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

# Train the model
history = train_model(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epochs_max,
)


# 半对数坐标绘制
plt.figure(figsize=(10, 5))
plt.plot(history["train_losses"], label="Train Loss", marker='o', markersize=3)  # 加标记点更清晰
plt.plot(history["val_losses"], label="Validation Loss", marker='s', markersize=3)
plt.xlabel("Epochs (Linear Scale)")  # x轴保持线性（轮次是均匀变化的）
plt.ylabel("Loss (Log Scale)")  # y轴对数刻度（损失值跨度大）
plt.yscale('log')  # 关键：将y轴设为对数刻度
plt.title("Training and Validation Losses (Semilog Plot)")
plt.legend()
plt.grid(True, which="both", ls="--")  # 加网格线，方便读取数值

# Evaluate the model（修改：传入pred_mask_dir参数）
test_loss, iou, precision, recall, f1 = evaluate_model(model, output_dir, pred_mask_dir, test_dataloader, loss_fn, device)

plt.text(
    0.8, 0.8,
    f"Test Loss: {test_loss:.2f}\nIoU: {iou:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1: {f1:.2f}",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor='white', alpha=0.8),
    verticalalignment='top'
)

plt.savefig(os.path.join(output_dir, "loss_semilog.png"))
plt.close()

logging.info(f"The output masks are saved in {output_dir}.")
logging.info(f"The predicted masks are saved in {pred_mask_dir}.")