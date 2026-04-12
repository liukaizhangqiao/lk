
import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ===================== 配置 =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
input_image_reshape = (512, 512)

# ===================== 配置参数（和训练时保持一致）=====================
device = "cuda" if torch.cuda.is_available() else "cpu"
input_image_reshape = (512, 512)
model_path = "DZPCS_U-Net_cracktree200.pth"  # 模型权重路径
test_image_path = "cracktree200_test_image/6192.png"           # 要测试的图像路径
save_result_path = "cracktree200_test_image/6192_pre.png"              # 输出结果路径
# =====================================================================
# 模型结构和训练完全一致
class PavementModel(torch.nn.Module):
    def __init__(self, arch, encoder_name, in_channels=3, out_classes=1, **kwargs):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes,** kwargs
        )

    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

# 加载模型
def load_model():
    model = PavementModel("Unet", "resnet34", in_channels=3, out_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logging.info("模型加载成功")
    return model

# 预测黑白掩码
def predict_image(model, img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # 预处理
    img_in = cv2.resize(img_rgb, input_image_reshape)
    img_tensor = torch.from_numpy(img_in).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_tensor)
        prob = out.sigmoid().squeeze().cpu().numpy()
        pred_mask = (prob > 0.5).astype(np.uint8) * 255  # 黑白掩码

    pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return img, pred_mask

# 拼接：原图 + 黑白掩码
def save_combined(original_img, mask, save_path):
    # 掩码转3通道方便拼接
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # 左右拼接
    combined = np.hstack((original_img, mask_3ch))
    cv2.imwrite(save_path, combined)
    logging.info(f"拼接图已保存: {save_path}")

if __name__ == "__main__":
    model = load_model()
    original_img, pred_mask = predict_image(model, test_image_path)
    save_combined(original_img, pred_mask, save_result_path)