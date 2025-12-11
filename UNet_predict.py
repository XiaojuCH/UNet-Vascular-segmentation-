import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as T
import cv2
from Unet_train import UNet   # 如果你的 UNet 在 Unet_test.py 里

# -------------------------
# 1. 加载模型
# -------------------------
def load_model(model_path="save_model/checkpoints/run_best.pth", device="cuda"):
    model = UNet(n_classes=2).to(device)
    # 加载检查点（包含net, epoch, optimizer等）
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # 只加载模型权重
    model.load_state_dict(checkpoint['net'])
    model.eval()
    return model


# -------------------------
# 2. 对单张图做推理
# -------------------------
def predict_single(model, img_path, device="cuda"):
    # 打开图片并转 tensor
    img = Image.open(img_path).convert("RGB")
    tf = T.ToTensor()
    img_tensor = tf(img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        out = model(img_tensor)
        pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()

    return pred  # 返回 mask（0/1）矩阵


# -------------------------
# 3. 可视化 + 保存结果
# -------------------------
def save_mask(mask, save_path):
    # mask 是 0/1，转成 0/255 才能显示
    mask_img = (mask * 255).astype(np.uint8)
    cv2.imwrite(save_path, mask_img)
    print("预测 mask 已保存：", save_path)

def save_color_mask(mask, save_path):
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    color[mask == 1] = [0, 0, 200]   # 前景红色
    color[mask == 0] = [0, 0, 0]     # 背景黑色

    cv2.imwrite(save_path, color)


# -------------------------
# 4. 主函数
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model("save_model/checkpoints/run_best.pth", device)

    img_path = "1.png"
    save_path = "pred_mask.png"

    mask = predict_single(model, img_path, device)
    save_color_mask(mask, img_path+save_path)
    
    print("成功将文件%s的预测结果保存到%s" % (img_path, img_path+save_path))
