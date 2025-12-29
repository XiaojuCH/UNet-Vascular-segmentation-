import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as T
import cv2
from Unet_with_attention import UNet_Attention
from Unet_train import UNet  # 导入标准UNet

# -------------------------
# 1. 加载带注意力的模型
# -------------------------
def load_attention_model(model_path, device="cuda"):
    """加载带注意力机制的UNet模型"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 从checkpoint中读取注意力配置
    if 'meta' in checkpoint and 'attention_type' in checkpoint['meta']:
        attention_type = checkpoint['meta']['attention_type']
        use_attention_at = checkpoint['meta']['use_attention_at']
        print(f"[Attention Model] 从checkpoint加载配置: attention_type={attention_type}, use_attention_at={use_attention_at}")
    else:
        attention_type = 'enhanced'
        use_attention_at = ['bottleneck', 'decoder']
        print(f"[Attention Model] 使用默认配置: attention_type={attention_type}, use_attention_at={use_attention_at}")

    model = UNet_Attention(
        n_classes=2,
        attention_type=attention_type,
        use_attention_at=use_attention_at
    ).to(device)

    model.load_state_dict(checkpoint['net'])
    model.eval()
    return model


# -------------------------
# 2. 加载标准UNet模型
# -------------------------
def load_baseline_model(model_path, device="cuda"):
    """加载标准UNet模型（无注意力）"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    print(f"[Baseline Model] 加载标准UNet模型")

    model = UNet(n_classes=2).to(device)
    model.load_state_dict(checkpoint['net'])
    model.eval()
    return model


# -------------------------
# 3. 对单张图做推理（支持双模型）
# -------------------------
def predict_single(model, img_path, device="cuda"):
    """使用单个模型对图像进行预测"""
    # 打开图片并转 tensor
    img = Image.open(img_path).convert("RGB")
    tf = T.ToTensor()
    img_tensor = tf(img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        out = model(img_tensor)
        pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()

    return pred  # 返回 mask（0/1）矩阵


def predict_dual_models(attention_model, baseline_model, img_path, device="cuda"):
    """使用两个模型同时对图像进行预测"""
    # 打开图片并转 tensor
    img = Image.open(img_path).convert("RGB")
    tf = T.ToTensor()
    img_tensor = tf(img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        # 带注意力的模型预测
        out_attention = attention_model(img_tensor)
        pred_attention = torch.argmax(out_attention, dim=1).squeeze().cpu().numpy()

        # 标准模型预测
        out_baseline = baseline_model(img_tensor)
        pred_baseline = torch.argmax(out_baseline, dim=1).squeeze().cpu().numpy()

    return pred_attention, pred_baseline


# -------------------------
# 4. 可视化 + 保存结果
# -------------------------
def save_mask(mask, save_path):
    """保存二值mask（0/255）"""
    mask_img = (mask * 255).astype(np.uint8)
    cv2.imwrite(save_path, mask_img)

def save_color_mask(mask, save_path, color=[0, 255, 255]):
    """保存彩色mask"""
    color_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_img[mask == 1] = color  # 前景颜色
    color_img[mask == 0] = [0, 0, 0]  # 背景黑色
    cv2.imwrite(save_path, color_img)


# -------------------------
# 5. 主函数（双模型批量预测）
# -------------------------
if __name__ == "__main__":
    import os
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print("="*70)

    # ==================== 配置参数 ====================
    # 模型路径配置
    attention_model_path = "save_model_attention/checkpoints/run_best.pth"
    baseline_model_path = "save_model_baseline/checkpoints/run_best.pth"

    # 输入输出路径配置
    input_dir = "D:\Projects_\JiaBi\TransUNET\predict_input"  # 待预测的图像文件夹
    output_dir = "D:\Projects_\JiaBi\TransUNET\predict_output"  # 预测结果保存文件夹

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "attention"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "baseline"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparison"), exist_ok=True)

    # ==================== 加载模型 ====================
    print("\n加载模型...")
    attention_model = load_attention_model(attention_model_path, device)
    baseline_model = load_baseline_model(baseline_model_path, device)
    print("两个模型加载完成！")
    print("="*70)

    # ==================== 获取所有图像文件 ====================
    # 支持的图像格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # 获取文件夹中所有图像文件
    img_files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(image_extensions)]

    if len(img_files) == 0:
        print(f"错误: 在 {input_dir} 中没有找到图像文件！")
        exit(1)

    print(f"\n找到 {len(img_files)} 张图像")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("="*70)

    # ==================== 批量预测 ====================
    print("\n开始批量预测...")

    for img_file in tqdm(img_files, desc="预测进度"):
        img_path = os.path.join(input_dir, img_file)
        base_name = os.path.splitext(img_file)[0]

        try:
            # 使用两个模型同时预测
            pred_attention, pred_baseline = predict_dual_models(
                attention_model, baseline_model, img_path, device
            )

            # 保存结果
            # 1. 带注意力模型的结果（青色）
            attention_save_path = os.path.join(output_dir, "attention", f"{base_name}_attention.png")
            save_color_mask(pred_attention, attention_save_path, color=[0, 255, 255])

            # 2. 标准模型的结果（红色）
            baseline_save_path = os.path.join(output_dir, "baseline", f"{base_name}_baseline.png")
            save_color_mask(pred_baseline, baseline_save_path, color=[0, 0, 255])

            # 3. 对比图：将两个预测结果叠加显示
            comparison_img = np.zeros((pred_attention.shape[0], pred_attention.shape[1], 3), dtype=np.uint8)
            comparison_img[pred_attention == 1] = [0, 255, 255]  # 注意力模型：青色
            comparison_img[pred_baseline == 1] = [0, 0, 255]     # 标准模型：红色
            overlap = (pred_attention == 1) & (pred_baseline == 1)
            comparison_img[overlap] = [0, 255, 0]  # 重叠区域：绿色

            comparison_save_path = os.path.join(output_dir, "comparison", f"{base_name}_comparison.png")
            cv2.imwrite(comparison_save_path, comparison_img)

        except Exception as e:
            print(f"\n警告: 处理 {img_file} 时出错: {str(e)}")
            continue

    # ==================== 完成 ====================
    print("\n" + "="*70)
    print("批量预测完成！")
    print(f"共处理 {len(img_files)} 张图像")
    print(f"\n结果保存在:")
    print(f"  - 注意力模型: {os.path.join(output_dir, 'attention')}")
    print(f"  - 标准模型: {os.path.join(output_dir, 'baseline')}")
    print(f"  - 对比图: {os.path.join(output_dir, 'comparison')}")
    print("\n颜色说明:")
    print("  - 青色 = 注意力模型预测")
    print("  - 红色 = 标准模型预测")
    print("  - 绿色 = 两者重叠区域")
    print("="*70)
