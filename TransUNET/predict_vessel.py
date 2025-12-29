import argparse
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 导入 Scipy 用于计算 HD95
from scipy.ndimage import distance_transform_edt

# 导入 UNet++ 模型
from networks.unet_plus_plus import UNetPlusPlus 

# 导入数据集相关
from torch.utils.data import DataLoader, random_split
from datasets.dataset_vessel import VesselDataset

import sys

# ==================== 1. 本地定义评估指标函数 ====================
def compute_hd95(pred, gt, spacing=None):
    """计算 Hausdorff Distance 95%"""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    
    if pred.sum() == 0 or gt.sum() == 0: 
        return 0.0
        
    dt_gt = distance_transform_edt(~gt, sampling=spacing)
    dt_pred = distance_transform_edt(~pred, sampling=spacing)
    
    sds_pred = dt_gt[pred]
    sds_gt = dt_pred[gt]
    
    if sds_pred.size + sds_gt.size == 0: 
        return 0.0
        
    return np.percentile(np.concatenate([sds_pred, sds_gt]), 95)

def calculate_metrics_local(pred, target, num_classes=2):
    """
    本地计算 Batch 级别的评估指标: Acc, mIoU, Dice, HD95
    """
    # 确保输入是 numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.data.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.data.cpu().numpy()
    
    # 确保是 int 类型
    pred = pred.astype(int)
    target = target.astype(int)
    
    batch_acc = []
    batch_miou = []
    batch_dice = []
    batch_hd95 = []

    for i in range(pred.shape[0]):
        p = pred[i]
        t = target[i]
        
        # Pixel Accuracy
        acc = np.sum(p == t) / (p.size + 1e-8)
        batch_acc.append(acc)
        
        # IoU & Dice
        class_ious = []
        class_dices = []
        
        for c in range(num_classes):
            p_c = (p == c)
            t_c = (t == c)
            intersection = np.logical_and(p_c, t_c).sum()
            union = np.logical_or(p_c, t_c).sum()
            
            # IoU
            if union == 0:
                iou = 1.0 
            else:
                iou = intersection / (union + 1e-8)
            class_ious.append(iou)
            
            # Dice
            if p_c.sum() + t_c.sum() == 0:
                dice = 1.0
            else:
                dice = 2 * intersection / (p_c.sum() + t_c.sum() + 1e-8)
            class_dices.append(dice)
            
        batch_miou.append(np.mean(class_ious))
        batch_dice.append(np.mean(class_dices))
        
        # HD95 (只针对血管类，即 label=1)
        try:
            if np.sum(p == 1) > 0 and np.sum(t == 1) > 0:
                hd95_val = compute_hd95(p == 1, t == 1)
                batch_hd95.append(hd95_val)
            else:
                batch_hd95.append(0.0)
        except Exception:
            batch_hd95.append(0.0)
            
    return np.mean(batch_acc), np.mean(batch_miou), np.mean(batch_dice), np.mean(batch_hd95)

# ==================== 2. 参数设置 ====================
def get_args():
    parser = argparse.ArgumentParser()

    # 模型参数
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型路径')
    parser.add_argument('--img_size', type=int, default=256, help='输入图像尺寸')
    parser.add_argument('--num_classes', type=int, default=2, help='分类数量')
    
    # 数据集参数
    parser.add_argument('--img_dir', type=str, default='../dataset/image', help='图像文件夹路径')
    parser.add_argument('--mask_dir', type=str, default='../dataset/mask', help='掩码文件夹路径')
    parser.add_argument('--train_ratio', type=float, default=0.70, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 预测输出参数
    parser.add_argument('--output_dir', type=str, default='./predict_output_Transunet_SCSA', help='预测结果保存目录')
    parser.add_argument('--save_overlay', action='store_true', help='是否保存叠加可视化结果')
    parser.add_argument('--save_original_and_gt', action='store_true', help='是否保存原始图片和真实标签')
    parser.add_argument('--evaluate_metrics', action='store_true', help='是否计算评估指标')
    
    # [新增] 是否在 resize 后的尺寸上评估（如果选这个，指标应该和训练时一致）
    parser.add_argument('--eval_on_resize', action='store_true', 
                        help='如果开启，将在256x256尺寸上计算指标（与训练时一致）；否则在原图尺寸计算（更真实但通常稍低）。')

    args = parser.parse_args()
    return args


def load_model(args, device):
    """加载训练好的 UNet++ 模型"""
    print("加载 UNet++ 模型...")
    # 注意：如果使用了 SCSA 注意力，确保这里的 UNetPlusPlus 类定义包含了 SCSA 模块
    model = UNetPlusPlus(num_classes=args.num_classes, input_channels=3)
    model = model.to(device)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    print(f"加载模型成功: {args.model_path}")
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        print(f"Checkpoint Epoch: {checkpoint['epoch']}, mIoU: {checkpoint.get('val_miou', 'N/A')}")

    model.eval()
    return model


def save_prediction(pred_np, output_path, original_size):
    """保存预测结果"""
    # pred_np 是 0 和 1
    pred_img = (pred_np * 255).astype(np.uint8)
    pred_pil = Image.fromarray(pred_img, mode='L')
    pred_pil = pred_pil.resize(original_size, Image.NEAREST)
    pred_pil.save(output_path)


def save_overlay(original_img_np, pred_np, output_path, original_size, alpha=0.3):
    """保存叠加结果"""
    pred_resized = Image.fromarray((pred_np * 255).astype(np.uint8), mode='L')
    pred_resized = pred_resized.resize(original_size, Image.NEAREST)
    pred_resized = np.array(pred_resized)

    mask = pred_resized > 127
    color_mask = np.zeros_like(original_img_np, dtype=np.uint8)
    color_mask[mask] = [0, 255, 255] # 青色

    original_img_f = original_img_np.astype(np.float32)
    color_mask_f = color_mask.astype(np.float32)

    overlay = original_img_f * (1 - alpha) + color_mask_f * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    Image.fromarray(overlay).save(output_path)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_original_and_gt:
        os.makedirs(os.path.join(args.output_dir, 'original_images'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'ground_truth_masks'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'prediction_masks'), exist_ok=True)
    if args.save_overlay:
        os.makedirs(os.path.join(args.output_dir, 'overlay_results'), exist_ok=True)


    # ==================== 数据集加载 ====================
    print("="*70)
    print("加载数据集...")
    full_dataset = VesselDataset(
        args.img_dir,
        args.mask_dir,
        img_size=args.img_size,
        augment=False,
        verbose=True
    )

    total_size = len(full_dataset)
    train_size = int(args.train_ratio * total_size)
    val_size = int(args.val_ratio * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset, test_dataset_indices = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    test_dataset = torch.utils.data.Subset(full_dataset, test_dataset_indices.indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    print(f"成功加载 {len(test_dataset)} 张测试图像.")

    # ==================== 加载模型 ====================
    model = load_model(args, device)

    print("\n" + "="*70)
    mode_str = "Resize后尺寸 (256x256) - 模拟训练指标" if args.eval_on_resize else "原始尺寸 - 真实应用指标"
    print(f"开始对测试集进行预测... (评估模式: {mode_str})")

    total_acc, total_miou, total_dice, total_hd95 = 0, 0, 0, 0
    num_batches = 0

    for i, sampled_batch in enumerate(tqdm(test_loader, desc="预测进度")):
        # 获取数据
        image_batch = sampled_batch['image'].to(device)
        label_batch = sampled_batch['label'].to(device) # 这是 256x256 的 GT
        
        # 获取原始信息
        original_filepath = sampled_batch['original_filepath'][0]
        original_image_np = sampled_batch['original_image'].numpy()[0] 
        # 这里原始掩码可能已经包含了 0, 255 或者 0, 1
        original_mask_np = sampled_batch['original_mask'].numpy()[0]
        
        # [修复 1] 智能处理 GT 掩码的值
        unique_vals = np.unique(original_mask_np)
        if len(unique_vals) > 1 and np.max(unique_vals) == 1:
            # 如果最大值是1，说明已经是二值图，不需要阈值处理，直接用
            gt_mask_binary = original_mask_np.astype(int)
            gt_mask_for_save = (original_mask_np * 255).astype(np.uint8)
        else:
            # 否则假设是 0-255，做阈值处理
            gt_mask_binary = (original_mask_np > 127).astype(int)
            gt_mask_for_save = (gt_mask_binary * 255).astype(np.uint8)

        base_name = os.path.splitext(os.path.basename(original_filepath))[0]
        original_size = (original_image_np.shape[1], original_image_np.shape[0])

        # 预测
        with torch.no_grad():
            outputs = model(image_batch)
            pred = torch.argmax(outputs, dim=1).squeeze(0) # Tensor [256, 256]
            pred_np = pred.cpu().numpy() # Numpy [256, 256]

        # 1. 保存原始图片和GT
        if args.save_original_and_gt:
            Image.fromarray(original_image_np.astype(np.uint8)).save(
                os.path.join(args.output_dir, 'original_images', f'{base_name}_original.png'))
            # 保存修正后的 GT（确保是黑白的）
            Image.fromarray(gt_mask_for_save, mode='L').save(
                os.path.join(args.output_dir, 'ground_truth_masks', f'{base_name}_gt.png'))

        # 2. 保存预测掩码
        pred_output_path = os.path.join(args.output_dir, 'prediction_masks', f'{base_name}_pred.png')
        save_prediction(pred_np, pred_output_path, original_size)

        # 3. 保存叠加可视化
        if args.save_overlay:
            overlay_output_path = os.path.join(args.output_dir, 'overlay_results', f'{base_name}_overlay.png')
            save_overlay(original_image_np, pred_np, overlay_output_path, original_size)
            
        # 4. 计算指标
        if args.evaluate_metrics:
            if args.eval_on_resize:
                # 模式 A: 在 256x256 上计算 (跟训练时一致)
                # pred 已经是 256x256, label_batch 也是 256x256
                acc, miou, mdice, hd95 = calculate_metrics_local(
                    pred.unsqueeze(0), 
                    label_batch, 
                    num_classes=args.num_classes
                )
            else:
                # 模式 B: 在原始尺寸上计算 (真实效果)
                # 需要把 pred resize 回原始尺寸
                pred_pil = Image.fromarray(pred_np.astype(np.uint8), mode='L')
                pred_resized = pred_pil.resize(original_size, Image.NEAREST)
                pred_final = np.array(pred_resized).astype(int) # 0, 1
                
                # 计算指标
                acc, miou, mdice, hd95 = calculate_metrics_local(
                    np.expand_dims(pred_final, 0), # [1, H_orig, W_orig]
                    np.expand_dims(gt_mask_binary, 0), # [1, H_orig, W_orig]
                    num_classes=args.num_classes
                )

            total_acc += acc
            total_miou += miou
            total_dice += mdice
            total_hd95 += hd95
            num_batches += 1

    print("\n" + "="*70)
    print("预测完成！")
    print(f"结果已保存到目录: {args.output_dir}")

    if args.evaluate_metrics and num_batches > 0:
        avg_acc = total_acc / num_batches
        avg_miou = total_miou / num_batches
        avg_dice = total_dice / num_batches
        avg_hd95 = total_hd95 / num_batches
        
        print("\n" + "="*70)
        print(f"测试集平均评估指标 ({mode_str}):")
        print(f"  - Pixel Acc: {avg_acc:.4f}")
        print(f"  - mIoU: {avg_miou:.4f}")
        print(f"  - Dice: {avg_dice:.4f}")
        print(f"  - HD95: {avg_hd95:.4f}")
        print("="*70)

if __name__ == "__main__":
    main()