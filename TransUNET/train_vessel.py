import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm, trange
from datetime import timedelta

# [新增] 导入 scipy 用于计算 HD95
from scipy.ndimage import distance_transform_edt

# 导入 TransUNet 模型
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# 导入 UNet++ 模型
from networks.unet_plus_plus import UNetPlusPlus

# 导入数据集
from datasets.dataset_vessel import VesselDataset

# 添加父目录到路径
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
# [修改] 这里不再导入 calculate_metrics，防止引用错文件
from utils import save_checkpoint, EarlyStopping 


# ==================== 定义评估函数 (直接写在这里，避免 import 错误) ====================
def compute_hd95(pred, gt, spacing=None):
    """
    使用 scipy 计算 Hausdorff Distance 95%
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    # 计算距离变换
    dt_gt = distance_transform_edt(~gt, sampling=spacing)
    dt_pred = distance_transform_edt(~pred, sampling=spacing)

    sds_pred = dt_gt[pred]
    sds_gt = dt_pred[gt]

    ns = sds_pred.size + sds_gt.size
    if ns == 0:
        return 0.0
    
    hd95 = np.percentile(np.concatenate([sds_pred, sds_gt]), 95)
    return hd95

def calculate_metrics(pred, target, num_classes=2):
    """
    计算 Batch 级别的评估指标: Acc, mIoU, Dice, HD95
    """
    pred = pred.data.cpu().numpy()
    target = target.data.cpu().numpy()
    
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
# ====================================================================================


# 定义DiceLoss
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def get_args():
    parser = argparse.ArgumentParser()

    # 数据集参数
    parser.add_argument('--img_dir', type=str, default='../dataset/image',
                        help='图像文件夹路径')
    parser.add_argument('--mask_dir', type=str, default='../dataset/mask',
                        help='掩码文件夹路径')
    
    # 路径保存参数 (默认值只是个占位，建议运行时通过命令行指定)
    parser.add_argument('--save_dir', type=str, default='../save_model_Transunet_SCSA',
                        help='模型保存路径')
    
    # [关键参数] 选择模型架构
    parser.add_argument('--model_arch', type=str, required=True, choices=['unet++', 'transunet'], 
                        help='选择模型架构: unet++ 或 transunet')

    # TransUNet 特有参数
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                        help='ViT模型类型')
    parser.add_argument('--n_skip', type=int, default=3,
                        help='跳跃连接数量')
    parser.add_argument('--vit_patches_size', type=int, default=16,
                        help='ViT patch大小')

    # 通用模型参数
    parser.add_argument('--img_size', type=int, default=256,
                        help='输入图像尺寸')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='分类数量（背景+血管）')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='最大训练轮数')
    parser.add_argument('--base_lr', type=float, default=0.0001,
                        help='初始学习率')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='warmup轮数')

    # 数据划分
    parser.add_argument('--train_ratio', type=float, default=0.70,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='验证集比例')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='GPU数量')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='是否使用确定性训练')
    parser.add_argument('--early_stop_patience', type=int, default=15,
                        help='Early stopping patience')

    args = parser.parse_args()
    return args


def trainer_vessel(args):
    """
    甲襞血管分割训练器
    """
    # 设置随机种子
    if args.deterministic:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # 设置日志
    logging.basicConfig(
        filename=os.path.join(args.save_dir, "train.log"),
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ==================== 数据集加载 ====================
    logging.info("="*70)
    logging.info("加载数据集...")

    full_dataset = VesselDataset(
        args.img_dir,
        args.mask_dir,
        img_size=args.img_size,
        augment=True,
        verbose=True
    )

    total_size = len(full_dataset)
    train_size = int(args.train_ratio * total_size)
    val_size = int(args.val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    dataset_no_aug = VesselDataset(
        args.img_dir,
        args.mask_dir,
        img_size=args.img_size,
        augment=False,
        verbose=False
    )

    val_dataset_clean = torch.utils.data.Subset(dataset_no_aug, val_dataset.indices)
    test_dataset_clean = torch.utils.data.Subset(dataset_no_aug, test_dataset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset_clean,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset_clean,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    logging.info(f"数据集划分:")
    logging.info(f"  - 训练集: {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)")
    logging.info(f"  - 验证集: {len(val_dataset_clean)} ({len(val_dataset_clean)/total_size*100:.1f}%)")
    logging.info(f"  - 测试集: {len(test_dataset_clean)} ({len(test_dataset_clean)/total_size*100:.1f}%)")

    # ==================== 模型初始化 ====================
    logging.info("="*70)
    
    model = None
    
    if args.model_arch == 'transunet':
        logging.info("初始化 TransUNet (with SCSA if configured) 模型...")
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.img_size / args.vit_patches_size),
                int(args.img_size / args.vit_patches_size)
            )

        model = ViT_seg(config_vit, img_size=args.img_size, num_classes=args.num_classes)
        model = model.to(device)
        
        # TransUNet 需要加载预训练权重
        pretrained_path = 'imagenet21k_R50+ViT-B_16.npz'
        if os.path.exists(pretrained_path):
            logging.info(f"加载预训练权重: {pretrained_path}")
            model.load_from(weights=np.load(pretrained_path))
        else:
            logging.warning(f"未找到预训练权重文件: {pretrained_path}，将从头训练")

    elif args.model_arch == 'unet++':
        logging.info("初始化 UNet++ (with SCSA if configured) 模型...")
        model = UNetPlusPlus(num_classes=args.num_classes, input_channels=3)
        model = model.to(device)

    else:
        raise ValueError(f"未知的模型架构: {args.model_arch}")

    # 多GPU支持
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    # ==================== 损失函数和优化器 ====================
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=0.0001)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs,
        eta_min=args.base_lr * 0.01
    )

    writer = SummaryWriter(os.path.join(args.save_dir, 'log'))

    # ==================== 训练循环 ====================
    logging.info("="*70)
    logging.info(f"开始训练 {args.model_arch}...")

    best_miou = 0.0
    best_epoch = 0

    early_stop = EarlyStopping(patience=args.early_stop_patience, verbose=True)

    epoch_times = []

    for epoch in trange(1, args.max_epochs + 1, desc="Epochs"):
        epoch_start = time.time()

        # ==================== 训练阶段 ====================
        model.train()
        train_loss = 0
        train_ce_loss = 0
        train_dice_loss = 0

        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch = sampled_batch['image'].to(device)
            label_batch = sampled_batch['label'].to(device)

            outputs = model(image_batch)

            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_ce_loss += loss_ce.item()
            train_dice_loss += loss_dice.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_ce = train_ce_loss / len(train_loader)
        avg_train_dice = train_dice_loss / len(train_loader)

        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0
        val_acc = 0
        val_miou = 0
        val_dice = 0
        val_hd95 = 0

        with torch.no_grad():
            for sampled_batch in val_loader:
                image_batch = sampled_batch['image'].to(device)
                label_batch = sampled_batch['label'].to(device)

                outputs = model(image_batch)

                loss_ce = ce_loss(outputs, label_batch.long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                val_loss += loss.item()

                pred = torch.argmax(outputs, dim=1)
                
                # 计算指标 (使用本地函数)
                acc, miou, mdice, hd95 = calculate_metrics(pred, label_batch, num_classes=2)
                
                val_acc += acc
                val_miou += miou
                val_dice += mdice
                val_hd95 += hd95

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        avg_val_miou = val_miou / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_hd95 = val_hd95 / len(val_loader)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = args.max_epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta = str(timedelta(seconds=int(eta_seconds)))

        log_msg = (f"[Epoch {epoch}/{args.max_epochs}] "
                   f"Val mIoU: {avg_val_miou:.4f} "
                   f"Val Dice: {avg_val_dice:.4f} "
                   f"Val HD95: {avg_val_hd95:.4f} " 
                   f"Val Acc: {avg_val_acc:.4f} | "
                   f"Train Loss: {avg_train_loss:.6f} | "
                   f"Time: {epoch_time:.2f}s ETA: {eta}")
        logging.info(log_msg)

        writer.add_scalar('train/loss', avg_train_loss, epoch)
        writer.add_scalar('train/ce_loss', avg_train_ce, epoch)
        writer.add_scalar('train/dice_loss', avg_train_dice, epoch)
        writer.add_scalar('val/loss', avg_val_loss, epoch)
        writer.add_scalar('val/miou', avg_val_miou, epoch)
        writer.add_scalar('val/dice', avg_val_dice, epoch)
        writer.add_scalar('val/acc', avg_val_acc, epoch)
        writer.add_scalar('val/hd95', avg_val_hd95, epoch)
        writer.add_scalar('info/lr', current_lr, epoch)

        if avg_val_miou > best_miou + 1e-5:
            best_miou = avg_val_miou
            best_epoch = epoch

            meta = {
                'epoch': epoch,
                'timestamp': time.time(),
                'val_miou': float(avg_val_miou),
                'val_acc': float(avg_val_acc),
                'val_dice': float(avg_val_dice),
                'val_hd95': float(avg_val_hd95),
                'train_loss': float(avg_train_loss)
            }

            save_checkpoint(model, epoch, avg_val_miou, ckpt_dir, 'run_best',
                          optimizer=optimizer, meta=meta)

            msg = (f">>> New run-best at epoch {epoch}, "
                   f"Val mIoU={avg_val_miou:.4f} "
                   f"Val Dice={avg_val_dice:.4f} "
                   f"Val HD95={avg_val_hd95:.4f}")
            logging.info(msg)

        early_stop(avg_val_miou, epoch)
        if early_stop.early_stop:
            msg = f"\n⏹️ EarlyStopping triggered at epoch {early_stop.best_epoch}"
            logging.info(msg)
            break

    logging.info("\n" + "="*70)
    logging.info("训练完成！")

    best_ckpt_path = os.path.join(ckpt_dir, 'run_best.pth')
    if os.path.exists(best_ckpt_path):
        logging.info(f"\n加载最佳模型: {best_ckpt_path}")
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        if args.n_gpu > 1:
            model.module.load_state_dict(ckpt['net'])
        else:
            model.load_state_dict(ckpt['net'])
        logging.info(f"最佳模型来自 Epoch {best_epoch}, Val mIoU: {best_miou:.4f}")

    # ==================== 测试集评估 ====================
    logging.info("\n" + "="*70)
    logging.info("测试集评估 (使用最佳模型)")

    model.eval()
    test_loss = 0
    test_acc = 0
    test_miou = 0
    test_dice = 0
    test_hd95 = 0

    with torch.no_grad():
        for sampled_batch in test_loader:
            image_batch = sampled_batch['image'].to(device)
            label_batch = sampled_batch['label'].to(device)

            outputs = model(image_batch)

            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            test_loss += loss.item()

            pred = torch.argmax(outputs, dim=1)
            
            # 同样调用本地函数
            acc, miou, mdice, hd95 = calculate_metrics(pred, label_batch, num_classes=2)
            
            test_acc += acc
            test_miou += miou
            test_dice += mdice
            test_hd95 += hd95

    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_acc / len(test_loader)
    avg_test_miou = test_miou / len(test_loader)
    avg_test_dice = test_dice / len(test_loader)
    avg_test_hd95 = test_hd95 / len(test_loader)

    logging.info(f"\n测试集结果:")
    logging.info(f"  - Loss: {avg_test_loss:.6f}")
    logging.info(f"  - Pixel Acc: {avg_test_acc:.4f}")
    logging.info(f"  - mIoU: {avg_test_miou:.4f}")
    logging.info(f"  - Dice: {avg_test_dice:.4f}")
    logging.info(f"  - HD95: {avg_test_hd95:.4f}")

    final_model_path = os.path.join(args.save_dir, f"{args.model_arch}_final.pth")
    if args.n_gpu > 1:
        torch.save(model.module.state_dict(), final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)

    logging.info(f"\n模型已保存:")
    logging.info(f"  - 最佳模型: {best_ckpt_path}")
    logging.info(f"  - 最终模型: {final_model_path}")

    writer.close()

    return "Training Finished!"


if __name__ == "__main__":
    args = get_args()
    trainer_vessel(args)