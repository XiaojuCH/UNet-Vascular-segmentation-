"""
U-Net 训练工具函数
包含：Early Stopping、检查点保存、指标计算、测试函数、可视化等
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt


# =========================
#   Early Stopping 类
# =========================
class EarlyStopping:
    """早停机制，监控验证 mIoU"""
    def __init__(self, patience=15, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = 0.0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True


# =========================
#   保存检查点函数
# =========================
def save_checkpoint(model, epoch, val_miou, save_dir, name, optimizer=None, meta=None):
    """保存模型检查点"""
    state = {
        'net': model.state_dict(),
        'epoch': epoch,
        'val_miou': val_miou,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'meta': meta or {}
    }
    path = os.path.join(save_dir, f'{name}.pth')
    torch.save(state, path)
    return path


# =========================
#   评估指标计算
# =========================
def calculate_metrics(pred, target, num_classes=2):
    """
    计算分割指标：Pixel Accuracy, mIoU, Dice
    pred: 预测结果 (B, H, W)
    target: 真实标签 (B, H, W)
    """
    pred = pred.view(-1)
    target = target.view(-1)

    # Pixel Accuracy
    correct = (pred == target).sum().item()
    total = target.numel()
    pixel_acc = correct / total

    # 计算每个类别的 IoU 和 Dice
    ious = []
    dices = []

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        if union == 0:
            iou = 1.0  # 如果该类别不存在，认为IoU为1
            dice = 1.0
        else:
            iou = intersection / union
            dice = 2 * intersection / (pred_cls.sum().item() + target_cls.sum().item())

        ious.append(iou)
        dices.append(dice)

    miou = np.mean(ious)
    mdice = np.mean(dices)

    return pixel_acc, miou, mdice, ious, dices


# =========================
#   测试函数
# =========================
def test_model(model, test_loader, device, loss_fn):
    """在测试集上评估模型"""
    model.eval()
    test_loss = 0
    test_acc = 0
    test_miou = 0
    test_mdice = 0

    with torch.no_grad():
        for img, mask in test_loader:
            img = img.to(device)
            mask = mask.to(device)

            out = model(img)
            loss = loss_fn(out, mask)
            test_loss += loss.item()

            pred = torch.argmax(out, dim=1)
            pixel_acc, miou, mdice, _, _ = calculate_metrics(pred, mask, num_classes=2)
            test_acc += pixel_acc
            test_miou += miou
            test_mdice += mdice

    # 计算平均值
    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_acc / len(test_loader)
    avg_test_miou = test_miou / len(test_loader)
    avg_test_mdice = test_mdice / len(test_loader)

    return avg_test_loss, avg_test_acc, avg_test_miou, avg_test_mdice


# =========================
#   绘制训练曲线
# =========================
def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         train_mious, val_mious, train_mdices, val_mdices,
                         save_path):
    """
    绘制训练和验证的Loss、Accuracy、mIoU、Dice曲线

    Args:
        train_losses: 训练集loss列表
        val_losses: 验证集loss列表
        train_accs: 训练集accuracy列表
        val_accs: 验证集accuracy列表
        train_mious: 训练集mIoU列表
        val_mious: 验证集mIoU列表
        train_mdices: 训练集Dice列表
        val_mdices: 验证集Dice列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epoch_range = range(1, len(train_losses) + 1)

    # Loss曲线
    axes[0, 0].plot(epoch_range, train_losses, 'b-', linewidth=2, label='Train Loss')
    axes[0, 0].plot(epoch_range, val_losses, 'r-', linewidth=2, label='Val Loss')
    axes[0, 0].set_xlabel("Epoch", fontsize=11)
    axes[0, 0].set_ylabel("Loss", fontsize=11)
    axes[0, 0].set_title("Training & Validation Loss", fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy曲线
    axes[0, 1].plot(epoch_range, train_accs, 'b-', linewidth=2, label='Train Acc')
    axes[0, 1].plot(epoch_range, val_accs, 'r-', linewidth=2, label='Val Acc')
    axes[0, 1].set_xlabel("Epoch", fontsize=11)
    axes[0, 1].set_ylabel("Pixel Accuracy", fontsize=11)
    axes[0, 1].set_title("Pixel Accuracy", fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # mIoU曲线
    axes[1, 0].plot(epoch_range, train_mious, 'b-', linewidth=2, label='Train mIoU')
    axes[1, 0].plot(epoch_range, val_mious, 'r-', linewidth=2, label='Val mIoU')
    axes[1, 0].set_xlabel("Epoch", fontsize=11)
    axes[1, 0].set_ylabel("mIoU", fontsize=11)
    axes[1, 0].set_title("Mean IoU", fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Dice曲线
    axes[1, 1].plot(epoch_range, train_mdices, 'b-', linewidth=2, label='Train Dice')
    axes[1, 1].plot(epoch_range, val_mdices, 'r-', linewidth=2, label='Val Dice')
    axes[1, 1].set_xlabel("Epoch", fontsize=11)
    axes[1, 1].set_ylabel("Dice Coefficient", fontsize=11)
    axes[1, 1].set_title("Dice Coefficient", fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存至: {save_path}")


# =========================
#   打印训练总结
# =========================
def print_training_summary(run_best_miou, best_epoch, val_losses, val_accs,
                          val_mious, val_mdices, test_loss, test_acc,
                          test_miou, test_mdice, best_ckpt_path,
                          final_model_path, curves_path):
    """
    打印训练总结信息
    """
    print("\n" + "="*70)
    print("训练总结")
    print("="*70)
    print(f"最佳验证 mIoU: {run_best_miou:.4f} (Epoch {best_epoch if best_epoch else 'N/A'})")
    print(f"\n最终验证指标 (Epoch {len(val_losses)}):")
    print(f"  - Loss: {val_losses[-1]:.6f}")
    print(f"  - Pixel Acc: {val_accs[-1]:.4f}")
    print(f"  - mIoU: {val_mious[-1]:.4f}")
    print(f"  - Dice: {val_mdices[-1]:.4f}")
    print(f"\n测试集指标:")
    print(f"  - Loss: {test_loss:.6f}")
    print(f"  - Pixel Acc: {test_acc:.4f}")
    print(f"  - mIoU: {test_miou:.4f}")
    print(f"  - Dice: {test_mdice:.4f}")
    print(f"\n保存文件:")
    print(f"  - 最佳模型: {best_ckpt_path}")
    print(f"  - 最终模型: {final_model_path}")
    print(f"  - 训练曲线: {curves_path}")
    print("="*70)
