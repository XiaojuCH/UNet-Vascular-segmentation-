"""
带注意力机制的U-Net模型
用于甲襞血管分割任务
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import time
from datetime import timedelta
from tqdm import trange

# 导入工具函数
from utils import (
    EarlyStopping,
    save_checkpoint,
    calculate_metrics,
    test_model,
    plot_training_curves,
    print_training_summary
)

# 导入注意力模块和配置
from New_UA import get_scsa_attention
from attention_config import get_attention_config, print_all_configs


# =========================
#   带注意力的U-Net模型
# =========================
class DoubleConv(nn.Module):
    """双卷积块"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet_Attention(nn.Module):
    """
    带注意力机制的U-Net
    适用于甲襞血管分割任务

    Args:
        n_classes: 分类数量（背景+血管=2）
        attention_type: 注意力类型 ('original' 或 'enhanced')
        use_attention_at: 在哪些层使用注意力 (list of str: 'encoder', 'bottleneck', 'decoder')
    """
    def __init__(self, n_classes=2, attention_type='enhanced',
                 use_attention_at=['bottleneck', 'decoder']):
        super().__init__()
        self.attention_type = attention_type
        self.use_attention_at = use_attention_at

        # Encoder
        self.down1 = DoubleConv(3, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        # Bottleneck
        self.middle = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        # 输出层
        self.out = nn.Conv2d(64, n_classes, 1)

        # 注意力模块
        self.attention_modules = nn.ModuleDict()

        # Encoder注意力
        if 'encoder' in use_attention_at:
            self.attention_modules['enc1'] = get_scsa_attention(attention_type, 64)
            self.attention_modules['enc2'] = get_scsa_attention(attention_type, 128)
            self.attention_modules['enc3'] = get_scsa_attention(attention_type, 256)
            self.attention_modules['enc4'] = get_scsa_attention(attention_type, 512)

        # Bottleneck注意力
        if 'bottleneck' in use_attention_at:
            self.attention_modules['bottleneck'] = get_scsa_attention(attention_type, 1024)

        # Decoder注意力
        if 'decoder' in use_attention_at:
            self.attention_modules['dec4'] = get_scsa_attention(attention_type, 512)
            self.attention_modules['dec3'] = get_scsa_attention(attention_type, 256)
            self.attention_modules['dec2'] = get_scsa_attention(attention_type, 128)
            self.attention_modules['dec1'] = get_scsa_attention(attention_type, 64)

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
        if 'enc1' in self.attention_modules:
            c1 = self.attention_modules['enc1'](c1)
        p1 = F.max_pool2d(c1, 2)

        c2 = self.down2(p1)
        if 'enc2' in self.attention_modules:
            c2 = self.attention_modules['enc2'](c2)
        p2 = F.max_pool2d(c2, 2)

        c3 = self.down3(p2)
        if 'enc3' in self.attention_modules:
            c3 = self.attention_modules['enc3'](c3)
        p3 = F.max_pool2d(c3, 2)

        c4 = self.down4(p3)
        if 'enc4' in self.attention_modules:
            c4 = self.attention_modules['enc4'](c4)
        p4 = F.max_pool2d(c4, 2)

        # Bottleneck
        mid = self.middle(p4)
        if 'bottleneck' in self.attention_modules:
            mid = self.attention_modules['bottleneck'](mid)

        # Decoder
        up4 = self.up4(mid)
        merge4 = torch.cat([up4, c4], dim=1)
        c5 = self.conv4(merge4)
        if 'dec4' in self.attention_modules:
            c5 = self.attention_modules['dec4'](c5)

        up3 = self.up3(c5)
        merge3 = torch.cat([up3, c3], dim=1)
        c6 = self.conv3(merge3)
        if 'dec3' in self.attention_modules:
            c6 = self.attention_modules['dec3'](c6)

        up2 = self.up2(c6)
        merge2 = torch.cat([up2, c2], dim=1)
        c7 = self.conv2(merge2)
        if 'dec2' in self.attention_modules:
            c7 = self.attention_modules['dec2'](c7)

        up1 = self.up1(c7)
        merge1 = torch.cat([up1, c1], dim=1)
        c8 = self.conv1(merge1)
        if 'dec1' in self.attention_modules:
            c8 = self.attention_modules['dec1'](c8)

        return self.out(c8)


# =========================
#   数据集（与原版相同）
# =========================
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=True, verbose=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        img_files = set(os.listdir(img_dir))
        mask_files = set(os.listdir(mask_dir))
        self.list = sorted(list(img_files & mask_files))
        if verbose:
            print(f"找到 {len(self.list)} 对匹配的图像和掩码文件")
        self.augment = augment

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        name = self.list[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name))

        if self.augment:
            if np.random.rand() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if np.random.rand() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

        img = T.ToTensor()(img)
        mask = torch.tensor(np.array(mask) // 255, dtype=torch.long)

        return img, mask


# =========================
#   训练函数
# =========================
def train():
    train_imgs = "dataset/image"
    train_masks = "dataset/mask"
    save_dir = "save_model_attention"
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("="*70)
    print("甲襞血管分割 - U-Net with Enhanced Attention")
    print("="*70)

    # 获取注意力配置
    attention_config = get_attention_config()
    print(f"\n注意力配置: {attention_config['description']}")
    print(f"预估显存占用: {attention_config['estimated_vram']}")

    # 数据集划分：训练集70%、验证集15%、测试集15%
    full_dataset = SegDataset(train_imgs, train_masks, augment=True, verbose=True)

    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    dataset_no_aug = SegDataset(train_imgs, train_masks, augment=False, verbose=False)
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices
    val_dataset_clean = torch.utils.data.Subset(dataset_no_aug, val_indices)
    test_dataset_clean = torch.utils.data.Subset(dataset_no_aug, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset_clean, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset_clean, batch_size=8, shuffle=False, num_workers=0)

    # 创建带注意力的模型（使用配置文件）
    model = UNet_Attention(
        n_classes=2,
        attention_type=attention_config['attention_type'],
        use_attention_at=attention_config['use_attention_at']
    ).to(device)

    print(f"\n模型配置:")
    print(f"  - 注意力类型: {attention_config['attention_type']}")
    print(f"  - 注意力位置: {', '.join(attention_config['use_attention_at'])}")
    print(f"  - 任务: 甲襞血管分割")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 50
    print(f"\n开始训练...")
    print(f"  - 训练集: {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)")
    print(f"  - 验证集: {len(val_dataset_clean)} ({len(val_dataset_clean)/total_size*100:.1f}%)")
    print(f"  - 测试集: {len(test_dataset_clean)} ({len(test_dataset_clean)/total_size*100:.1f}%)")
    print("="*70)

    # 记录指标
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_mious, val_mious = [], []
    train_mdices, val_mdices = [], []
    epoch_times = []
    run_best_miou = 0.0

    early_stop = EarlyStopping(patience=15, verbose=True)

    for epoch in trange(1, epochs + 1, desc="Epochs"):
        epoch_start = time.time()

        # 训练阶段
        model.train()
        train_loss = train_acc = train_miou = train_mdice = 0

        for i, (img, mask) in enumerate(train_loader):
            img, mask = img.to(device), mask.to(device)
            out = model(img)
            loss = loss_fn(out, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = torch.argmax(out, dim=1)
            pixel_acc, miou, mdice, _, _ = calculate_metrics(pred, mask, num_classes=2)
            train_acc += pixel_acc
            train_miou += miou
            train_mdice += mdice

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_train_miou = train_miou / len(train_loader)
        avg_train_mdice = train_mdice / len(train_loader)

        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        train_mious.append(avg_train_miou)
        train_mdices.append(avg_train_mdice)

        # 验证阶段
        model.eval()
        val_loss = val_acc = val_miou = val_mdice = 0

        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                out = model(img)
                loss = loss_fn(out, mask)
                val_loss += loss.item()

                pred = torch.argmax(out, dim=1)
                pixel_acc, miou, mdice, _, _ = calculate_metrics(pred, mask, num_classes=2)
                val_acc += pixel_acc
                val_miou += miou
                val_mdice += mdice

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        avg_val_miou = val_miou / len(val_loader)
        avg_val_mdice = val_mdice / len(val_loader)

        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)
        val_mious.append(avg_val_miou)
        val_mdices.append(avg_val_mdice)

        # 计算时间和ETA
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta = str(timedelta(seconds=int(eta_seconds)))

        print(f"[Epoch {epoch}/{epochs}] "
              f"Val mIoU: {avg_val_miou:.4f} Val Acc: {avg_val_acc:.4f} "
              f"Val Dice: {avg_val_mdice:.4f} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Time: {epoch_time:.2f}s ETA: {eta}")

        # 保存最佳模型
        if avg_val_miou > run_best_miou + 1e-5:
            run_best_miou = avg_val_miou
            meta = {
                'epoch': epoch,
                'timestamp': time.time(),
                'val_miou': float(avg_val_miou),
                'val_acc': float(avg_val_acc),
                'val_dice': float(avg_val_mdice),
                'train_loss': float(avg_train_loss),
                'attention_type': 'enhanced',
                'use_attention_at': ['bottleneck', 'decoder']
            }
            save_checkpoint(model, epoch, avg_val_miou, ckpt_dir, 'run_best',
                          optimizer=optimizer, meta=meta)
            print(f">>> New run-best at epoch {epoch}, Val mIoU={avg_val_miou:.4f} "
                  f"Val Acc={avg_val_acc:.4f} Val Dice={avg_val_mdice:.4f}")

        # Early Stopping
        early_stop(avg_val_miou, epoch)
        if early_stop.early_stop:
            print(f"\n⏹️ EarlyStopping triggered at epoch {early_stop.best_epoch}")
            break

    # 训练结束
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)

    # 加载最佳模型
    print("\n===== 加载最佳模型 (run_best) =====")
    best_ckpt_path = os.path.join(ckpt_dir, 'run_best.pth')
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['net'])
        best_epoch = ckpt['epoch']
        best_miou = ckpt['val_miou']
        print(f"最佳模型来自 Epoch {best_epoch}, Val mIoU: {best_miou:.4f}")
    else:
        print("未找到最佳模型检查点，使用最后一个epoch的模型")

    # 绘制训练曲线
    os.makedirs(save_dir, exist_ok=True)
    curves_path = os.path.join(save_dir, "training_curves.png")
    plot_training_curves(
        train_losses, val_losses,
        train_accs, val_accs,
        train_mious, val_mious,
        train_mdices, val_mdices,
        curves_path
    )

    # 保存最终模型
    final_model_path = os.path.join(save_dir, "unet_attention_final.pth")
    torch.save(model.state_dict(), final_model_path)

    # 测试集评估
    print("\n" + "="*70)
    print("===== 测试集评估 (使用最佳模型) =====")
    print("="*70)

    test_loss, test_acc, test_miou, test_mdice = test_model(
        model, test_loader, device, loss_fn
    )

    print(f"\n测试集结果:")
    print(f"  - Loss: {test_loss:.6f}")
    print(f"  - Pixel Acc: {test_acc:.4f}")
    print(f"  - mIoU: {test_miou:.4f}")
    print(f"  - Dice: {test_mdice:.4f}")

    # 打印最终总结
    print_training_summary(
        run_best_miou, best_epoch if os.path.exists(best_ckpt_path) else None,
        val_losses, val_accs, val_mious, val_mdices,
        test_loss, test_acc, test_miou, test_mdice,
        best_ckpt_path, final_model_path, curves_path
    )


if __name__ == "__main__":
    train()
