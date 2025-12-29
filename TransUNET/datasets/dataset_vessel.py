import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class VesselDataset(Dataset):
    """
    甲襞血管分割数据集
    适配PNG/JPG格式的图像和掩码
    """
    def __init__(self, img_dir, mask_dir, img_size=256, augment=True, verbose=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

        # 只保留同时在 image 和 mask 文件夹中都存在的文件
        img_files = set(os.listdir(img_dir))
        mask_files = set(os.listdir(mask_dir))
        # 取交集并排序，确保一一对应
        self.file_list = sorted(list(img_files & mask_files))

        if verbose:
            print(f"找到 {len(self.file_list)} 对匹配的图像和掩码文件")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        
        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        # 读取图像和掩码
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # ==================== [新增] 保存原始信息用于预测 ====================
        # 在做resize和augmentation之前，保存原始数据的副本
        # 这里的 numpy 数组将用于测试集评估时的 reshape 回原图
        original_image_np = np.array(img)
        original_mask_np = np.array(mask)
        # ===================================================================

        # 数据增强 (仅在 augment=True 时启用，通常用于训练集)
        if self.augment:
            # 1. 随机水平翻转
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            # 2. 随机垂直翻转
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            # 3. 随机旋转 (0, 90, 180, 270度)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)

            # 4. 随机小角度旋转 (-15到15度)
            if random.random() > 0.7:
                angle = random.uniform(-15, 15)
                img = TF.rotate(img, angle)
                mask = TF.rotate(mask, angle)

            # 5. 随机亮度调整 (仅调整图像)
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.7, 1.3)
                img = TF.adjust_brightness(img, brightness_factor)

        # 调整大小到模型输入尺寸 (例如 256x256)
        # 注意：PIL resize 传入 (W, H)
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # 转换为numpy数组
        img = np.array(img)
        mask = np.array(mask)

        # 归一化图像到 [0, 1]
        img = img.astype(np.float32) / 255.0

        # 掩码二值化: 0 (背景) 和 1 (血管)
        mask = (mask > 127).astype(np.float32)

        # 转换为tensor格式
        # 图像: (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        # 掩码: (H, W) -> (H, W)
        mask_tensor = torch.from_numpy(mask).long()

        sample = {
            'image': img_tensor,
            'label': mask_tensor,
            'case_name': name,
            # [新增] 返回路径和原始数据，解决 KeyError
            'original_filepath': img_path,
            'original_maskpath': mask_path,
            'original_image': original_image_np,
            'original_mask': original_mask_np
        }

        return sample


# 保留你原有的 RandomGenerator，防止 utils 或其他代码引用报错
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # 这里仅作保留，实际上 VesselDataset 内部已经处理了增强
        # 如果需要兼容 TransUNet 原有的 transform 逻辑，可以保留
        return sample