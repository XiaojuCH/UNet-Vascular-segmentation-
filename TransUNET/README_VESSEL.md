# TransUNet 甲襞血管分割

这是一个适配你的甲襞血管分割数据集的TransUNet训练和预测脚本。

## 📁 项目结构

```
TransUNET/
├── datasets/
│   ├── dataset_vessel.py          # 自定义数据集类（PNG格式）
│   └── dataset_synapse.py         # 原始Synapse数据集类
├── networks/
│   ├── vit_seg_modeling.py        # TransUNet模型定义
│   ├── vit_seg_configs.py         # 模型配置（已更新为2类）
│   └── ...
├── train_vessel.py                # 训练脚本（主要）
├── predict_vessel.py              # 预测脚本
├── run_train.bat                  # Windows训练启动脚本
├── run_train.sh                   # Linux/Mac训练启动脚本
├── imagenet21k_R50+ViT-B_16.npz  # 预训练权重
└── README_VESSEL.md               # 本文档
```

## 🚀 快速开始

### 1. 环境要求

确保已安装以下依赖：

```bash
pip install torch torchvision
pip install numpy pillow tqdm
pip install tensorboardX
pip install ml-collections
pip install scipy
```

### 2. 数据集准备

你的数据集已经准备好了：
- 图像路径: `../dataset/image/`
- 掩码路径: `../dataset/mask/`
- 数据量: 619对图像-掩码对
- 图像尺寸: 256×256 RGB
- 掩码格式: 二值图像（0=背景，255=血管）

数据集会自动划分为：
- 训练集: 70% (433张)
- 验证集: 15% (93张)
- 测试集: 15% (93张)

### 3. 开始训练

#### 方法1: 使用启动脚本（推荐）

**Windows:**
```bash
cd TransUNET
run_train.bat
```

**Linux/Mac:**
```bash
cd TransUNET
chmod +x run_train.sh
./run_train.sh
```

#### 方法2: 直接运行Python脚本

```bash
cd TransUNET
python train_vessel.py \
    --img_dir ../dataset/image \
    --mask_dir ../dataset/mask \
    --save_dir ../save_model_transunet \
    --img_size 256 \
    --batch_size 4 \
    --max_epochs 50 \
    --base_lr 0.0001 \
    --early_stop_patience 15 \
    --seed 42
```

### 4. 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--img_dir` | `../dataset/image` | 图像文件夹路径 |
| `--mask_dir` | `../dataset/mask` | 掩码文件夹路径 |
| `--save_dir` | `../save_model_transunet` | 模型保存路径 |
| `--img_size` | `256` | 输入图像尺寸 |
| `--batch_size` | `4` | 批次大小 |
| `--max_epochs` | `50` | 最大训练轮数 |
| `--base_lr` | `0.0001` | 初始学习率 |
| `--early_stop_patience` | `15` | Early stopping耐心值 |
| `--num_classes` | `2` | 类别数（背景+血管） |
| `--vit_name` | `R50-ViT-B_16` | ViT模型类型 |
| `--n_skip` | `3` | 跳跃连接数量 |

### 5. 训练输出

训练过程中会生成以下文件：

```
save_model_transunet/
├── checkpoints/
│   └── run_best.pth              # 最佳模型（基于验证集mIoU）
├── log/                          # TensorBoard日志
├── train.log                     # 训练日志文件
└── transunet_final.pth          # 最终模型
```

### 6. 监控训练过程

#### 查看日志文件
```bash
tail -f save_model_transunet/train.log
```

#### 使用TensorBoard
```bash
tensorboard --logdir=save_model_transunet/log
```

然后在浏览器中打开 `http://localhost:6006`

### 7. 模型预测

训练完成后，使用最佳模型进行预测：

```bash
python predict_vessel.py \
    --model_path ../save_model_transunet/checkpoints/run_best.pth \
    --input_dir ../dataset/image \
    --output_dir ../predictions \
    --save_overlay
```

预测参数说明：
- `--model_path`: 训练好的模型路径
- `--input_dir`: 待预测的图像文件夹
- `--output_dir`: 预测结果保存路径
- `--save_overlay`: 是否保存叠加可视化结果（可选）

预测输出：
```
predictions/
├── frame_xxx_pred.png           # 预测掩码
└── overlay/
    └── frame_xxx_overlay.png    # 叠加可视化（红色=血管）
```

## 📊 训练日志示例

```
[Epoch 1/50] Val mIoU: 0.7234 Val Acc: 0.9456 Val Dice: 0.8123 | Train Loss: 0.234567 | LR: 0.000100 | Time: 45.23s ETA: 0:36:52
>>> New run-best at epoch 1, Val mIoU=0.7234 Val Acc=0.9456 Val Dice=0.8123

[Epoch 2/50] Val mIoU: 0.7456 Val Acc: 0.9512 Val Dice: 0.8345 | Train Loss: 0.198765 | LR: 0.000098 | Time: 44.12s ETA: 0:35:18
>>> New run-best at epoch 2, Val mIoU=0.7456 Val Acc=0.9512 Val Dice=0.8345
...
```

## 🔧 关键修改说明

相比原始TransUNet，主要做了以下适配：

1. **数据集适配**
   - 创建了 `VesselDataset` 类，支持PNG格式图像
   - 添加了丰富的数据增强（翻转、旋转、亮度调整）
   - 自动划分训练/验证/测试集

2. **模型配置**
   - 类别数从9改为2（背景+血管）
   - 输入尺寸支持256×256
   - 预训练权重路径更新

3. **训练策略**
   - 使用Adam优化器（更适合小数据集）
   - 添加Cosine学习率调度
   - 添加Early Stopping机制
   - 混合损失：0.5 * CE + 0.5 * Dice

4. **评估指标**
   - Pixel Accuracy
   - mIoU (Mean Intersection over Union)
   - Dice Coefficient

## 📈 与UNet对比

你已经训练过UNet baseline，可以对比以下指标：

| 模型 | mIoU | Dice | Pixel Acc | 参数量 |
|------|------|------|-----------|--------|
| UNet | ? | ? | ? | ~31M |
| TransUNet | ? | ? | ? | ~105M |

TransUNet的优势：
- ✅ 全局上下文建模能力（Transformer）
- ✅ 预训练权重（ImageNet-21K）
- ✅ 更强的特征提取能力

TransUNet的劣势：
- ❌ 参数量更大（3倍于UNet）
- ❌ 训练时间更长
- ❌ 需要更多数据（你的619张可能偏少）

## ⚠️ 注意事项

1. **数据量问题**
   - 你的数据集只有619张，对TransUNet来说偏小
   - 建议使用强数据增强（已添加）
   - 可能需要降低batch size避免过拟合

2. **预训练权重**
   - 确保 `imagenet21k_R50+ViT-B_16.npz` 在TransUNET目录下
   - 如果没有，训练会从头开始（不推荐）

3. **显存要求**
   - TransUNet比UNet需要更多显存
   - 如果显存不足，降低batch_size到2或1

4. **训练时间**
   - TransUNet训练速度比UNet慢约2-3倍
   - 预计每个epoch需要1-2分钟（取决于硬件）

## 🐛 常见问题

### Q1: 提示找不到预训练权重
```
A: 确保 imagenet21k_R50+ViT-B_16.npz 在 TransUNET/ 目录下
```

### Q2: CUDA out of memory
```
A: 降低batch_size，例如改为2或1
   python train_vessel.py --batch_size 2
```

### Q3: 训练速度太慢
```
A: 这是正常的，TransUNet比UNet慢2-3倍
   可以降低max_epochs或使用更小的模型
```

### Q4: 验证集指标不提升
```
A: 可能是过拟合，尝试：
   1. 增加数据增强强度
   2. 降低学习率
   3. 使用更小的batch_size
   4. 提前停止训练（已有Early Stopping）
```

## 📝 TODO

- [ ] 添加测试集评估脚本
- [ ] 添加模型对比脚本（UNet vs TransUNet）
- [ ] 添加可视化工具
- [ ] 支持多GPU训练

## 📧 联系方式

如有问题，请查看训练日志或联系开发者。

---

**祝训练顺利！🎉**
