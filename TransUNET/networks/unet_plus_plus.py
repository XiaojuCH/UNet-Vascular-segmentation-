import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 1. 定义 SCSA (scSE) 注意力模块 ====================
class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SCSEModule, self).__init__()
        
        # --- Channel Squeeze and Excitation (cSE) ---
        # 关注"什么"特征是重要的
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 确保缩减后的通道数至少为1
        reduced_channels = max(1, in_channels // reduction)
        
        self.channel_excitation = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # --- Spatial Squeeze and Excitation (sSE) ---
        # 关注"哪里"是重要的（抑制背景）
        self.spatial_excitation = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # cSE 分支: 给不同通道加权
        chn_se = self.avg_pool(x)
        chn_se = self.channel_excitation(chn_se)
        chn_se = x * chn_se
        
        # sSE 分支: 给不同空间位置加权
        spa_se = self.spatial_excitation(x)
        spa_se = x * spa_se
        
        # 融合: 并行相加 (Concurrent Addition)
        return chn_se + spa_se


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_scsa=True):
        super(ConvBlock, self).__init__()
        
        # 基础卷积块
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # 如果启用 SCSA，则初始化该模块
        self.use_scsa = use_scsa
        if self.use_scsa:
            self.scsa = SCSEModule(out_ch)

    def forward(self, x):
        x = self.conv(x)
        
        # 在卷积特征提取后，应用注意力
        if self.use_scsa:
            x = self.scsa(x)
            
        return x

# ==================== 3. UNet++ 主体 ====================
class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes=2, input_channels=3, deep_supervision=False):
        super(UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision
        # 滤波器通道数配置
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ----------------- Encoder (Backbone) -----------------
        # L0
        self.conv0_0 = ConvBlock(input_channels, nb_filter[0], use_scsa=True)
        # L1
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1], use_scsa=True)
        # L2
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2], use_scsa=True)
        # L3
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3], use_scsa=True)
        # L4
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4], use_scsa=True)

        # ----------------- Nested Skip Pathways -----------------
        # L0 层级的跳跃连接
        self.conv0_1 = ConvBlock(nb_filter[0]+nb_filter[1], nb_filter[0], use_scsa=True)
        self.conv0_2 = ConvBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], use_scsa=True)
        self.conv0_3 = ConvBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], use_scsa=True)
        self.conv0_4 = ConvBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], use_scsa=True)

        # L1 层级的跳跃连接
        self.conv1_1 = ConvBlock(nb_filter[1]+nb_filter[2], nb_filter[1], use_scsa=True)
        self.conv1_2 = ConvBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], use_scsa=True)
        self.conv1_3 = ConvBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], use_scsa=True)

        # L2 层级的跳跃连接
        self.conv2_1 = ConvBlock(nb_filter[2]+nb_filter[3], nb_filter[2], use_scsa=True)
        self.conv2_2 = ConvBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], use_scsa=True)

        # L3 层级的跳跃连接
        self.conv3_1 = ConvBlock(nb_filter[3]+nb_filter[4], nb_filter[3], use_scsa=True)

        # ----------------- Output -----------------
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Backbone (下采样路径)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Nested Skip Pathways (密集的跳跃连接路径)
        # 每一层都在融合来自左边和下边的特征
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # 输出
        output = self.final(x0_4)
        
        return output