# 文件: New_UA.py
# 说明: 包含多种SCSA改进方案，通过统一接口调用
# -------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from SCSA import SCSA  # 导入原版SCSA作为基准


class MultiScaleFeatureEnhancement(nn.Module):
    """多尺度特征增强 - 针对皮肤病变多尺度特性"""
    def __init__(self, in_ch, scales=[1, 2, 3]):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList()
        
        for scale in scales:
            if scale == 1:
                conv = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)
            else:
                conv = nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
                    nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
                )
            self.convs.append(conv)
            
        self.fusion = nn.Conv2d(in_ch * (len(scales) + 1), in_ch, 1)
        self.norm = nn.BatchNorm2d(in_ch)
        self.act = nn.GELU()
        
    def forward(self, x):
        features = [x]
        for conv in self.convs:
            feat = conv(x)
            if feat.shape[2:] != x.shape[2:]:
                feat = F.interpolate(feat, size=x.shape[2:], mode='bilinear', align_corners=False)
            features.append(feat)
            
        fused = self.fusion(torch.cat(features, dim=1))
        return self.act(self.norm(fused))

class BoundaryAwareModule(nn.Module):
    """边界感知模块 - 强调病变边界信息"""
    def __init__(self, in_ch):
        super().__init__()
        self.boundary_conv = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.boundary_norm = nn.BatchNorm2d(in_ch)
        self.attention = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 4, in_ch, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        boundary = torch.abs(self.boundary_conv(x) - x)
        boundary = self.boundary_norm(boundary)
        attention_weights = self.attention(torch.cat([x, boundary], dim=1))
        return x * (1 + attention_weights)


class SCSA_Enhanced(SCSA):
    """方案6: 综合增强SCSA（多尺度+边界感知）"""
    def __init__(self, in_ch: int, num_semantic: int = 32, reduction: int = 4, topk: int = 8, 
                 use_multiscale: bool = True, 
                 use_boundary_aware: bool = True, **kwargs):
        super().__init__(in_ch, num_semantic, reduction, topk)
        
        # 可选的增强模块
        self.modules_list = nn.ModuleList()
 
        if use_multiscale:
            self.multiscale = MultiScaleFeatureEnhancement(in_ch)
            self.modules_list.append(self.multiscale)
            
        if use_boundary_aware:
            self.boundary_aware = BoundaryAwareModule(in_ch)
            self.modules_list.append(self.boundary_aware)
        
    def forward(self, x: torch.Tensor, sem_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 依次应用增强模块
        for module in self.modules_list:
            x = module(x)
        
        return super().forward(x, sem_map)

# ==================== 统一接口函数 ====================

def get_scsa_attention(attention_type: str, in_ch: int, **kwargs):

    attention_map = {
        'original': SCSA,
        'enhanced': SCSA_Enhanced
    }

    if attention_type not in attention_map:
        raise ValueError(f"不支持的注意力类型: {attention_type}。可选值: {list(attention_map.keys())}")
    
    attention_class = attention_map[attention_type]
    return attention_class(in_ch=in_ch, **kwargs)
