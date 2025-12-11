"""
注意力机制配置文件
根据GPU显存情况选择合适的配置
"""

# =========================
#   预设配置方案
# =========================

ATTENTION_CONFIGS = {
    # 方案1: 最轻量级（显存占用最小，速度最快）
    'lightweight': {
        'attention_type': 'original',  # 使用原版SCSA（无多尺度和边界感知）
        'use_attention_at': ['bottleneck'],  # 只在瓶颈层
        'description': '最轻量级 - 适合显存<6GB',
        'estimated_vram': '~4GB'
    },

    # 方案2: 平衡版（推荐）
    'balanced': {
        'attention_type': 'enhanced',  # 增强版SCSA
        'use_attention_at': ['bottleneck'],  # 只在瓶颈层
        'description': '平衡版 - 适合显存6-8GB（推荐）',
        'estimated_vram': '~5-6GB'
    },

    # 方案3: 性能优先
    'performance': {
        'attention_type': 'enhanced',  # 增强版SCSA
        'use_attention_at': ['bottleneck', 'decoder'],  # 瓶颈层+解码器
        'description': '性能优先 - 适合显存8-12GB',
        'estimated_vram': '~7-9GB'
    },

    # 方案4: 最强性能（显存占用最大）
    'maximum': {
        'attention_type': 'enhanced',  # 增强版SCSA
        'use_attention_at': ['encoder', 'bottleneck', 'decoder'],  # 全部使用
        'description': '最强性能 - 适合显存>12GB',
        'estimated_vram': '~10-14GB'
    }
}

# =========================
#   当前使用的配置
# =========================
# 修改这里来切换配置方案
CURRENT_CONFIG = 'balanced'  # 可选: 'lightweight', 'balanced', 'performance', 'maximum'

# =========================
#   获取配置函数
# =========================
def get_attention_config(config_name=None):
    """
    获取注意力配置

    Args:
        config_name: 配置名称，如果为None则使用CURRENT_CONFIG

    Returns:
        dict: 配置字典
    """
    if config_name is None:
        config_name = CURRENT_CONFIG

    if config_name not in ATTENTION_CONFIGS:
        raise ValueError(f"未知的配置: {config_name}。可选: {list(ATTENTION_CONFIGS.keys())}")

    return ATTENTION_CONFIGS[config_name]


def print_all_configs():
    """打印所有可用配置"""
    print("="*70)
    print("可用的注意力配置方案:")
    print("="*70)
    for name, config in ATTENTION_CONFIGS.items():
        current = " (当前)" if name == CURRENT_CONFIG else ""
        print(f"\n[{name}]{current}")
        print(f"  描述: {config['description']}")
        print(f"  注意力类型: {config['attention_type']}")
        print(f"  使用位置: {', '.join(config['use_attention_at'])}")
        print(f"  预估显存: {config['estimated_vram']}")
    print("\n" + "="*70)


if __name__ == "__main__":
    # 测试：打印所有配置
    print_all_configs()

    # 获取当前配置
    config = get_attention_config()
    print(f"\n当前配置: {CURRENT_CONFIG}")
    print(f"配置详情: {config}")
