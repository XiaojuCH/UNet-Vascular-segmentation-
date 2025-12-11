#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
from pathlib import Path

# ========== 用户只需要改这里 ==========
FOLDER_1 = Path(r"D:\zy\NailFold\nailFolder\save\3")   # 原始混杂文件夹
ROOT_DIR   = Path(r"D:\zy\NailFold\nailFolder\save\data_process")            # 根目录
# =====================================

# 目标结构：ROOT_DIR / mixed_folder / {image,json,mask}
new_folder = ROOT_DIR / FOLDER_1.name
(new_folder / "image").mkdir(parents=True, exist_ok=True)
(new_folder / "json").mkdir(parents=True, exist_ok=True)
(new_folder / "mask").mkdir(parents=True, exist_ok=True)

# 常见图片后缀（可按需增删）
IMG_SUFFIX = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif'}

for item in FOLDER_1.iterdir():
    if item.is_file():
        if item.suffix.lower() == '.json':
            shutil.move(str(item), str(new_folder / "json"))
        elif item.suffix.lower() in IMG_SUFFIX:
            shutil.move(str(item), str(new_folder / "image"))
        # 其余文件不动，也可按需加 elif 分支

print("完成！目标目录：", new_folder)