import base64
import json
import os
import os.path as osp
import numpy as np
import PIL.Image
from labelme import utils

# jpgs_path = r"D:\zy\NailFold\nailFolder\save\2" # 保存成 jpg 的文件夹
root_path = r"D:\zy\NailFold\nailFolder\save\data_process\3"
pngs_path = root_path + "\mask" # mask 保存文件夹
json_path = root_path + "\json" # json文件夹

for fname in os.listdir(json_path):
    if not fname.endswith(".json"):
        continue

    data = json.load(open(osp.join(json_path, fname), "r"))

    # 读取图像
    if data["imageData"]:
        imageData = data["imageData"]
    else:
        with open(osp.join(json_path, data["imagePath"]), "rb") as f:
            imageData = base64.b64encode(f.read()).decode("utf-8")

    img = utils.img_b64_to_arr(imageData)

    # Label 定义
    label_name_to_value = {
        "_background_": 0,
        "0": 1
    }

    # 🌟 正确解包！
    lbl, _ = utils.shapes_to_label(img.shape, data["shapes"], label_name_to_value)

    # 🌟 最安全：直接把 lbl==1 变成 255
    mask = (lbl == 1).astype(np.uint8) * 255

    basename = fname.replace(".json", "")
    PIL.Image.fromarray(mask).save(osp.join(pngs_path, basename + ".png"))

    # 保存原图（可选）
    # PIL.Image.fromarray(img).save(osp.join(jpgs_path, basename + ".jpg"))

    print("Saved:", basename)
