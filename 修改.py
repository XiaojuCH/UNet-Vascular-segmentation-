import os
import json

src_label = "package"      # 想换的旧名字
dst_label = "0"            # 换成的新名字
json_dir  = r"D:\zy\NailFold\nailFolder\save\2"  # JSON 文件夹

cnt = 0
for root, _, files in os.walk(json_dir):
    for fname in files:
        if not fname.lower().endswith('.json'):
            continue
        json_path = os.path.join(root, fname)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        changed = False
        for shape in data.get('shapes', []):
            if shape.get('label') == src_label:
                shape['label'] = dst_label
                changed = True

        if changed:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            cnt += 1
            print(f'Updated: {json_path}')

print(f'\nDone! {cnt} files renamed "{src_label}" -> "{dst_label}".')