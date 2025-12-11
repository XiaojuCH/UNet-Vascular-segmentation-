import os

# ==== 配置区 ====
yolo_txt_folder = r"D:\zy\nailFolder\nailFold1"  # YOLO txt 文件夹
classes_file = r"D:\zy\nailFolder\nailFold1\classes.txt"   # 类别文件

# ==== 读取类别列表 ====
with open(classes_file, 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f.readlines()]

max_index = len(classes) - 1
print(f"类别总数: {len(classes)}，最大索引: {max_index}")

# ==== 遍历 YOLO txt 文件 ====
error_files = []

for fname in os.listdir(yolo_txt_folder):
    if not fname.endswith(".txt"):
        continue
    file_path = os.path.join(yolo_txt_folder, fname)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():  # 跳过空行
                continue
            parts = line.strip().split()
            try:
                class_index = int(parts[0])
                if class_index < 0 or class_index > max_index:
                    print(f"错误: {fname} 第 {line_num} 行, 类别索引 {class_index} 超出范围")
                    error_files.append(fname)
            except ValueError:
                print(f"错误: {fname} 第 {line_num} 行, 类别索引不是整数: {parts[0]}")
                error_files.append(fname)

if not error_files:
    print("检查完成，所有类别索引正常 ✅")
else:
    print(f"共 {len(set(error_files))} 个文件存在问题 ❌")
