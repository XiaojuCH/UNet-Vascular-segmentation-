"""
交互式裁剪工具 (OpenCV)
功能：
- 在预览图上按 W 创建一个 256x256 的裁剪框（默认在鼠标位置或图像中心）
- 鼠标按住并拖动可以移动选中框（固定大小 256x256）
- 每按一次 W 生成一个独立的框，最多 20 个
- 按 Q 对文件夹内的所有图片进行“连续裁切”：以每个框的位置裁切并保存到输出文件夹
使用方法：
1. 修改下方 IMAGE_FOLDER 和 SAVE_FOLDER 为你的输入/输出路径（支持相对或绝对路径）
2. 运行脚本：python batch_crop_draggable.py
3. 在弹出的窗口里：
   - 用鼠标在图上移动，当按 W 时会在当前鼠标位置生成一个 256x256 框（如果超出边界会自动修正）
   - 点击框并拖动来移动（按住鼠标左键拖动）
   - 按数字键 1-9 / 0 选择对应索引的框（方便切换），或通过点击来选中框
   - 按 W 可继续添加（最多 20 个）
   - 按 Q 会开始对 IMAGE_FOLDER 中所有图片进行裁切，并把裁切结果保存到 SAVE_FOLDER，每张原图会生成 N 个裁切图（N = 框数量）
   - 按 ESC 或关闭窗口退出（未按 Q 不会保存裁切结果）
输出命名规则： <origname>__box<idx>.png
"""
import cv2
import os
import glob
import sys
import numpy as np
from PIL import Image

# ==== 可选 tif 支持 ====
try:
    import tifffile
    USE_TIFFILE = True
except:
    USE_TIFFILE = False

# ===================== 配置区 =====================
video_name = "WIN_20251018_11_30_56_Pro"
image_root = r"D:\zy\NailFold\nailFolder\11111_save\\"
save_root = r"D:\zy\NailFold\nailFolder\save\org_img\\"
IMAGE_FOLDER = image_root + video_name
SAVE_FOLDER = save_root + video_name + "_save"

MAX_IMAGES = 100
MAX_BOXES = 20
BOX_SIZE = 256
MAX_DISPLAY = 1200  # 窗口显示最大宽或高，超大图会缩放显示
# ==================================================

# ------------ 图像读取函数 ----------------
def load_image_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tif", ".tiff"]:
        try:
            if USE_TIFFILE:
                img = tifffile.imread(path)
            else:
                img = np.array(Image.open(path))
            # 如果是 RGB，转换为 BGR
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = img[:, :, ::-1]
            return img
        except Exception as e:
            print("TIFF 读取失败：", path, "错误：", e)
            return None
    img = cv2.imread(path)
    return img

# ------------ 初始化 ----------------
if not os.path.isdir(IMAGE_FOLDER):
    print("错误：IMAGE_FOLDER 不存在：", IMAGE_FOLDER)
    sys.exit(1)

os.makedirs(SAVE_FOLDER, exist_ok=True)

exts = ('*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.tiff')
image_list = []
for e in exts:
    image_list.extend(sorted(glob.glob(os.path.join(IMAGE_FOLDER, e))))

if not image_list:
    print("错误：输入文件夹内没有图片：", IMAGE_FOLDER)
    sys.exit(1)

# 预览图
preview_path = image_list[0]
orig = load_image_any(preview_path)
if orig is None:
    print("无法读取预览图：", preview_path)
    sys.exit(1)

# 如果灰度转 BGR
if len(orig.shape) == 2:
    orig = cv2.cvtColor(orig.astype(np.uint8), cv2.COLOR_GRAY2BGR)

if orig.dtype != np.uint8:
    orig = cv2.normalize(orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

img_h, img_w = orig.shape[:2]

# ====== 窗口缩放比例 ======
scale = 1.0
if max(img_w, img_h) > MAX_DISPLAY:
    scale = min(MAX_DISPLAY / img_w, MAX_DISPLAY / img_h)
display_w, display_h = int(img_w * scale), int(img_h * scale)

win_name = "CropDesigner - W:add box  Q:batch crop  ESC:exit"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, display_w, display_h)

boxes = []
selected_idx = -1
dragging = False
drag_offset = (0,0)
mouse_pos = (img_w//2, img_h//2)

def clamp_box(x,y):
    x = int(x); y=int(y)
    if x < 0: x = 0
    if y < 0: y = 0
    if x + BOX_SIZE > img_w: x = img_w - BOX_SIZE
    if y + BOX_SIZE > img_h: y = img_h - BOX_SIZE
    return x,y

def draw_overlay(frame):
    disp = cv2.resize(frame, (display_w, display_h)) if scale != 1 else frame.copy()
    overlay = disp.copy()
    mask = np.zeros_like(disp)
    alpha = 0.3
    cv2.addWeighted(mask, alpha, overlay, 1-alpha, 0, overlay)

    for i, b in enumerate(boxes):
        x, y = int(b['x']*scale), int(b['y']*scale)
        w, h = int(b['w']*scale), int(b['h']*scale)
        thickness = 2 if i == selected_idx else 1
        color = (0,255,0) if i == selected_idx else (180,200,0)
        cv2.rectangle(overlay, (x,y), (x+w-1,y+h-1), color, thickness)

        label = f"{i+1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6*scale, 2)
        cv2.rectangle(overlay, (x,y-22), (x+tw+6,y), (50,50,50), -1)
        cv2.putText(overlay, label, (x+3,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6*scale, (255,255,255), 1, cv2.LINE_AA)

    # 信息提示
    info = [
        "W: Create rectangle   mouseleft: Drag rectangle   key number: Select rectangle",
        f"Number: {len(boxes)}/{MAX_BOXES}",
        "Q: Batch cut   ESC: Exit"
    ]
    y0 = 20
    for line in info:
        cv2.putText(overlay, line, (10,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5*scale, (220,220,220), 1, cv2.LINE_AA)
        y0 += int(18*scale)
    return overlay

def mouse_cb(event, x, y, flags, param):
    global dragging, selected_idx, drag_offset, mouse_pos
    # 显示坐标 → 原图坐标
    ox, oy = int(x/scale), int(y/scale)
    mouse_pos = (ox, oy)
    if event == cv2.EVENT_LBUTTONDOWN:
        found = -1
        for i in reversed(range(len(boxes))):
            b = boxes[i]
            if b['x'] <= ox < b['x'] + BOX_SIZE and b['y'] <= oy < b['y'] + BOX_SIZE:
                found = i
                break
        if found >=0:
            selected_idx = found
            dragging = True
            drag_offset = (ox - boxes[selected_idx]['x'], oy - boxes[selected_idx]['y'])
        else:
            selected_idx = -1
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and selected_idx >=0:
            nx = ox - drag_offset[0]
            ny = oy - drag_offset[1]
            nx, ny = clamp_box(nx, ny)
            boxes[selected_idx]['x'] = nx
            boxes[selected_idx]['y'] = ny
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

cv2.setMouseCallback(win_name, mouse_cb)

# ================== 主循环 ==================
while True:
    frame = draw_overlay(orig)
    cv2.imshow(win_name, frame)
    key = cv2.waitKey(20) & 0xFF

    if key == 27:  # ESC
        break
    elif key in [ord('w'), ord('W')]:
        if len(boxes) >= MAX_BOXES:
            print("达到最大框数量", MAX_BOXES)
            continue
        mx, my = mouse_pos
        nx, ny = clamp_box(mx - BOX_SIZE//2, my - BOX_SIZE//2)
        boxes.append({'x':nx,'y':ny,'w':BOX_SIZE,'h':BOX_SIZE})
        selected_idx = len(boxes)-1
        print(f"新增框 #{selected_idx+1} at ({nx},{ny})")
    elif key in [ord('q'), ord('Q')]:
        if not boxes:
            print("没有框，跳过裁切")
            boxes = []
        cv2.destroyAllWindows()
        break  # 退出窗口，开始裁切

# ===== 批量裁切 =====
if boxes:
    print("开始批量裁切...")
    total = 0
    print(f"图片总数：{len(image_list)}, 实际裁切数量：{min(len(image_list), MAX_IMAGES)}, 框总数：{len(boxes)}")

    # ★★★ 只裁前 100 张 ★★★
    for img_path in image_list[:MAX_IMAGES]:
        img = load_image_any(img_path)
        if img is None:
            print("无法读取：", img_path)
            continue

        base = os.path.splitext(os.path.basename(img_path))[0]

        for i, b in enumerate(boxes):
            x, y = b['x'], b['y']
            crop = img[y:y + BOX_SIZE, x:x + BOX_SIZE]

            if crop.shape[0] != BOX_SIZE or crop.shape[1] != BOX_SIZE:
                crop = cv2.copyMakeBorder(
                    crop, 0, BOX_SIZE - crop.shape[0], 0, BOX_SIZE - crop.shape[1],
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )

            out_path = os.path.join(SAVE_FOLDER, f"{base}_{video_name}_box{i + 1}.png")
            success = cv2.imwrite(out_path, crop)

            if success:
                total += 1
            else:
                print(f"裁切失败")

    print(f"\n批量裁切完成，共生成 {total} 张图 → {SAVE_FOLDER}")
else:
    print("没有框，未执行裁切。")
