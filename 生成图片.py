import cv2
import os

# ===========================
video_name = "WIN_20251018_11_30_56_Pro"
video_root = r"F:\workImage\video\\"
save_root = r"D:\zy\NailFold\nailFolder\11111_save\\"
video_path = video_root + video_name + ".mp4"    # 输入视频
save_dir   = save_root + video_name      # 输出图片文件夹
prefix     = "frame"                  # 图片前缀
ext        = "jpg"                    # jpg/png 都行
# ===========================

os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

print("视频打开状态:", cap.isOpened())
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("视频总帧数:", total)

frame_id = 0
while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("结束读取")
        break

    save_path = os.path.join(save_dir, f"frame_{video_name}_{frame_id:d}.jpg")
    cv2.imwrite(save_path, frame)

    frame_id += 1

cap.release()
print("完成")