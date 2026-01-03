import os
import subprocess
import re

VIDEO_DIR = r"D:\Programming\Python\Machine Learning\RaceCar YOLO\25dec"
FRAME_DIR = r"D:\Programming\Python\Machine Learning\RaceCar YOLO\25dec\frames"
FPS = 3

# Map filename keyword -> output folder
CATEGORIES = {
    "red": "red",
    "blue": "blue",
    "both": "both",
    "sky": "sky"
}

os.makedirs(FRAME_DIR, exist_ok=True)

for fname in os.listdir(VIDEO_DIR):
    if not fname.lower().endswith(".mov"):
        continue

    lower = fname.lower()
    category = None

    for key in CATEGORIES:
        if key in lower:
            category = CATEGORIES[key]
            break

    if category is None:
        print(f"⚠️ Skipping (unknown type): {fname}")
        continue

    out_dir = os.path.join(FRAME_DIR, category)
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(fname)[0]
    out_pattern = os.path.join(out_dir, f"{base_name}_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-i", os.path.join(VIDEO_DIR, fname),
        "-vf", f"fps={FPS}",
        out_pattern
    ]

    print(f"🎬 Processing {fname} → {out_dir}")
    subprocess.run(cmd, check=True)

print("✅ Frame extraction complete.")
