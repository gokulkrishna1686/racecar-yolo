import os
import shutil

SOURCE_ROOT = r"25dec/frames"
DEST_DIR = r"frames_unlabeled"

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

os.makedirs(DEST_DIR, exist_ok=True)

count = 0

for subfolder in ["blue", "both", "red", "sky"]:
    src_dir = os.path.join(SOURCE_ROOT, subfolder)

    if not os.path.isdir(src_dir):
        print(f"⚠️ Skipping missing folder: {src_dir}")
        continue

    for filename in os.listdir(src_dir):
        if filename.lower().endswith(IMAGE_EXTS):
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(DEST_DIR, filename)

            if os.path.exists(dst_path):
                print(f"⚠️ Duplicate filename skipped: {filename}")
                continue

            shutil.copy(src_path, dst_path)
            count += 1

print(f"✅ Done. Copied {count} images to '{DEST_DIR}'.")
