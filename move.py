import os
import shutil
import random

# -------- CONFIG --------
IMAGES_DIR = r"D:\Programming\Python\Machine Learning\RaceCar YOLO\25dec\frames_manual"
LABELS_DIR = r"D:\Programming\Python\Machine Learning\RaceCar YOLO\labeled"
OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.95
IMAGE_EXT = ".jpg"
# ------------------------

# Create output directories
for split in ["train", "test"]:
    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

# Collect all images
images = [
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith(IMAGE_EXT)
]

if not images:
    raise RuntimeError("❌ No images found in frames_manual/")

# Shuffle & split
random.shuffle(images)
split_idx = int(len(images) * TRAIN_RATIO)

train_images = images[:split_idx]
test_images  = images[split_idx:]

def move_pair(filename, split):
    name = os.path.splitext(filename)[0]

    src_img = os.path.join(IMAGES_DIR, filename)
    dst_img = f"{OUTPUT_DIR}/images/{split}/{filename}"
    shutil.copy(src_img, dst_img)

    label_file = f"{name}.txt"
    src_label = os.path.join(LABELS_DIR, label_file)

    if os.path.exists(src_label):
        dst_label = f"{OUTPUT_DIR}/labels/{split}/{label_file}"
        shutil.copy(src_label, dst_label)

# Process splits
for img in train_images:
    move_pair(img, "train")

for img in test_images:
    move_pair(img, "test")

print("✅ Dataset split complete")
print(f"Train images: {len(train_images)}")
print(f"Test images : {len(test_images)}")
