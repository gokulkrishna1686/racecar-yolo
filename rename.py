import os
import re

# CHANGE THIS to your folder path
FOLDER_PATH = r"D:\Programming\Python\Machine Learning\RaceCar YOLO\25dec"   # e.g. r"C:\Users\Gokul\Desktop\dataset"

# Mapping from detected prefix -> clean prefix
PREFIX_MAP = {
    "red": "red_car",
    "blue": "blue_car",
    "both": "both_cars",
    "sky": "sky",
    "test": "test"
}

# Collect files by category
files_by_prefix = {k: [] for k in PREFIX_MAP}

for filename in os.listdir(FOLDER_PATH):
    if not os.path.isfile(os.path.join(FOLDER_PATH, filename)):
        continue

    name, ext = os.path.splitext(filename)
    name_lower = name.lower()

    for key in PREFIX_MAP:
        if name_lower.startswith(key):
            files_by_prefix[key].append(filename)
            break

# Rename files
for key, files in files_by_prefix.items():
    files.sort()  # consistent ordering
    clean_prefix = PREFIX_MAP[key]

    for idx, old_name in enumerate(files, start=1):
        ext = os.path.splitext(old_name)[1]
        new_name = f"{clean_prefix}_{idx:02d}{ext}"

        old_path = os.path.join(FOLDER_PATH, old_name)
        new_path = os.path.join(FOLDER_PATH, new_name)

        print(f"{old_name}  -->  {new_name}")
        os.rename(old_path, new_path)

print("✅ Renaming complete.")
