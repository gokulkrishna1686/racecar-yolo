import os

LABEL_DIR = "runs/detect/predict3/labels"
CONF_THRESH = 0.4

for file in os.listdir(LABEL_DIR):
    path = os.path.join(LABEL_DIR, file)
    new_lines = []

    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:  # cls x y w h conf
                conf = float(parts[5])
                if conf >= CONF_THRESH:
                    new_lines.append(" ".join(parts[:5]))

    with open(path, "w") as f:
        f.write("\n".join(new_lines))
