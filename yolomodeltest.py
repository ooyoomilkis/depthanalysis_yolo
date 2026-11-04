import sys
import cv2
import numpy as np
import json
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# 1. Get the image path from the command line, or open a file picker
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    from tkinter import Tk, filedialog
    Tk().withdraw()
    path = filedialog.askopenfilename(
        title="Choose an image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")],
    )

if not path:
    sys.exit("No image selected.")

# 2. Load models (device=-1 is CPU, use device=0 for an NVIDIA GPU)
depth_pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=-1,
)
yolo = YOLO("yolo11n.pt")

#3. running depth estimation
img = Image.open(path).convert("RGB")
depth = np.array(depth_pipe(img)["depth"])
depth_u8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite("depth.png", cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO))

# 4. YOLO + annotated image
frame = cv2.imread(path)
results = yolo(frame)[0]
ann = Annotator(frame.copy(), line_width=1, font_size=5)

for box in results.boxes:
    cls = int(box.cls)
    conf = float(box.conf)
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    rel_depth = float(np.median(depth[y1:y2, x1:x2]))
    print(f"{yolo.names[cls]}: relative depth {rel_depth:.1f}")
    ann.box_label(
        (x1, y1, x2, y2),
        f"{yolo.names[cls]} {conf:.2f} d={rel_depth:.0f}",
        color=colors(cls, True),
    )

cv2.imwrite("detections.png", ann.result())

#adding spatial awareness (not fully accurate without real-world coordinates)
H, W = frame.shape[:2]
objs = []
for i, box in enumerate(results.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    objs.append({
        "id": i,
        "label": yolo.names[int(box.cls)],
        "box": (x1, y1, x2, y2),
        "depth": float(np.median(depth[y1:y2, x1:x2])),  # higher = closer
        "h_pos": "left" if cx < W/3 else "right" if cx > 2*W/3 else "center",
        "v_pos": "top" if cy < H/3 else "bottom" if cy > 2*H/3 else "middle",
    })

relations = []
for a in objs:
    for b in objs:
        if a["id"] >= b["id"]:
            continue
        side = "left of" if (a["box"][0] + a["box"][2]) < (b["box"][0] + b["box"][2]) else "right of"
        depth_rel = "closer than" if a["depth"] > b["depth"] else "farther than"
        relations.append(f'{a["label"]}#{a["id"]} is {side} and {depth_rel} {b["label"]}#{b["id"]}')

for o in objs: print(o)
for r in relations: print(r)

print("Saved depth.png and detections.png")

# PART 2- Detailed depth analysis
d = depth.astype(np.float32)
dn = (d - d.min()) / (d.max() - d.min() + 1e-6)   # 0 = farthest, 1 = closest

# Slope map: how fast depth changes per pixel (high = edge or tilted surface)
gy, gx = np.gradient(dn)
slope = np.hypot(gx, gy)

# --- Global stats and depth zones ---
analysis = {
    "global": {
        "mean": float(dn.mean()),
        "median": float(np.median(dn)),
        "std": float(dn.std()),
        "foreground_pct": float((dn > 0.66).mean() * 100),
        "midground_pct": float(((dn > 0.33) & (dn <= 0.66)).mean() * 100),
        "background_pct": float((dn <= 0.33).mean() * 100),
    },
    "objects": [],
}

# --- Per-object stats ---
for o in objs:
    x1, y1, x2, y2 = o["box"]
    crop = dn[y1:y2, x1:x2]
    if crop.size == 0:
        continue
    p5, p95 = np.percentile(crop, [5, 95])
    mean_slope = float(slope[y1:y2, x1:x2].mean())
    analysis["objects"].append({
        "id": o["id"],
        "label": o["label"],
        "median_depth": float(np.median(crop)),
        "min_depth": float(crop.min()),
        "max_depth": float(crop.max()),
        "depth_spread": float(p95 - p5),            # how much it extends in depth
        "surface": "flat" if mean_slope < 0.01 else "sloped/curved",
        "area_pct": float((x2 - x1) * (y2 - y1) / (H * W) * 100),
        "zone": "foreground" if np.median(crop) > 0.66
                else "midground" if np.median(crop) > 0.33 else "background",
    })

# Rank objects from closest to farthest
analysis["objects"].sort(key=lambda r: r["median_depth"], reverse=True)
for rank, r in enumerate(analysis["objects"], 1):
    r["closeness_rank"] = rank

# --- 4x4 region grid --- (also can try different sizes like 3x3, 5x5, etc.)
N = 4
analysis["grid"] = []
for row in range(N):
    for col in range(N):
        cell = dn[row*H//N:(row+1)*H//N, col*W//N:(col+1)*W//N]
        analysis["grid"].append({
            "row": row, "col": col, "mean_depth": float(cell.mean()),
        })

# --- Depth edge map (where depth changes sharply) ---
edges = (np.clip(slope / (slope.max() + 1e-6), 0, 1) * 255).astype(np.uint8)
cv2.imwrite("depth_edges.png", cv2.applyColorMap(edges, cv2.COLORMAP_MAGMA))

with open("depth_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)

g = analysis["global"]
print("\nDEPTH ANALYSIS")
print(f'Foreground {g["foreground_pct"]:.0f}% | Midground {g["midground_pct"]:.0f}% | Background {g["background_pct"]:.0f}%')
for r in analysis["objects"]:
    print(f'#{r["closeness_rank"]} {r["label"]}#{r["id"]}: depth {r["median_depth"]:.2f} '
          f'({r["zone"]}), spread {r["depth_spread"]:.2f}, {r["surface"]}, {r["area_pct"]:.1f}% of image')
print("Saved depth_analysis.json and depth_edges.png")
