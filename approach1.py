import glob
import os
import subprocess
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm
from transformers import pipeline

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load HuggingFace Depth Estimation Pipeline
print("Loading depth estimation model...")
pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0 if device == "cuda" else -1,
)

video_path = "sample.mp4"
fps = 2
output_dir = "approach1_extracted_frames"

# Ensure the target folder exists
os.makedirs(output_dir, exist_ok=True)

# Clean up any existing frame images inside the folder
for f in glob.glob(os.path.join(output_dir, "frame_*.jpg")):
    try:
        os.remove(f)
    except OSError:
        pass

# Define the file pattern path inside the directory
output_pattern = os.path.join(output_dir, "frame_%03d.jpg")

# Run ffmpeg via subprocess
print(f"Extracting frames into '{output_dir}'...")
ffmpeg_cmd = [
    "ffmpeg",
    "-i",
    video_path,
    "-vf",
    f"fps={fps}",
    "-qscale:v",
    "2",
    output_pattern,
    "-hide_banner",
    "-loglevel",
    "error",
]

try:
    subprocess.run(ffmpeg_cmd, check=True)
except (subprocess.CalledProcessError, FileNotFoundError):
    print(
        "Error running ffmpeg. Make sure ffmpeg is installed and added to your system PATH."
    )
    sys.exit(1)

# Retrieve extracted frame paths
frame_files = sorted(glob.glob(os.path.join(output_dir, "frame_*.jpg")))
print(f"{len(frame_files)} frames extracted.")

if not frame_files:
    raise FileNotFoundError("No frames were extracted from the video.")
images = []
for f in tqdm(frame_files, desc="Loading frames"):
    img = cv2.imread(f)
    if img is not None:
        images.append(img)

print("Stitching panorama (this can take a bit)...")
stitcher = cv2.Stitcher_create()
status, stitched = stitcher.stitch(images)

if status != cv2.Stitcher_OK:
    raise RuntimeError(
        f"Stitching failed with status code {status}. Ensure frames have sufficient overlap."
    )

# Crop out black borders left after stitching
gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

if contours:
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    pad_x, pad_y = int(w * 0.04), int(h * 0.04)
    stitched = stitched[y + pad_y : y + h - pad_y, x + pad_x : x + w - pad_x]

pano_img = Image.fromarray(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
width, height = pano_img.size

print("Running depth estimation...")
with torch.no_grad():
    depth_res = pipe(pano_img)

raw_tensor = depth_res["predicted_depth"].squeeze().cpu().numpy()
disparity_map = cv2.resize(
    raw_tensor, (width, height), interpolation=cv2.INTER_CUBIC
)

depth_map = 1.0 / (disparity_map + 1e-5)

# Clip extremes
d_min, d_max = np.percentile(depth_map, 5), np.percentile(depth_map, 95)
depth_map = np.clip(depth_map, d_min, d_max)
depth_map = (depth_map - d_min) / (d_max - d_min + 1e-5)

# Scale depth & cast to float32 (required by cv2.bilateralFilter)
depth_map = ((depth_map * 5.0) + 1.0).astype(np.float32)
depth_map = cv2.bilateralFilter(depth_map, d=9, sigmaColor=0.1, sigmaSpace=10)

# Cylindrical Projection
u, v = np.meshgrid(np.arange(width), np.arange(height))
fov_x = 140.0 * (np.pi / 180.0)
theta = (u / width - 0.5) * fov_x
focal_length = width / fov_x

X = depth_map * np.sin(theta)
Z = depth_map * np.cos(theta)
Y = -(v - height / 2.0) * (depth_map / focal_length)

points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
colors = (np.array(pano_img) / 255.0).reshape(-1, 3)

# Subsample & Outlier Filter
step = 2
points = points[::step]
colors = colors[::step]

print("Filtering outlier points...")
tree = cKDTree(points)
dists, _ = tree.query(points, k=21)
mean_dist = dists[:, 1:].mean(axis=1)

thresh = mean_dist.mean() + 2.0 * mean_dist.std()
mask = mean_dist < thresh

points = points[mask]
colors = colors[mask]
print(f"{points.shape[0]} points after cleanup.")

# Save to ASCII PLY file
colors_255 = (colors * 255).astype(np.uint8)
output_filename = "room_cylindrical.ply"

header = (
    "ply\n"
    "format ascii 1.0\n"
    f"element vertex {points.shape[0]}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "property uchar red\n"
    "property uchar green\n"
    "property uchar blue\n"
    "end_header\n"
)

with open(output_filename, "w") as f:
    f.write(header)
    for p, c in zip(points, colors_255):
        f.write(
            f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]:d} {c[1]:d} {c[2]:d}\n"
        )

print(
    f"Done! Point cloud successfully saved to: {os.path.abspath(output_filename)}"
)