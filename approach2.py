import glob
import os
import subprocess
import sys
import cv2
import numpy as np

# Optional imports handled gracefully
try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from plyfile import PlyData, PlyElement

    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False

from PIL import Image
from scipy.spatial import cKDTree
import torch
from tqdm import tqdm

# 1. Device Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

video_path = "sample.mp4"
fps = 2

# Clean up previous frame files
for f in glob.glob("frame_*.jpg"):
    try:
        os.remove(f)
    except OSError:
        pass

# 2. Extract frames using ffmpeg via subprocess
print("Extracting video frames...")
ffmpeg_cmd = [
    "ffmpeg",
    "-i",
    video_path,
    "-vf",
    f"fps={fps}",
    "-qscale:v",
    "2",
    "frame_%03d.jpg",
    "-hide_banner",
    "-loglevel",
    "error",
]

try:
    subprocess.run(ffmpeg_cmd, check=True)
except (subprocess.CalledProcessError, FileNotFoundError):
    print(
        "Error running ffmpeg. Ensure ffmpeg is installed and added to your system PATH."
    )
    sys.exit(1)

frame_files = sorted(glob.glob("frame_*.jpg"))
print(f"{len(frame_files)} frames extracted.")

if not frame_files:
    raise FileNotFoundError("No frames were extracted. Check video_path.")

images = []
for f in tqdm(frame_files, desc="Loading frames"):
    img = cv2.imread(f)
    if img is not None:
        images.append(img)

# 3. Stitch Panorama
print("Stitching panorama, this can take a bit...")
stitcher = cv2.Stitcher_create()
status, stitched = stitcher.stitch(images)

if status != cv2.Stitcher_OK:
    raise RuntimeError(
        f"Stitching failed with status code {status}. Frames probably don't overlap enough."
    )

# Crop black borders left after stitching
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

# 4. Load ZoeDepth Model
print(
    "Loading ZoeDepth (Indoor Metric Depth Model)... First run downloads weights (~1.4GB)."
)
zoe = (
    torch.hub.load(
        "isl-org/ZoeDepth", "ZoeD_N", pretrained=True, trust_repo=True
    )
    .to(device)
    .eval()
)
print("Model loaded.")

# 5. Run Metric Depth Estimation
print("Running Metric Depth Estimation (Meters)...")
with torch.no_grad():
    depth_numpy = zoe.infer_pil(pano_img)

# Cast to float32 to prevent cv2.bilateralFilter datatype error
depth_numpy = depth_numpy.astype(np.float32)
depth_map = cv2.bilateralFilter(
    depth_numpy, d=9, sigmaColor=0.5, sigmaSpace=15
)

# 6. Sampling & Coordinate Projection
STRIDE = 2  # Set to 1 for full resolution, or 3-4 for lighter point cloud

u, v = np.meshgrid(np.arange(0, width, STRIDE), np.arange(0, height, STRIDE))
depth_sampled = depth_map[::STRIDE, ::STRIDE]
color_sampled = np.array(pano_img)[::STRIDE, ::STRIDE]

fov_x = 120.0 * (np.pi / 180.0)
fov_y = fov_x * (height / width)

theta = (u / width - 0.5) * fov_x
phi = (v / height - 0.5) * fov_y

# Spherical -> Cartesian
X = depth_sampled * np.sin(theta) * np.cos(phi)
Y = -depth_sampled * np.sin(phi)
Z = depth_sampled * np.cos(theta) * np.cos(phi)

points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
colors = (color_sampled / 255.0).reshape(-1, 3)
print(f"Initial point cloud: {len(points):,} points")

# 7. Outlier Removal
print("Removing statistical outliers...")
tree = cKDTree(points)
dists, _ = tree.query(points, k=31)
mean_dists = dists[:, 1:].mean(axis=1)

global_mean = mean_dists.mean()
global_std = mean_dists.std()
threshold = global_mean + 1.5 * global_std

keep_mask = mean_dists < threshold
points = points[keep_mask]
colors = colors[keep_mask]
print(f"Kept {keep_mask.sum():,} / {len(keep_mask):,} points")

# 8. Export to PLY
ply_path = "room_zoedepth.ply"
vertex_colors = (colors * 255).astype(np.uint8)

if HAS_PLYFILE:
    vertex_data = np.zeros(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    (
        vertex_data["x"],
        vertex_data["y"],
        vertex_data["z"],
    ) = (
        points[:, 0],
        points[:, 1],
        points[:, 2],
    )
    (
        vertex_data["red"],
        vertex_data["green"],
        vertex_data["blue"],
    ) = (
        vertex_colors[:, 0],
        vertex_colors[:, 1],
        vertex_colors[:, 2],
    )

    el = PlyElement.describe(vertex_data, "vertex")
    PlyData([el], text=False).write(ply_path)
else:
    # Fallback ASCII writer if plyfile package is not installed
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(ply_path, "w") as f:
        f.write(header)
        for p, c in zip(points, vertex_colors):
            f.write(
                f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n"
            )

print(f"Successfully saved point cloud to {os.path.abspath(ply_path)}")

# 9. Plotly Preview
if HAS_PLOTLY:
    print("Generating Plotly 3D preview...")
    plot_points, plot_colors = points, colors
    MAX_PLOT_POINTS = 40000

    if len(plot_points) > MAX_PLOT_POINTS:
        idx = np.random.choice(
            len(plot_points), MAX_PLOT_POINTS, replace=False
        )
        plot_points = plot_points[idx]
        plot_colors = plot_colors[idx]

    colors_str = [
        f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in plot_colors
    ]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=plot_points[:, 0],
                y=plot_points[:, 1],
                z=plot_points[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=colors_str),
            )
        ]
    )
    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=0),
        height=700,
    )
    fig.show()
else:
    print(
        "Plotly is not installed (`pip install plotly`). Skipping 3D interactive plot preview."
    )