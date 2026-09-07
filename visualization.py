import open3d as o3d
import sys
import os

def visualize_point_cloud(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        sys.exit(1)

    print(f"Loading point cloud from: {file_path}")
    
    # Read the PLY file
    pcd = o3d.io.read_point_cloud(file_path)

    # Check if the point cloud is empty
    if pcd.is_empty():
        print("Error: The point cloud is empty or the file format is invalid.")
        sys.exit(1)

    # Print basic information about the point cloud
    print(pcd)
    print(f"Number of points: {len(pcd.points)}")

    # Visualize the point cloud
    print("Opening visualization window... (Close the window to exit the script)")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="3D Point Cloud Viewer",
        width=1024,
        height=768,
        left=50,
        top=50,
        point_show_normal=False # Set to True if your .ply has normals and you want to see them
    )

if __name__ == "__main__":
    # Replace this string with the path to your .ply file
    # ply_file_path = "./room_cylindrical.ply"  # approach1.py visulization
    ply_file_path = "./room_zoedepth.ply"  # approach2.py visualization
   
    
    visualize_point_cloud(ply_file_path)