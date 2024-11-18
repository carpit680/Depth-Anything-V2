import argparse
import cv2
import glob
import numpy as np
import open3d as o3d
import os
from PIL import Image
import torch

from depth_anything_v2.dpt import DepthAnythingV2
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def create_comparison_cube():
    """
    Create a 50x50x50 mm cube for size comparison.
    The cube is positioned in front of the origin (Z > 0).
    """
    cube = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.05, depth=0.05)
    cube.paint_uniform_color([1, 0, 0])  # Red color for visibility
    cube.translate((0, 0, 0.5))  # Position the cube in front of the origin
    return cube

def create_coordinate_axes():
    """
    Create a coordinate frame with X, Y, Z axes for reference.
    """
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    return axes

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate depth maps and point clouds from images.')
    parser.add_argument('--encoder', default='vitl', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder to use.')
    parser.add_argument('--load-from', default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', type=str, required=False,
                        help='Path to the pre-trained model weights.')
    parser.add_argument('--max-depth', default=10, type=float,
                        help='Maximum depth value for the depth map.')
    parser.add_argument('--vid-path', default='test.mp4', type=str, required=False,
                        help='Path to the input video.')
    parser.add_argument('--focal-length-x', default=413.08, type=float,
                        help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', default=413.08, type=float,
                        help='Focal length along the y-axis.')

    args = parser.parse_args()

    # Determine the device to use (CUDA, MPS, or CPU)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Model configuration based on the chosen encoder
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Initialize the DepthAnythingV2 model with the specified configuration
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Create the comparison cube
    comparison_cube = create_comparison_cube()

    # Create coordinate axes
    coordinate_axes = create_coordinate_axes()

    cap = cv2.VideoCapture(args.vid_path)

    # Open3D Visualizer for real-time updates
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Real-Time Point Cloud Visualization")

    # Add the comparison cube and coordinate axes
    vis.add_geometry(comparison_cube)
    vis.add_geometry(coordinate_axes)

    # Placeholder for the point cloud
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    height, width = 518, 691
    # Generate mesh grid and calculate point cloud coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - width / 2) / args.focal_length_x
    y = (y - height / 2) / args.focal_length_y
    # Process each video frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pred = depth_anything.infer_image(frame, height)

        z = np.array(pred)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = np.array(frame).reshape(-1, 3) / 255.0

        # Update the point cloud
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Update the visualizer
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    cap.release()
    vis.destroy_window()

if __name__ == '__main__':
    main()
