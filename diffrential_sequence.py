"""
================================================================================
POINT CLOUD DIFFERENCE DETECTOR - Command-Line Usage
================================================================================

This script compares two PLY point cloud files and extracts points that differ
beyond a specified distance threshold. It generates difference visualization and
statistical analysis.

USAGE:
    python diffrential_sequence.py <file1.ply> <file2.ply> [OPTIONS]

REQUIRED ARGUMENTS:
    file1.ply               First PLY point cloud file
    file2.ply               Second PLY point cloud file

OPTIONAL ARGUMENTS:
    -o, --output PATH       Output file path (default: differences.ply)
    -t, --threshold VALUE   Distance threshold for detecting differences (default: 0.001)
    -h, --help              Show help message

EXAMPLES:
    1. Basic usage with default threshold (0.001):
       python diffrential_sequence.py model1.ply model2.ply

    2. Specify custom output filename:
       python diffrential_sequence.py model1.ply model2.ply -o result_diff.ply

    3. Use custom distance threshold:
       python diffrential_sequence.py model1.ply model2.ply -t 0.005

    4. Combine custom output and threshold:
       python diffrential_sequence.py model1.ply model2.ply -o result.ply -t 0.01

    5. View help:
       python diffrential_sequence.py --help

OUTPUT FILES:
    - {output}.ply              - Point cloud with differences
    - {output}_stats.txt        - Statistical summary
    - {output}_histogram.png    - Distance distribution histogram

FEATURES:
    - Detects points that differ between two point clouds
    - Calculates distance statistics (mean, min, max)
    - Generates histogram of distance distribution
    - Shows top 10 points with largest differences
    - Saves detailed statistics to text file

NOTES:
    - Input PLY files must be in ASCII or binary format
    - Both files should have the same properties (x, y, z, r, g, b)
    - Threshold is in the same units as the point cloud coordinates
    - Smaller threshold values detect smaller differences
================================================================================
"""

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import argparse

def compare_point_clouds(ply1_path, ply2_path, output_path, threshold=0.001):
    """
    Compare two PLY point clouds and save different points
    
    Parameters:
    -----------
    ply1_path : str
        Path to the first PLY file
    ply2_path : str
        Path to the second PLY file
    output_path : str
        Path to save the differences file
    threshold : float
        Distance threshold to detect different points (default: 0.001)
    """
    
    print(f"Reading files...")
    print(f"  File 1: {ply1_path}")
    print(f"  File 2: {ply2_path}")
    
    # Read PLY files
    pcd1 = o3d.io.read_point_cloud(ply1_path)
    pcd2 = o3d.io.read_point_cloud(ply2_path)
    
    # Convert to numpy array
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    
    print(f"\nInitial statistics:")
    print(f"  Number of points in file 1: {len(points1)}")
    print(f"  Number of points in file 2: {len(points2)}")
    
    # Check dimensions
    if points1.shape[1] != 3 or points2.shape[1] != 3:
        raise ValueError("Both files must have 3D coordinates")
    
    # Build KD tree for nearest neighbor search
    print("\nBuilding KD tree for point matching search...")
    tree1 = KDTree(points1)
    
    # Find nearest points in file 1 for each point in file 2
    distances, indices_in_pcd1 = tree1.query(points2, k=1)
    
    # Detect different points based on threshold
    different_mask = distances > threshold
    
    # Count different points
    num_different = np.sum(different_mask)
    percentage = (num_different / len(points2)) * 100
    
    print(f"\nComparison results (threshold: {threshold}):")
    print(f"  Number of different points: {num_different}")
    print(f"  Percentage of different points: {percentage:.2f}%")
    print(f"  Mean distance of different points: {np.mean(distances[different_mask]):.6f}")
    print(f"  Maximum distance: {np.max(distances[different_mask]):.6f}")
    print(f"  Minimum distance of different points: {np.min(distances[different_mask]):.6f}")
    
    if num_different == 0:
        print("\n✅ No different points found! Files are identical.")
        return
    
    # Extract different points
    # Different points from file 1 (corresponding points)
    different_from_pcd1 = points1[indices_in_pcd1[different_mask]]
    
    # Different points from file 2
    different_from_pcd2 = points2[different_mask]
    
    # Combine all different points
    all_different_points = np.vstack([different_from_pcd1, different_from_pcd2])
    
    # Create new PLY file
    pcd_diff = o3d.geometry.PointCloud()
    pcd_diff.points = o3d.utility.Vector3dVector(all_different_points)
    
    # Save file
    print(f"\nSaving different points to: {output_path}")
    o3d.io.write_point_cloud(output_path, pcd_diff, write_ascii=False)
    
    print(f"  Total number of points saved: {len(all_different_points)}")
    
    # Save statistics to a text file
    stats_path = output_path.replace('.ply', '_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Point Cloud Comparison\n")
        f.write(f"================\n")
        f.write(f"File 1: {ply1_path}\n")
        f.write(f"File 2: {ply2_path}\n")
        f.write(f"Output: {output_path}\n\n")
        f.write(f"Distance threshold: {threshold}\n\n")
        f.write(f"Statistics:\n")
        f.write(f"  Number of points in file 1: {len(points1)}\n")
        f.write(f"  Number of points in file 2: {len(points2)}\n")
        f.write(f"  Number of different points: {num_different}\n")
        f.write(f"  Percentage of different points: {percentage:.2f}%\n")
        f.write(f"  میانگین فاصله: {np.mean(distances[different_mask]):.6f}\n")
        f.write(f"  حداکثر فاصله: {np.max(distances[different_mask]):.6f}\n")
        f.write(f"  حداقل فاصله متفاوت: {np.min(distances[different_mask]):.6f}\n")
    
    print(f"آمار کامل در: {stats_path}")
    
    # نمایش هیستوگرام فواصل
    if num_different > 0:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.hist(distances[different_mask], bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', label=f'آستانه ({threshold})')
        plt.xlabel('فاصله')
        plt.ylabel('تعداد نقاط')
        plt.title('توزیع فواصل نقاط متفاوت')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        hist_path = output_path.replace('.ply', '_histogram.png')
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        print(f"هیستوگرام ذخیره شد در: {hist_path}")
        
        # نمایش 10 نقطه با بیشترین اختلاف
        print(f"\nTop 10 points with largest differences:")
        top_indices = np.argsort(distances[different_mask])[-10:][::-1]
        for i, idx in enumerate(top_indices, 1):
            actual_idx = np.where(different_mask)[0][idx]
            print(f"  {i}. Point {actual_idx}: distance = {distances[actual_idx]:.6f}")

def main():
    parser = argparse.ArgumentParser(description='Compare two PLY files and extract different points')
    parser.add_argument('ply1', help='Path to the first PLY file')
    parser.add_argument('ply2', help='Path to the second PLY file')
    parser.add_argument('-o', '--output', default='differences.ply', 
                       help='Path to output file (default: differences.ply)')
    parser.add_argument('-t', '--threshold', type=float, default=0.001,
                       help='Distance threshold to detect different points (default: 0.001)')
    
    args = parser.parse_args()
    
    # Run comparison
    compare_point_clouds(
        ply1_path=args.ply1,
        ply2_path=args.ply2,
        output_path=args.output,
        threshold=args.threshold
    )

if __name__ == "__main__":
    main()