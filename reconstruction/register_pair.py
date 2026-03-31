import argparse
import os
import sys
import open3d as o3d

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import preprocess_point_cloud as pp
import ransac_global_registration as gr
import icp_registration as icpr
from tools.estimate_params import estimate_params


def main():
    parser = argparse.ArgumentParser(description="Register two point clouds with multi-scale ICP.")
    parser.add_argument("--source", required=True, help="Path to source .ply")
    parser.add_argument("--target", required=True, help="Path to target .ply")
    parser.add_argument("--output", default="", help="Optional output merged .ply")
    args = parser.parse_args()

    source_raw = o3d.io.read_point_cloud(args.source)
    target_raw = o3d.io.read_point_cloud(args.target)

    voxel_size, ransac_dist, icp_threshold, _ = estimate_params(source_raw)
    source, target, source_fpfh, target_fpfh, _ = pp.prepare_dataset(args.source, args.target, voxel_size)
    result_ransac = gr.execute_global_registration(
        source, target, source_fpfh, target_fpfh, voxel_size,
        distance_threshold=ransac_dist,
        ransac_n=4,
        max_iteration=200000,
        confidence=0.999,
        mutual_filter=False
    )

    voxel_sizes = [voxel_size * 2.0, voxel_size, max(voxel_size * 0.5, voxel_size * 0.25)]
    result_icp = icpr.ICP_registration_point_to_plane_robust_multi_scale(
        source_raw, target_raw, voxel_sizes, result_ransac.transformation
    )

    aligned = o3d.geometry.PointCloud(source_raw)
    aligned.transform(result_icp.transformation)
    aligned.paint_uniform_color([1.0, 0.0, 0.0])
    target_raw.paint_uniform_color([0.0, 1.0, 0.0])
    o3d.visualization.draw([aligned, target_raw])

    if args.output:
        merged = aligned + target_raw
        o3d.io.write_point_cloud(args.output, merged)
        print(f"Saved merged point cloud to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

