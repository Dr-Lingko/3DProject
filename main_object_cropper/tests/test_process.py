import numpy as np
import open3d as o3d
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from point_cloud_processing import remove_outliers, voxel_downsample


def make_cloud_with_outlier() -> o3d.geometry.PointCloud:
    pts = np.random.randn(1000, 3).astype(np.float64)
    outlier = np.array([[50.0, 50.0, 50.0]], dtype=np.float64)
    pts = np.vstack([pts, outlier])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def test_outlier_removal():
    pcd = make_cloud_with_outlier()
    filtered, _ = remove_outliers(
        pcd,
        method="statistical",
        nb_neighbors=20,
        std_ratio=2.0,
        radius=0.02,
        min_points=16,
    )
    assert len(filtered.points) < len(pcd.points)


def test_voxel_downsample():
    pcd = make_cloud_with_outlier()
    down = voxel_downsample(pcd, voxel_size=0.5)
    assert len(down.points) <= len(pcd.points)


if __name__ == "__main__":
    np.random.seed(0)
    test_outlier_removal()
    test_voxel_downsample()
    print("test_process: OK")

