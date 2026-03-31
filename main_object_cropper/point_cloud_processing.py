import os
import copy
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import open3d as o3d


def load_point_cloud(pcd_path: str) -> o3d.geometry.PointCloud:
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"点云文件不存在：{pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    if len(pcd.points) == 0:
        raise ValueError("点云为空，无法处理。")
    return pcd


def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    if voxel_size <= 0:
        return pcd
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def smooth_point_cloud(
    pcd: o3d.geometry.PointCloud,
    k: int,
    iterations: int,
    alpha: float,
) -> o3d.geometry.PointCloud:
    if k <= 1 or iterations <= 0 or alpha <= 0:
        return pcd

    smoothed = copy.deepcopy(pcd)
    points = np.asarray(smoothed.points)
    for _ in range(iterations):
        tree = o3d.geometry.KDTreeFlann(smoothed)
        new_points = points.copy()
        for i in range(len(points)):
            _, idx, _ = tree.search_knn_vector_3d(smoothed.points[i], k)
            mean = points[idx].mean(axis=0)
            new_points[i] = (1.0 - alpha) * points[i] + alpha * mean
        points = new_points
        smoothed.points = o3d.utility.Vector3dVector(points)
    return smoothed


def remove_outliers(
    pcd: o3d.geometry.PointCloud,
    method: str,
    nb_neighbors: int,
    std_ratio: float,
    radius: float,
    min_points: int,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    method = method.lower()
    if method == "statistical":
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    elif method == "radius":
        _, ind = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)
    else:
        raise ValueError(f"未知离群点方法：{method}")

    filtered = pcd.select_by_index(ind)
    return filtered, np.asarray(ind, dtype=np.int64)


def save_point_cloud(pcd: o3d.geometry.PointCloud, output_path: str) -> None:
    ok = o3d.io.write_point_cloud(output_path, pcd)
    if not ok:
        raise IOError(f"保存点云失败：{output_path}")


def visualize_point_cloud(pcd: o3d.geometry.PointCloud, title: str) -> None:
    o3d.visualization.draw_geometries([pcd], window_name=title)


@dataclass
class ProcessConfig:
    input_path: str
    output_path: str
    voxel: float = 0.0
    outlier_method: str = "statistical"
    nb_neighbors: int = 20
    std_ratio: float = 2.0
    radius: float = 0.02
    min_points: int = 16
    smooth_enabled: bool = False
    smooth_k: int = 20
    smooth_iterations: int = 1
    smooth_alpha: float = 0.5
    visualize_input: bool = True
    visualize_output: bool = True
    visualize_compare: bool = False


# 在这里直接改参数，不需要终端输入
CONFIG = ProcessConfig(
    input_path=r"E:\3DProject\D2\main_object_fixed\super_sqz.ply",
    output_path=r"E:\3DProject\D2\main_object_fixed\super_sqz_filtered.ply",
    voxel=0.5,
    outlier_method="statistical",  # "statistical" 或 "radius"
    nb_neighbors=10,
    std_ratio=2.0,
    radius=0.5,
    min_points=16,
    smooth_enabled=True,
    smooth_k=20,
    smooth_iterations=2,
    smooth_alpha=0.4,
    visualize_input=True,
    visualize_output=True,
    visualize_compare=False,
)


def main(config: ProcessConfig) -> None:
    pcd = load_point_cloud(config.input_path)

    if config.visualize_input:
        visualize_point_cloud(pcd, "输入点云")

    down = voxel_downsample(pcd, config.voxel)
    filtered, _ = remove_outliers(
        down,
        method=config.outlier_method,
        nb_neighbors=config.nb_neighbors,
        std_ratio=config.std_ratio,
        radius=config.radius,
        min_points=config.min_points,
    )

    if config.smooth_enabled:
        filtered = smooth_point_cloud(
            filtered,
            k=config.smooth_k,
            iterations=config.smooth_iterations,
            alpha=config.smooth_alpha,
        )

    if config.visualize_compare:
        input_vis = copy.deepcopy(pcd)
        output_vis = copy.deepcopy(filtered)
        input_vis.paint_uniform_color([0.2, 0.6, 1.0])
        output_vis.paint_uniform_color([1.0, 0.5, 0.2])
        o3d.visualization.draw_geometries(
            [input_vis, output_vis],
            window_name="输入(蓝) vs 输出(橙)",
        )

    save_point_cloud(filtered, config.output_path)
    print(f"完成：{config.output_path}")
    print(f"点数变化：{len(pcd.points)} -> {len(down.points)} -> {len(filtered.points)}")

    if config.visualize_output:
        visualize_point_cloud(filtered, "输出点云")


if __name__ == "__main__":
    main(CONFIG)
