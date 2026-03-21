import open3d as o3d
import numpy as np

def compute_average_density(pcd):
    # 计算每个点的最近邻距离
    distances = pcd.compute_nearest_neighbor_distance()
    # 计算平均密度
    avg_density = 1.0 / np.mean(distances)
    print(f"平均点云密度: {avg_density}")
    return avg_density

def estimate_params(pcd, voxel_scale=2.0, ransac_scale=2.0, icp_scale=1.5, use_median=True):
    distances = np.asarray(pcd.compute_nearest_neighbor_distance())
    if distances.size == 0:
        voxel_size = 1.0
        ransac_distance = 2.0
        icp_threshold = 1.5
        stats = {"mean": 0.0, "median": 0.0, "std": 0.0, "count": 0}
        print("点云为空，使用默认参数。")
        return voxel_size, ransac_distance, icp_threshold, stats

    mean_nn_distance = float(np.mean(distances))
    median_nn_distance = float(np.median(distances))
    std_nn_distance = float(np.std(distances))
    base = median_nn_distance if use_median else mean_nn_distance

    voxel_size = base * voxel_scale
    ransac_distance = voxel_size * ransac_scale
    icp_threshold = voxel_size * icp_scale

    stats = {
        "mean": mean_nn_distance,
        "median": median_nn_distance,
        "std": std_nn_distance,
        "count": int(distances.size)
    }
    print(f"建议下采样半径: {voxel_size:.4f}")
    print(f"建议RANSAC距离阈值: {ransac_distance:.4f}")
    print(f"建议ICP距离阈值: {icp_threshold:.4f}")
    return voxel_size, ransac_distance, icp_threshold, stats

if __name__ == "__main__":
    file_path = r"E:\3DProject\D1\main_object_fixed\1.ply"
    pcd = o3d.io.read_point_cloud(file_path)
    compute_average_density(pcd)
    estimate_params(pcd)