import open3d as o3d
import numpy as np

def compute_average_density(pcd):
    # 计算每个点的最近邻距离
    distances = pcd.compute_nearest_neighbor_distance()
    # 计算平均密度
    avg_density = 1.0 / np.mean(distances)
    print(f"平均点云密度: {avg_density}")
    return avg_density

def estimate_params(pcd):
    distances = pcd.compute_nearest_neighbor_distance()
    mean_nn_distance = np.mean(distances)
    voxel_size = mean_nn_distance * 2
    icp_threshold = voxel_size * 1.5
    print(f"建议下采样半径: {voxel_size:.4f}")
    print(f"建议ICP距离阈值: {icp_threshold:.4f}")
    return voxel_size, icp_threshold

if __name__ == "__main__":
    file_path = r"E:\3DProject\testdata\bunny\data\bun000.ply"
    pcd = o3d.io.read_point_cloud(file_path)
    compute_average_density(pcd)
    estimate_params(pcd)