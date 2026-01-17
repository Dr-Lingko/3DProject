import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sample_ply_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud("E:/3DProject/testdata/kongjian/kongjian1_no_plane.ply")


    points = np.asarray(pcd.points)
    mask = ~np.isclose(points[:, 2], 0)  # 过滤z=0的点
    filtered_pcd = pcd.select_by_index(np.where(mask)[0])

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=10, min_points=10, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw([pcd])
