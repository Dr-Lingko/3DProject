import open3d as o3d
import numpy as np

def ply_read(path):

    pcd = o3d.io.read_point_cloud(path)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])    # 翻转点云，否则点云会是倒置的。

    points = np.asarray(pcd.points)
    mask = ~np.isclose(points[:, 2], 0)  # 过滤z=0的点
    filtered_pcd = pcd.select_by_index(np.where(mask)[0])

    return filtered_pcd
