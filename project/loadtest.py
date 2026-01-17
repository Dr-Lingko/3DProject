import open3d as o3d
import numpy as np

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("E:/3DProject/testdata/kongjian/kongjian1.ply")
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * np.array([-1, -1, -1]))

    # o3d.visualization.draw_geometries([pcd])
    down_pcd = pcd.voxel_down_sample(voxel_size=10)
    print(down_pcd)
    down_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
    )
    o3d.visualization.draw_geometries([down_pcd])