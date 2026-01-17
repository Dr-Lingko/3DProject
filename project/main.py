import open3d as o3d
import numpy as np

import PLY_Read
from point_cloud_plane_segmentation import plane_segmentation
from point_cloud_dbscan_clustering import dbscan_clustering

if __name__ == "__main__":
    pcd = PLY_Read.ply_read("E:/3DProject/testdata/kongjian/kongjian5.ply")

    o3d.visualization.draw([pcd])

    pcd = PLY_Read.filter_outliers(pcd)

    # o3d.visualization.draw([pcd])

    rest = plane_segmentation(pcd)

    # rest = PLY_Read.filter_outliers(rest)

    # 可视化
    o3d.visualization.draw([rest])

    main = dbscan_clustering(rest)

    o3d.visualization.draw([main])

    # 保存
    # o3d.io.write_point_cloud("E:/3DProject/testdata/kongjian/kongjian2_no_plane.ply", rest)