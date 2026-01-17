import open3d as o3d
import numpy as np

from PLY_Read import ply_read
from point_cloud_plane_segmentation import plane_segmentation

if __name__ == "__main__":
    pcd = ply_read("E:/3DProject/testdata/kongjian/kongjian1.ply")
    rest = plane_segmentation(pcd)
    o3d.visualization.draw([rest])
    o3d.io.write_point_cloud("E:/3DProject/testdata/kongjian/kongjian1_no_plane.ply", rest)