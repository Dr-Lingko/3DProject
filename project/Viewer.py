import open3d as o3d

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("E:/3DProject/testdata/kongjian/kongjian1_no_plane.ply")
    o3d.visualization.draw_geometries([pcd])
