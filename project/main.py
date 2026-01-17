import open3d as o3d
import os


import PLY_Read
from point_cloud_plane_segmentation import plane_segmentation
from point_cloud_dbscan_clustering import dbscan_clustering

if __name__ == "__main__":

    input_dir = "E:/3DProject/testdata/kongjian/"
    output_dir = "E:/3DProject/testdata/kongjian/main_object/"
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.endswith('.ply'):

            print(f"Processing file: {fname}")
            pcd = PLY_Read.ply_read(os.path.join(input_dir, fname))    # 读取点云
            pcd_filtered = PLY_Read.filter_outliers(pcd)    # 离群点过滤
            pcd_without_plane = plane_segmentation(pcd_filtered)    # 平面分割，去除地面
            main_object = dbscan_clustering(pcd_without_plane)    # DBSCAN聚类,提取主体

            if main_object is not None:

                output_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_main_object.ply")
                o3d.io.write_point_cloud(output_path, main_object)
                print(f"Saved main object to: {output_path}")

            else:
                print(f"No main object found")


    # o3d.visualization.draw([pcd])
    #
    # pcd = PLY_Read.filter_outliers(pcd)     # 离群点过滤
    # o3d.visualization.draw([pcd])
    #
    # rest = plane_segmentation(pcd)      # 平面分割，去除地面
    #
    # main = dbscan_clustering(rest)      # DBSCAN聚类，提取主体
    #
    # o3d.visualization.draw([main])      # 显示主体点云