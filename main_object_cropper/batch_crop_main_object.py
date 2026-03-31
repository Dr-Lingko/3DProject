import open3d as o3d
import os
import numpy as np

import ply_read
from point_cloud_plane_segmentation import plane_segmentation
from point_cloud_dbscan_clustering import dbscan_clustering
from point_cloud_correct_flying_pixels import correct_flying_pixels

def  filter_outliers(pcd, nb_neighbors=20, std_ratio=2):

    # 离群点过滤
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    filtered_pcd = pcd.select_by_index(ind)
    # o3d.visualization.draw([filtered_pcd])
    return filtered_pcd

def main():
    input_dir = r"E:\点云测量\P1B"
    output_dir = r"E:\点云测量\P1B\filtered"
    crop = False  # 是否进行平面分割和DBSCAN聚类提取主体，False则只进行飘点修正和离群点过滤
    
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.endswith('.ply'):

            print(f"Processing file: {fname}")
            pcd = ply_read.ply_read(os.path.join(input_dir, fname))    # 读取点云
            pcd_filtered = correct_flying_pixels(pcd,80,20,0.5)    # 飘点修正
            if crop:
                pcd_without_plane = plane_segmentation(pcd_filtered)    # 平面分割，去除地面
                main_object = dbscan_clustering(pcd_without_plane)    # DBSCAN聚类,提取主体
            else:
                main_object = pcd_filtered
                
            down_main_object = main_object.voxel_down_sample(voxel_size=3)      # 体素下采样，减少点云数量，加快后续处理速度
            down_main_object = filter_outliers(down_main_object, nb_neighbors=20, std_ratio=0.3)    # 离群点过滤
            down_main_object.paint_uniform_color([0, 1, 0])



            if down_main_object is not None:

                output_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_main_object.ply")
                o3d.io.write_point_cloud(output_path, down_main_object)
                print(f"Saved main object to: {output_path}")

            else:
                print(f"No main object found")
                
if __name__ == "__main__":
    main()