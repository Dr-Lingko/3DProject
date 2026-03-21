import open3d as o3d

def  filter_outliers(pcd,nb_neighbors=20,std_ratio=2):

    # 离群点过滤
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    filtered_pcd = pcd.select_by_index(ind)
    # o3d.visualization.draw([filtered_pcd])

    return filtered_pcd

if __name__ == "__main__":

    pcd = o3d.io.read_point_cloud(r"E:\3DProject\testdata\bunny\data1\merged.ply")   # 读取点云


    pcd = pcd.voxel_down_sample(voxel_size=0.002)

    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(r"E:\3DProject\testdata\bunny\data1\merged_fixed.ply",pcd)
