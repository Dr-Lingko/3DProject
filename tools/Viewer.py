import open3d as o3d
import os
if __name__ == "__main__":
    # input_dir = r"E:\3DProject\testdata\kongjian"
    input_dir = r"E:\3DProject\D1\main_object_fixed"
    # input_dir = r"E:\点云测量\01\Data"
    # input_dir = r"E:\3DProject\testdata\bunny\data1"
    for fname in os.listdir(input_dir):
        if fname.endswith('.ply'):
            print(f"Viewing file: {fname}")
            pcd = o3d.io.read_point_cloud(os.path.join(input_dir, fname))
            # pcd.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw([pcd])

    # pcd1 = o3d.io.read_point_cloud("E:/3DProject/testdata/kongjian/main_object/5.ply")
    # down_pcd = pcd1.voxel_down_sample(voxel_size=2)
    # o3d.visualization.draw_geometries([down_pcd])