import open3d as o3d
import os
if __name__ == "__main__":
    input_dir = "E:/3DProject/testdata/kongjian/main_object/"
    for fname in os.listdir(input_dir):
        if fname.endswith('.ply'):
            print(f"Viewing file: {fname}")
            pcd = o3d.io.read_point_cloud(os.path.join(input_dir, fname))
            o3d.visualization.draw([pcd])

