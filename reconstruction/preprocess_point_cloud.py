import copy
import open3d as o3d
import numpy as np


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 1, 0])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*8, max_nn=100))
    return pcd_down, fpfh

def prepare_dataset(dir1,dir2,voxel_size):
    source = o3d.io.read_point_cloud(dir1)
    target = o3d.io.read_point_cloud(dir2)
    trans_init = np.asarray(
        [[1,0,0,0],
         [0,1,0,0],
         [0,0,1,0],
         [0,0,0,1]])
    source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))
    source, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target, target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source, target, source_fpfh, target_fpfh, voxel_size


if __name__ == "__main__": # 测试用
    voxel_size = 0.05
    dir1 = r"E:\3DProject\testdata\kongjian\main_object\kongjian1_main_object.ply"
    dir2 = r"E:\3DProject\testdata\kongjian\main_object\kongjian3_main_object.ply"
    prepare_dataset(dir1,dir2,voxel_size)
