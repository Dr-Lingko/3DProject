import open3d as o3d
import numpy as np
import copy
import os


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*8, max_nn=150))
    return pcd_down, fpfh

def execute_global_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(2000000, 600))
    return result

def merge(pcd1, pcd2):
    points = np.vstack((np.asarray(pcd1.points), np.asarray(pcd2.points)))
    colors = np.vstack((np.asarray(pcd1.colors), np.asarray(pcd2.colors)))
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(points)
    merged_pcd.colors = o3d.utility.Vector3dVector(colors)
    return merged_pcd

def registrate(pcd_1, pcd_2,voxel_size):
    pcd_1_down, pcd_1_fpfh = preprocess_point_cloud(pcd_1, voxel_size)
    pcd_2_down, pcd_2_fpfh = preprocess_point_cloud(pcd_2, voxel_size)
    result_ransac = execute_global_registration(pcd_1_down, pcd_2_down, pcd_1_fpfh, pcd_2_fpfh,
                                                   voxel_size)
    result_icp = o3d.pipelines.registration.registration_icp(
        pcd_1_down, pcd_2_down, voxel_size * 0.5, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=600))
    pcd_1_aligned = copy.deepcopy(pcd_1)
    pcd_1_aligned.transform(result_icp.transformation)
    merged_pcd = merge(pcd_2, pcd_1_aligned)
    #o3d.visualization.draw_geometries([merged_pcd])
    return merged_pcd


if __name__ == "__main__":
    voxel_size = 0.005
    folder = r"E:\3DProject\testdata\bunny\data"
    ply_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.ply')]
    ply_files.sort()  # 可选：按文件名排序
    point_clouds = [o3d.io.read_point_cloud(f) for f in ply_files]
    # 依次处理
    for i in range(1, len(point_clouds)):
        if i==1:
            tgt = point_clouds[i - 1]
        else:
            tgt = merged
        src = point_clouds[i]
        o3d.visualization.draw_geometries([src])
        o3d.visualization.draw_geometries([tgt])
        merged = registrate(src, tgt, voxel_size)
        print(f"Registered {i}/{len(point_clouds)-1}")
    o3d.visualization.draw_geometries([merged])



