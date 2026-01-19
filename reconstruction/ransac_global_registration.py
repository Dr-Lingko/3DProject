import open3d as o3d
import preprocess_point_cloud as pp

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

if __name__ == "__main__":
    voxel_size = 5
    dir1 = r"E:\3DProject\testdata\kongjian\main_object\kongjian8_main_object.ply"
    dir2 = r"E:\3DProject\testdata\kongjian\main_object\kongjian9_main_object.ply"
    source_down, target_down,source_fpfh, target_fpfh, voxel_size =  pp.prepare_dataset(dir1,dir2,voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)
    o3d.visualization.draw_geometries([source_down, target_down])
    pp.draw_registration_result(source_down, target_down, result_ransac.transformation)