import open3d as o3d
import preprocess_point_cloud as pp

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size,
                                distance_threshold=None,
                                ransac_n=4,
                                max_iteration=200000,
                                confidence=0.999,
                                mutual_filter=False):
    if distance_threshold is None:
        distance_threshold = voxel_size * 2.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, mutual_filter,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, confidence))
    return result

if __name__ == "__main__":
    voxel_size = 10
    dir1 = r"E:\3DProject\D1\main_object_fixed\6.ply"
    dir2 = r"E:\3DProject\D1\main_object_fixed\7.ply"
    source_down, target_down,source_fpfh, target_fpfh, voxel_size =  pp.prepare_dataset(dir1,dir2,voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)
    o3d.visualization.draw_geometries([source_down, target_down])
    pp.draw_registration_result(source_down, target_down, result_ransac.transformation)