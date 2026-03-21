import open3d as o3d

def ICP_registration(source, target, threshold,trans_init):
    print("Apply point-to-point ICP")
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold,
        trans_init,o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return icp_result

def ICP_registration_point_to_plane_robust(source, target, threshold, trans_init, max_iteration=80):
    print("Apply point-to-plane ICP with robust loss")
    loss = o3d.pipelines.registration.TukeyLoss(k=threshold)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init, estimation, criteria
    )
    return icp_result


def ICP_registration_point_to_plane_robust_two_stage(source, target, threshold, trans_init,
                                                     coarse_factor=2.0,
                                                     coarse_iterations=60,
                                                     fine_iterations=40):
    coarse_threshold = threshold * coarse_factor
    result_coarse = ICP_registration_point_to_plane_robust(
        source, target, coarse_threshold, trans_init, max_iteration=coarse_iterations
    )
    result_fine = ICP_registration_point_to_plane_robust(
        source, target, threshold, result_coarse.transformation, max_iteration=fine_iterations
    )
    return result_fine

def _estimate_normals_for_icp(pcd, voxel_size):
    if voxel_size <= 0:
        return
    radius = voxel_size * 2.0
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))


def ICP_registration_point_to_plane_robust_multi_scale(source, target, voxel_sizes, trans_init,
                                                       max_iterations=None, loss_k_factor=1.0):
    if max_iterations is None:
        max_iterations = [60, 40, 30]
    current = trans_init
    for idx, voxel in enumerate(voxel_sizes):
        if voxel <= 0:
            continue
        source_down = source.voxel_down_sample(voxel)
        target_down = target.voxel_down_sample(voxel)
        _estimate_normals_for_icp(source_down, voxel)
        _estimate_normals_for_icp(target_down, voxel)
        threshold = voxel * 1.5
        loss = o3d.pipelines.registration.TukeyLoss(k=threshold * loss_k_factor)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations[min(idx, len(max_iterations) - 1)])
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, current, estimation, criteria
        )
        current = result.transformation
    return result
