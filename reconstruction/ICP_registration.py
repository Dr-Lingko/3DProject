import open3d as o3d
import numpy as np

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


def _score_registration(result, threshold):
    # 更偏好高 fitness，且对大 rmse 进行惩罚
    threshold = max(float(threshold), 1e-6)
    return float(result.fitness) - float(result.inlier_rmse) / threshold


def _centroid_aligned_init(source, target, trans_init):
    source_np = np.asarray(source.points)
    target_np = np.asarray(target.points)
    if source_np.size == 0 or target_np.size == 0:
        return trans_init
    init = np.array(trans_init, copy=True)
    r = init[:3, :3]
    source_centroid = source_np.mean(axis=0)
    target_centroid = target_np.mean(axis=0)
    init[:3, 3] = target_centroid - r @ source_centroid
    return init


def _run_multiscale_icp(source, target, voxel_sizes, trans_init, max_iterations, estimator_builder):
    current = trans_init
    result = None
    for idx, voxel in enumerate(voxel_sizes):
        if voxel <= 0:
            continue
        source_down = source.voxel_down_sample(voxel)
        target_down = target.voxel_down_sample(voxel)
        _estimate_normals_for_icp(source_down, voxel)
        _estimate_normals_for_icp(target_down, voxel)
        threshold = voxel * 1.5
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations[min(idx, len(max_iterations) - 1)]
        )
        estimator = estimator_builder(threshold)
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, current, estimator, criteria
        )
        current = result.transformation
    return result


def ICP_registration_point_to_plane_robust_multi_scale(source, target, voxel_sizes, trans_init,
                                                       max_iterations=None, loss_k_factor=1.0):
    if max_iterations is None:
        max_iterations = [60, 40, 30]
    if len(voxel_sizes) == 0:
        raise ValueError("voxel_sizes 不能为空")

    def point_to_plane_estimator_builder(threshold):
        loss = o3d.pipelines.registration.TukeyLoss(k=threshold * loss_k_factor)
        return o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

    init_candidates = [np.array(trans_init, copy=True)]
    centroid_init = _centroid_aligned_init(source, target, trans_init)
    if not np.allclose(centroid_init, init_candidates[0]):
        init_candidates.append(centroid_init)

    best_result = None
    last_threshold = max(voxel_sizes[-1] * 1.5, 1e-6)
    for init in init_candidates:
        candidate = _run_multiscale_icp(
            source,
            target,
            voxel_sizes,
            init,
            max_iterations,
            point_to_plane_estimator_builder,
        )
        if candidate is None:
            continue
        if best_result is None or _score_registration(candidate, last_threshold) > _score_registration(best_result, last_threshold):
            best_result = candidate

    if best_result is None:
        raise RuntimeError("ICP 多尺度配准失败，未生成有效结果")

    # 最终用 Generalized ICP 再细化一次，提高在人体局部平面区域的稳定性。
    source_fine = source.voxel_down_sample(max(voxel_sizes[-1] * 0.5, voxel_sizes[-1] * 0.25))
    target_fine = target.voxel_down_sample(max(voxel_sizes[-1] * 0.5, voxel_sizes[-1] * 0.25))
    _estimate_normals_for_icp(source_fine, voxel_sizes[-1])
    _estimate_normals_for_icp(target_fine, voxel_sizes[-1])
    gicp_threshold = last_threshold
    gicp_result = o3d.pipelines.registration.registration_generalized_icp(
        source_fine,
        target_fine,
        gicp_threshold,
        best_result.transformation,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=35),
    )
    if _score_registration(gicp_result, gicp_threshold) > _score_registration(best_result, gicp_threshold):
        return gicp_result
    return best_result
