import argparse
import open3d as o3d
import os
import re
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import preprocess_point_cloud as pp
import ransac_global_registration as gr
import icp_registration as icpr
from tools.estimate_params import estimate_params


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def relative_transform(source_pose, target_pose):
    return np.linalg.inv(target_pose) @ source_pose


def rotation_error_deg(transform):
    r = transform[:3, :3]
    cos_theta = (np.trace(r) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def compose_transform(rotation, translation):
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def clamp_relative_transform(transform, max_translation, max_rotation_deg):
    translation = transform[:3, 3]
    translation_norm = np.linalg.norm(translation)
    if translation_norm > max_translation and translation_norm > 1e-9:
        translation = translation * (max_translation / translation_norm)

    r = transform[:3, :3]
    cos_theta = (np.trace(r) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    max_rot_rad = float(np.deg2rad(max_rotation_deg))

    if theta <= max_rot_rad or theta < 1e-12:
        return compose_transform(r, translation)

    denom = 2.0 * np.sin(theta)
    if abs(denom) < 1e-12:
        return compose_transform(np.eye(3), translation)

    axis = np.array([
        r[2, 1] - r[1, 2],
        r[0, 2] - r[2, 0],
        r[1, 0] - r[0, 1],
    ]) / denom

    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return compose_transform(np.eye(3), translation)
    axis = axis / axis_norm

    k = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    r_clamped = np.eye(3) + np.sin(max_rot_rad) * k + (1.0 - np.cos(max_rot_rad)) * (k @ k)
    return compose_transform(r_clamped, translation)


def estimate_pair_params(source_pcd, target_pcd):
    src_voxel, src_ransac, src_icp, _ = estimate_params(source_pcd)
    tgt_voxel, tgt_ransac, tgt_icp, _ = estimate_params(target_pcd)
    voxel_size = max(src_voxel, tgt_voxel)
    ransac_dist = max(src_ransac, tgt_ransac)
    icp_threshold = max(src_icp, tgt_icp)
    return voxel_size, ransac_dist, icp_threshold


def score_icp_result(icp_result, threshold):
    threshold = max(float(threshold), 1e-6)
    return float(icp_result.fitness) - float(icp_result.inlier_rmse) / threshold


def register_pair_raw(source_raw, target_raw, voxel_size, ransac_dist, icp_threshold,
                      ransac_max_iter=200000, prior_init=None):
    source_down, source_fpfh = pp.preprocess_point_cloud(source_raw, voxel_size)
    target_down, target_fpfh = pp.preprocess_point_cloud(target_raw, voxel_size)

    result_ransac = gr.execute_global_registration(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        voxel_size,
        distance_threshold=ransac_dist,
        ransac_n=4,
        max_iteration=ransac_max_iter,
        confidence=0.999,
        mutual_filter=False,
    )

    voxel_sizes = [voxel_size * 2.0, voxel_size, max(voxel_size * 0.5, voxel_size * 0.25)]
    final_threshold = max(icp_threshold, voxel_sizes[-1] * 1.5)
    init_candidates = [result_ransac.transformation]
    if prior_init is not None:
        init_candidates.append(prior_init)

    result_icp = None
    for init in init_candidates:
        candidate_icp = icpr.ICP_registration_point_to_plane_robust_multi_scale(
            source_raw,
            target_raw,
            voxel_sizes,
            init,
        )
        if result_icp is None or score_icp_result(candidate_icp, final_threshold) > score_icp_result(result_icp, final_threshold):
            result_icp = candidate_icp

    max_rmse = final_threshold * 2.0
    return {
        "source_down": source_down,
        "target_down": target_down,
        "result_ransac": result_ransac,
        "result_icp": result_icp,
        "final_threshold": final_threshold,
        "max_rmse": max_rmse,
        "voxel_sizes": voxel_sizes,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Sequence registration with robust fallback and sparse loop closures.")
    parser.add_argument("--folder", required=True, help="Directory containing input .ply files")
    parser.add_argument("--output", default="merged.ply", help="Output merged point cloud file name")
    parser.add_argument("--min-fitness", type=float, default=0.15, help="Minimum acceptable ICP fitness")
    parser.add_argument("--loop-interval", type=int, default=4, help="Frame interval for sparse loop closure candidates")
    parser.add_argument("--loop-weight", type=float, default=1.0, help="Global optimization loop-closure preference")
    parser.add_argument("--max-loop-rot", type=float, default=20.0, help="Max allowed loop consistency rotation error in degrees")
    parser.add_argument("--max-loop-dist", type=float, default=0.0,
                        help="Max allowed pose-distance for loop candidates; 0 means auto")
    parser.add_argument("--max-loop-candidates", type=int, default=12,
                        help="Maximum accepted loop candidates after distance filtering")
    parser.add_argument("--motion-max-translation", type=float, default=0.0,
                        help="Clamp pairwise translation; 0 means auto by adaptive threshold")
    parser.add_argument("--motion-max-rotation", type=float, default=25.0,
                        help="Clamp pairwise rotation in degrees to avoid sudden pose jumps")
    return parser.parse_args()

def load(dir):
    ply_files = [f for f in os.listdir(dir) if f.lower().endswith('.ply')]
    ply_files.sort(key=natural_sort_key)
    ply_paths = [os.path.join(dir, f) for f in ply_files]
    print(f"将按顺序处理以下点云: {ply_files}")
    return ply_paths


if __name__ == "__main__":
    args = parse_args()

    folder = args.folder
    paths =  load(folder)
    if len(paths) < 2:
        raise ValueError("至少需要两个 .ply 文件进行序列配准。")

    pcds = []
    for i in range(len(paths)):
        pcd = o3d.io.read_point_cloud(paths[i])
        pcds.append(pcd)
    o3d.visualization.draw(pcds)

    min_fitness = args.min_fitness
    global_opt_distance = 0.0
    last_good_transform = np.eye(4)
    prev_applied_transform = np.eye(4)
    low_quality_streak = 0

    odometry = np.identity(4)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for i in range(len(paths) - 1):
        print(f"正在配准第 {i+1} 和第 {i+2} 个点云")

        voxel_size, ransac_dist, icp_threshold = estimate_pair_params(pcds[i], pcds[i + 1])
        pair_result = register_pair_raw(
            pcds[i],
            pcds[i + 1],
            voxel_size,
            ransac_dist,
            icp_threshold,
            ransac_max_iter=200000,
            prior_init=last_good_transform,
        )
        result_ransac = pair_result["result_ransac"]
        result_icp = pair_result["result_icp"]
        source_down = pair_result["source_down"]
        target_down = pair_result["target_down"]
        final_threshold = pair_result["final_threshold"]
        max_rmse = pair_result["max_rmse"]
        print(f"RANSAC fitness={result_ransac.fitness:.3f}, rmse={result_ransac.inlier_rmse:.3f}")

        if result_ransac.fitness < 0.05:
            voxel_size *= 2.0
            ransac_dist *= 2.0
            icp_threshold *= 2.0
            pair_result = register_pair_raw(
                pcds[i],
                pcds[i + 1],
                voxel_size,
                ransac_dist,
                icp_threshold,
                ransac_max_iter=300000,
                prior_init=last_good_transform,
            )
            result_ransac = pair_result["result_ransac"]
            result_icp = pair_result["result_icp"]
            source_down = pair_result["source_down"]
            target_down = pair_result["target_down"]
            final_threshold = pair_result["final_threshold"]
            max_rmse = pair_result["max_rmse"]
            print(f"RANSAC(retry) fitness={result_ransac.fitness:.3f}, rmse={result_ransac.inlier_rmse:.3f}")

        print(f"ICP voxel pyramid: {pair_result['voxel_sizes']}")
        print(f"ICP fitness={result_icp.fitness:.3f}, rmse={result_icp.inlier_rmse:.3f}")

        global_opt_distance = max(global_opt_distance, final_threshold)
        motion_max_translation = args.motion_max_translation if args.motion_max_translation > 0.0 else final_threshold * 2.0
        result_icp_transformation = clamp_relative_transform(
            result_icp.transformation,
            max_translation=motion_max_translation,
            max_rotation_deg=args.motion_max_rotation,
        )

        if result_icp.fitness < min_fitness or result_icp.inlier_rmse > max_rmse:
            low_quality_streak += 1
            print("本对点云配准质量低，先尝试局部重配准。")
            retry_init = last_good_transform
            retry_icp = icpr.ICP_registration_point_to_plane_robust_multi_scale(
                pcds[i], pcds[i + 1], pair_result["voxel_sizes"], retry_init
            )
            if retry_icp.fitness >= result_icp.fitness:
                result_icp = retry_icp
                result_icp_transformation = clamp_relative_transform(
                    result_icp.transformation,
                    max_translation=motion_max_translation,
                    max_rotation_deg=args.motion_max_rotation,
                )

            print(f"局部重配准后 ICP fitness={result_icp.fitness:.3f}, rmse={result_icp.inlier_rmse:.3f}")
            if result_icp.fitness >= min_fitness and result_icp.inlier_rmse <= max_rmse:
                print("局部重配准已达标，按正常相邻边写入位姿图。")
                information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    source_down, target_down, final_threshold, result_icp_transformation
                )
                last_good_transform = result_icp_transformation
                prev_applied_transform = result_icp_transformation
                low_quality_streak = 0
                odometry = np.dot(result_icp_transformation, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)
                    )
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(i,
                                                             i + 1,
                                                             result_icp_transformation,
                                                             information_icp,
                                                             uncertain=False))
                continue

            fallback_transformation = None
            if i >= 1:
                print("尝试跨一帧桥接配准（i-1 -> i+1）以修复潜在断点。")
                bridge_voxel, bridge_ransac, bridge_icp = estimate_pair_params(pcds[i - 1], pcds[i + 1])
                bridge_result = register_pair_raw(
                    pcds[i - 1],
                    pcds[i + 1],
                    bridge_voxel,
                    bridge_ransac,
                    bridge_icp,
                    ransac_max_iter=300000,
                )
                bridge_icp_result = bridge_result["result_icp"]
                bridge_threshold = bridge_result["final_threshold"]
                bridge_max_rmse = bridge_result["max_rmse"]
                print(f"桥接 ICP fitness={bridge_icp_result.fitness:.3f}, rmse={bridge_icp_result.inlier_rmse:.3f}")
                if bridge_icp_result.fitness >= min_fitness * 0.8 and bridge_icp_result.inlier_rmse <= bridge_max_rmse:
                    candidate_transform = bridge_icp_result.transformation @ np.linalg.inv(prev_applied_transform)
                    fallback_transformation = clamp_relative_transform(
                        candidate_transform,
                        max_translation=final_threshold * 2.0,
                        max_rotation_deg=12.0,
                    )
                    print("桥接成功，使用推导相邻变换作为降级约束。")

            if fallback_transformation is None:
                print("桥接未通过，使用连续性降级：沿上一段可信运动外推并以低置信边参与优化。")
                fallback_transformation = last_good_transform.copy()
            if low_quality_streak >= 2:
                fallback_transformation = clamp_relative_transform(
                    fallback_transformation,
                    max_translation=final_threshold * 1.5,
                    max_rotation_deg=8.0,
                )
            fallback_threshold = final_threshold
            information_fallback = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                source_down, target_down, fallback_threshold, fallback_transformation
            )
            prev_applied_transform = fallback_transformation
            odometry = np.dot(fallback_transformation, odometry)
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(
                    np.linalg.inv(odometry)
                )
            )
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(i,
                                                         i + 1,
                                                         fallback_transformation,
                                                         information_fallback,
                                                         uncertain=True))
            global_opt_distance = max(global_opt_distance, fallback_threshold)
            continue

        low_quality_streak = 0
        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source_down, target_down, final_threshold, result_icp_transformation
        )
        last_good_transform = result_icp_transformation
        prev_applied_transform = result_icp_transformation
        odometry = np.dot(result_icp_transformation, odometry)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(odometry)
            )
        )
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(i,
                                                     i + 1,
                                                     result_icp_transformation,
                                                     information_icp,
                                                     uncertain=False))

    def try_add_loop_edge(loop_src, loop_tgt):
        nonlocal_global = 0.0
        src_pose = pose_graph.nodes[loop_src].pose
        tgt_pose = pose_graph.nodes[loop_tgt].pose
        pose_dist = np.linalg.norm(src_pose[:3, 3] - tgt_pose[:3, 3])
        auto_max_loop_dist = max(global_opt_distance * 4.0, 1e-6)
        max_loop_dist = args.max_loop_dist if args.max_loop_dist > 0.0 else auto_max_loop_dist
        if pose_dist > max_loop_dist:
            print(f"闭环候选过远，跳过: src={loop_src+1}, tgt={loop_tgt+1}, dist={pose_dist:.3f}, limit={max_loop_dist:.3f}")
            return nonlocal_global

        print(f"尝试闭环配准: 第 {loop_src + 1} 个点云 -> 第 {loop_tgt + 1} 个点云")
        voxel_size, ransac_dist, icp_threshold = estimate_pair_params(pcds[loop_src], pcds[loop_tgt])
        predicted_loop = relative_transform(
            pose_graph.nodes[loop_src].pose,
            pose_graph.nodes[loop_tgt].pose,
        )
        pair_result = register_pair_raw(
            pcds[loop_src],
            pcds[loop_tgt],
            voxel_size,
            ransac_dist,
            icp_threshold,
            ransac_max_iter=300000,
            prior_init=predicted_loop,
        )
        result_icp = pair_result["result_icp"]
        source_down = pair_result["source_down"]
        target_down = pair_result["target_down"]
        final_threshold = pair_result["final_threshold"]
        max_rmse = pair_result["max_rmse"]

        print(f"闭环 ICP fitness={result_icp.fitness:.3f}, rmse={result_icp.inlier_rmse:.3f}")
        if result_icp.fitness < min_fitness or result_icp.inlier_rmse > max_rmse:
            print("闭环质量不足，未加入闭环边。")
            return nonlocal_global

        loop_delta = np.linalg.inv(predicted_loop) @ result_icp.transformation
        trans_err = np.linalg.norm(loop_delta[:3, 3])
        rot_err = rotation_error_deg(loop_delta)
        max_loop_trans_err = final_threshold * 2.5
        max_loop_rot_err = args.max_loop_rot
        print(f"闭环一致性检查: trans_err={trans_err:.3f}, rot_err={rot_err:.2f} deg")

        if trans_err > max_loop_trans_err or rot_err > max_loop_rot_err:
            print("闭环与链式轨迹冲突，拒绝该闭环以避免整体扭曲。")
            return nonlocal_global

        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source_down,
            target_down,
            final_threshold,
            result_icp.transformation,
        )
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                loop_src,
                loop_tgt,
                result_icp.transformation,
                information_icp,
                uncertain=True,
            )
        )
        print("闭环边通过一致性校验，已加入位姿图。")
        nonlocal_global = max(nonlocal_global, final_threshold)
        return nonlocal_global

    if len(paths) >= 3:
        loop_candidates = []
        stride = max(2, args.loop_interval)
        for idx in range(stride, len(paths)):
            tgt = idx - stride
            if idx - tgt > 1:
                loop_candidates.append((idx, tgt))
            tgt2 = idx - stride * 2
            if tgt2 >= 0 and idx - tgt2 > 1:
                loop_candidates.append((idx, tgt2))

        seen = set()
        dedup_candidates = []
        for src_idx, tgt_idx in loop_candidates:
            key = (src_idx, tgt_idx)
            if src_idx == tgt_idx or abs(src_idx - tgt_idx) <= 1:
                continue
            if key in seen:
                continue
            seen.add(key)
            dedup_candidates.append((src_idx, tgt_idx))

        # Keep only the nearest candidates in current chained pose space.
        dedup_candidates.sort(
            key=lambda p: np.linalg.norm(
                pose_graph.nodes[p[0]].pose[:3, 3] - pose_graph.nodes[p[1]].pose[:3, 3]
            )
        )
        dedup_candidates = dedup_candidates[:max(1, args.max_loop_candidates)]

        print(f"闭环候选数量: {len(dedup_candidates)}")
        for src_idx, tgt_idx in dedup_candidates:
            global_opt_distance = max(global_opt_distance, try_add_loop_edge(src_idx, tgt_idx))

    if global_opt_distance <= 0.0:
        global_opt_distance = 1.0

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=global_opt_distance,
        edge_prune_threshold=0.25,
        preference_loop_closure=args.loop_weight,
        reference_node=0
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option
    )

    m = o3d.geometry.PointCloud()

    print("Transform points and display")
    for point_id in range(len(pcds)):
        print(pose_graph.nodes[point_id].pose)
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        m += pcds[point_id]
    o3d.visualization.draw(m)
    o3d.io.write_point_cloud(os.path.join(folder, args.output), m)
