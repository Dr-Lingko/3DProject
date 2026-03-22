import open3d as o3d
import os
import re
import numpy as np
import preprocess_point_cloud as pp
import ransac_global_registration as gr
import icp_registration as icpr
from reconstruction.merge_utils import merge
from tools.estimate_params import estimate_params


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load(dir):
    ply_files = [f for f in os.listdir(dir) if f.endswith('.ply')]
    ply_files.sort(key=natural_sort_key)
    ply_paths = [os.path.join(dir, f) for f in ply_files]
    print(f"将按顺序处理以下点云: {ply_files}")
    return ply_paths


if __name__ == "__main__":

    folder = r"E:\3DProject\D2\main_object_fixed"
    paths =  load(folder)

    pcds = []
    for i in range(len(paths)):
        pcd = o3d.io.read_point_cloud(paths[i])
        pcds.append(pcd)
    o3d.visualization.draw(pcds)

    min_fitness = 0.15
    global_opt_distance = 0.0

    odometry = np.identity(4)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for i in range(len(paths) - 1):
        print(f"正在配准第 {i+1} 和第 {i+2} 个点云")

        voxel_size, ransac_dist, icp_threshold, stats = estimate_params(pcds[i])
        source, target, source_fpfh, target_fpfh, _ = pp.prepare_dataset(paths[i], paths[i + 1], voxel_size)
        result_ransac = gr.execute_global_registration(
            source, target, source_fpfh, target_fpfh, voxel_size,
            distance_threshold=ransac_dist,
            ransac_n=4,
            max_iteration=200000,
            confidence=0.999,
            mutual_filter=False
        )
        print(f"RANSAC fitness={result_ransac.fitness:.3f}, rmse={result_ransac.inlier_rmse:.3f}")

        if result_ransac.fitness < 0.05:
            voxel_size *= 2.0
            ransac_dist *= 2.0
            icp_threshold *= 2.0
            source, target, source_fpfh, target_fpfh, _ = pp.prepare_dataset(paths[i], paths[i + 1], voxel_size)
            result_ransac = gr.execute_global_registration(
                source, target, source_fpfh, target_fpfh, voxel_size,
                distance_threshold=ransac_dist,
                ransac_n=4,
                max_iteration=300000,
                confidence=0.999,
                mutual_filter=False
            )
            print(f"RANSAC(retry) fitness={result_ransac.fitness:.3f}, rmse={result_ransac.inlier_rmse:.3f}")

        voxel_sizes = [voxel_size * 2.0, voxel_size, max(voxel_size * 0.5, voxel_size * 0.25)]
        print(f"ICP voxel pyramid: {voxel_sizes}")
        result_icp = icpr.ICP_registration_point_to_plane_robust_multi_scale(
            pcds[i], pcds[i + 1], voxel_sizes, result_ransac.transformation
        )
        print(f"ICP fitness={result_icp.fitness:.3f}, rmse={result_icp.inlier_rmse:.3f}")

        final_threshold = max(icp_threshold, voxel_sizes[-1] * 1.5)
        max_rmse = final_threshold * 2.0
        global_opt_distance = max(global_opt_distance, final_threshold)

        if result_icp.fitness < min_fitness or result_icp.inlier_rmse > max_rmse:
            print("本对点云配准质量低，跳过以避免局部最优污染。")
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
            continue

        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, final_threshold, result_icp.transformation
        )
        odometry = np.dot(result_icp.transformation, odometry)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(odometry)
            )
        )
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(i,
                                                     i + 1,
                                                     result_icp.transformation,
                                                     information_icp,
                                                     uncertain=False))

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=global_opt_distance,
        edge_prune_threshold=0.25,
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
    o3d.io.write_point_cloud(os.path.join(folder, "merged.ply"), m)
