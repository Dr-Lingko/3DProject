import open3d as o3d
import os
import re
import numpy as np
import preprocess_point_cloud as pp
import ransac_global_registration as gr
import ICP_registration as icpr
from reconstruction.t import merge


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load(dir):
    ply_files = [f for f in os.listdir(folder) if f.endswith('.ply')]
    ply_files.sort(key=natural_sort_key)
    ply_paths = [os.path.join(folder, f) for f in ply_files]
    print(f"将按顺序处理以下点云: {ply_files}")
    return ply_paths


if __name__ == "__main__":

    folder = r"E:\3DProject\testdata\bunny\data"
    paths =  load(folder)

    pcds = []
    for i in range(len(paths) - 1):
        pcd = o3d.io.read_point_cloud(paths[i])
        pcds.append(pcd)
    o3d.visualization.draw(pcds)

    voxel_size = 8
    odometry = np.identity(4)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for i in range(len(paths) - 1):
        pcd1 = paths[i]
        pcd2 = paths[i + 1]

        print(f"正在配准第 {i+1} 和第 {i+2} 个点云")
        source, target, source_fpfh, target_fpfh, voxel_size = pp.prepare_dataset(paths[i], paths[i + 1], voxel_size)
        result_ransac = gr.execute_global_registration(source, target, source_fpfh, target_fpfh, voxel_size)
        result_icp = icpr.ICP_registration(source, target, 1.5*voxel_size, result_ransac.transformation)

        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, 1.5 * voxel_size, result_icp.transformation
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
        max_correspondence_distance=voxel_size * 1.5,
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
