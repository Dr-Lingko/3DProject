import open3d as o3d
import numpy as np

def plane_segmentation(filtered_pcd):

    if len(filtered_pcd.points) >= 3:
        plane_model, inliers = filtered_pcd.segment_plane(
            distance_threshold=10,
            ransac_n=3,
            num_iterations=1000
        )
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        inlier_cloud = filtered_pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = filtered_pcd.select_by_index(inliers, invert=True)
        # o3d.visualization.draw([inlier_cloud, outlier_cloud])
    else:
        print("过滤后点云数量不足，无法进行平面分割。")

    return outlier_cloud