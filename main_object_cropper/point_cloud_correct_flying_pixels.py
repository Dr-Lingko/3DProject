import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

def correct_flying_pixels(pcd, epsilon=np.deg2rad(30), nb_neighbors=20, std_ratio=2.0):
    # 估算法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    camera_dir = np.array([0, 0, 1])

    # 找离群点
    _, ind_inlier = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    all_indices = np.arange(len(points))
    outlier_indices = np.setdiff1d(all_indices, ind_inlier)

    # 在离群点中判定飞行像素
    cos_angles = np.dot(normals[outlier_indices], camera_dir) / (np.linalg.norm(normals[outlier_indices], axis=1) * np.linalg.norm(camera_dir))
    angles = np.arccos(np.clip(cos_angles, -1, 1))
    flying_mask = angles > epsilon
    flying_indices = outlier_indices[flying_mask]
    normal_indices = ind_inlier

    # 修正飞行像素
    tree = cKDTree(points[normal_indices])
    corrected_points = points.copy()
    if len(flying_indices) > 0 and len(normal_indices) > 0:
        dists, idxs = tree.query(points[flying_indices], k=1)
        for i, idx in enumerate(flying_indices):
            nearest_idx = normal_indices[idxs[i]]
            pt = points[nearest_idx]
            nt = normals[nearest_idx]
            a, b, c = nt
            d = -(a * pt[0] + b * pt[1] + c * pt[2])
            pf = points[idx]
            t = -(a * pf[0] + b * pf[1] + c * pf[2] + d) / (a * camera_dir[0] + b * camera_dir[1] + c * camera_dir[2])
            pf_proj = pf + t * camera_dir
            corrected_points[idx] = pf_proj

    corrected_pcd = o3d.geometry.PointCloud()
    corrected_pcd.points = o3d.utility.Vector3dVector(corrected_points)
    corrected_pcd.normals = o3d.utility.Vector3dVector(normals)
    return corrected_pcd
