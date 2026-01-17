import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def dbscan_clustering(pcd):

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=10, min_points=10, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw([pcd])

    # 统计每个聚类的点数，筛选点数最多的前两类
    unique_labels = np.unique(labels[labels >= 0])
    label_counts = [(label, np.sum(labels == label)) for label in unique_labels]
    label_counts_sorted = sorted(label_counts, key=lambda x: x[1], reverse=True)

    if len(label_counts_sorted) < 2:
        print("聚类数量不足，无法筛选主体和背景。")
        return None

    label1, label2 = label_counts_sorted[0][0], label_counts_sorted[1][0]
    idx1 = np.where(labels == label1)[0]
    idx2 = np.where(labels == label2)[0]

    # 计算两类的质心
    points = np.asarray(pcd.points)
    centroid1 = points[idx1].mean(axis=0)
    centroid2 = points[idx2].mean(axis=0)
    print(f"最大类标签: {label1}, 点数: {len(idx1)}, 质心: {centroid1}")
    print(f"次大类标签: {label2}, 点数: {len(idx2)}, 质心: {centroid2}")

    # 以z轴为“正方向”，z值较大的为主体
    if centroid1[2] > centroid2[2]:
        main_indices = idx1
        main_label = label1
    else:
        main_indices = idx2
        main_label = label2

    print(f"自动判定主体聚类标签: {main_label}")
    main_object = pcd.select_by_index(main_indices)

    return main_object