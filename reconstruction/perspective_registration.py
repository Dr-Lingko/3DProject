import open3d as o3d
import numpy as np

import os
import re
import argparse


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def parse_angles_deg(text):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("--angles cannot be empty")
    return [float(p) for p in parts]


def parse_point_or_auto(text):
    s = text.strip().lower()
    if s == "auto":
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("--axis-point must be 'auto' or 'x,y,z'")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)


def rotation_matrix_from_axis_angle(axis, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    if axis == "x":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
            dtype=np.float64,
        )
    if axis == "y":
        return np.array(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
            dtype=np.float64,
        )
    if axis == "z":
        return np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    raise ValueError("axis must be one of: x, y, z")


def compose_transform(rotation, translation=None):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rotation
    if translation is not None:
        t[:3, 3] = translation
    return t


def translation_matrix(vec3):
    t = np.eye(4, dtype=np.float64)
    t[:3, 3] = vec3
    return t


def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def estimate_orbit_radius_auto(pcds):
    centers = [np.asarray(p.get_center(), dtype=np.float64) for p in pcds if len(p.points) > 0]
    if not centers:
        return 1.0
    norms = [float(np.linalg.norm(c)) for c in centers]
    r = float(np.median(norms))
    return max(r, 1e-3)


def _fit_circle_2d(points2d):
    # Solve x^2 + y^2 + a x + b y + c = 0 for robust center estimation.
    x = points2d[:, 0]
    y = points2d[:, 1]
    a_mat = np.column_stack([x, y, np.ones_like(x)])
    b_vec = -(x * x + y * y)
    sol, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    a, b, _ = sol
    center = np.array([-a / 2.0, -b / 2.0], dtype=np.float64)
    return center


def estimate_axis_point_from_centers(pcds, axis):
    centers = [np.asarray(p.get_center(), dtype=np.float64) for p in pcds if len(p.points) > 0]
    if len(centers) < 3:
        return np.zeros(3, dtype=np.float64)
    c = np.vstack(centers)

    if axis == "x":
        circle_center = _fit_circle_2d(c[:, [1, 2]])
        return np.array([float(np.mean(c[:, 0])), circle_center[0], circle_center[1]], dtype=np.float64)
    if axis == "y":
        circle_center = _fit_circle_2d(c[:, [0, 2]])
        return np.array([circle_center[0], float(np.mean(c[:, 1])), circle_center[1]], dtype=np.float64)
    circle_center = _fit_circle_2d(c[:, [0, 1]])
    return np.array([circle_center[0], circle_center[1], float(np.mean(c[:, 2]))], dtype=np.float64)


def camera_position_on_orbit(radius, axis, angle_deg):
    base = np.array([0.0, 0.0, radius], dtype=np.float64)
    r = rotation_matrix_from_axis_angle(axis, angle_deg)
    return r @ base


def camera_to_world_lookat(eye, target, camera_forward_sign=1.0):
    if camera_forward_sign >= 0:
        z_axis = _normalize(target - eye)
    else:
        z_axis = _normalize(eye - target)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-9:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        x_axis = np.cross(up, z_axis)
    x_axis = _normalize(x_axis)
    y_axis = _normalize(np.cross(z_axis, x_axis))

    r_wc = np.column_stack([x_axis, y_axis, z_axis])
    return compose_transform(r_wc, eye)


def build_camera_orbit_prior_transforms(angles, axis, radius, angle_sign=1.0, camera_forward_sign=1.0):
    # Build transforms from each camera frame to a common world frame.
    world_transforms = []
    target = np.zeros(3, dtype=np.float64)
    ref_angle = float(angles[0])
    for a in angles:
        rel = angle_sign * (float(a) - ref_angle)
        eye = camera_position_on_orbit(radius, axis, rel)
        t_cw = camera_to_world_lookat(eye, target, camera_forward_sign=camera_forward_sign)
        world_transforms.append(t_cw)

    # Convert to transforms mapping each cloud directly into the reference-camera frame.
    t_w_ref = np.linalg.inv(world_transforms[0])
    return [t_w_ref @ t_cw for t_cw in world_transforms]


def build_turntable_prior_transforms(angles, axis, axis_point, angle_sign=1.0):
    priors = [np.eye(4, dtype=np.float64)]
    ref_angle = float(angles[0])
    for a in angles[1:]:
        rel = angle_sign * (float(a) - ref_angle)
        r = rotation_matrix_from_axis_angle(axis, rel)
        t = translation_matrix(axis_point) @ compose_transform(r) @ translation_matrix(-axis_point)
        priors.append(t)
    return priors


def estimate_voxel_size_auto(pcds):
    all_pts = []
    for pcd in pcds:
        pts = np.asarray(pcd.points)
        if pts.size > 0:
            all_pts.append(pts)
    if not all_pts:
        return 1.0
    stacked = np.vstack(all_pts)
    diag = np.linalg.norm(stacked.max(axis=0) - stacked.min(axis=0))
    voxel = diag / 250.0
    return max(voxel, 1e-4)


def prepare_icp_cloud(pcd, voxel_size):
    down = pcd.voxel_down_sample(voxel_size)
    radius = voxel_size * 2.0
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    return down


def clamp_translation(transform, max_translation):
    if max_translation <= 0.0:
        return transform
    t = transform.copy()
    v = t[:3, 3]
    n = np.linalg.norm(v)
    if n > max_translation and n > 1e-12:
        t[:3, 3] = v * (max_translation / n)
    return t


def rotation_angle_deg_from_matrix(r):
    cos_theta = (np.trace(r) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def axis_angle_from_rotation(r):
    cos_theta = (np.trace(r) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
    denom = 2.0 * np.sin(theta)
    if abs(denom) < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
    axis = np.array([
        r[2, 1] - r[1, 2],
        r[0, 2] - r[2, 0],
        r[1, 0] - r[0, 1],
    ], dtype=np.float64) / denom
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
    return axis / n, theta


def rotation_from_axis_angle(axis, theta):
    k = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=np.float64)
    i = np.eye(3, dtype=np.float64)
    return i + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def clamp_transform_delta(delta, max_rotation_deg, max_translation):
    out = delta.copy()
    out = clamp_translation(out, max_translation)
    if max_rotation_deg <= 0.0:
        return out
    r = out[:3, :3]
    axis, theta = axis_angle_from_rotation(r)
    max_theta = np.deg2rad(max_rotation_deg)
    if theta > max_theta:
        out[:3, :3] = rotation_from_axis_angle(axis, max_theta)
    return out


def register_with_view_prior(source, target, init_transform, icp_threshold, voxel_size, max_iteration, max_translation):
    source_down = prepare_icp_cloud(source, voxel_size)
    target_down = prepare_icp_cloud(target, voxel_size)

    loss = o3d.pipelines.registration.TukeyLoss(k=icp_threshold)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)

    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        icp_threshold,
        init_transform,
        estimation,
        criteria,
    )
    result.transformation = clamp_translation(result.transformation, max_translation)
    return result


def center_alignment_transform(source_pcd, target_pcd):
    src_c = np.asarray(source_pcd.get_center(), dtype=np.float64)
    tgt_c = np.asarray(target_pcd.get_center(), dtype=np.float64)
    return translation_matrix(tgt_c - src_c)


def load_point_clouds(folder, exclude_names=None):
    exclude = set(exclude_names or [])
    files = [f for f in os.listdir(folder) if f.lower().endswith(".ply")]
    files = [f for f in files if f not in exclude]
    files.sort(key=natural_sort_key)
    paths = [os.path.join(folder, f) for f in files]
    pcds = [o3d.io.read_point_cloud(p) for p in paths]
    return files, pcds


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Viewpoint-prior point-cloud registration. "
            "Use known capture angles (e.g. 0,90,180,270) as initialization and refine with ICP."
        )
    )
    parser.add_argument("--folder", required=True, help="Folder containing .ply point clouds")
    parser.add_argument("--angles", default="0,90,180,270", help="Capture angles in degrees, comma-separated")
    parser.add_argument(
        "--prior-model",
        default="turntable-object",
        choices=["turntable-object", "camera-orbit"],
        help="Geometric prior model",
    )
    parser.add_argument("--axis", default="auto", choices=["auto", "x", "y", "z"], help="Rotation axis for viewpoint change")
    parser.add_argument(
        "--axis-point",
        default="auto",
        help="Axis point for turntable-object model; 'auto' or 'x,y,z'",
    )
    parser.add_argument(
        "--forward-axis",
        default="auto",
        choices=["auto", "plus-z", "minus-z"],
        help="Camera forward direction in camera coordinates",
    )
    parser.add_argument(
        "--view-mode",
        default="camera-around-object",
        choices=["camera-around-object", "object-around-camera"],
        help="Geometric convention of capture motion",
    )
    parser.add_argument(
        "--orbit-radius",
        type=float,
        default=0.0,
        help="Camera orbit radius; 0 for auto-estimation from cloud centers",
    )
    parser.add_argument(
        "--disable-hypothesis-search",
        action="store_true",
        help="Disable automatic sign/forward-direction hypothesis search",
    )
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Downsample voxel size; 0 for auto")
    parser.add_argument("--icp-threshold", type=float, default=0.0, help="ICP correspondence threshold; 0 for auto")
    parser.add_argument("--max-iteration", type=int, default=80, help="Max ICP iterations")
    parser.add_argument(
        "--max-translation",
        type=float,
        default=0.0,
        help="Optional translation clamp in ICP result (same unit as point cloud, 0 disables)",
    )
    parser.add_argument(
        "--max-refine-rotation-deg",
        type=float,
        default=15.0,
        help="Maximum allowed ICP refinement rotation delta in degrees",
    )
    parser.add_argument(
        "--max-refine-translation",
        type=float,
        default=0.0,
        help="Maximum allowed ICP refinement translation delta; 0 means auto (2 * voxel_size)",
    )
    parser.add_argument(
        "--lock-prior-rotation",
        action="store_true",
        help="Keep prior rotation fixed and allow only translation refinement",
    )
    parser.add_argument(
        "--prior-only",
        action="store_true",
        help="Use only viewpoint prior (and center correction), skip ICP refinement",
    )
    parser.add_argument("--output", default="merged_view_prior.ply", help="Output merged point cloud filename")
    parser.add_argument("--visualize", action="store_true", help="Visualize merged point cloud")
    return parser.parse_args()

def main():
    args = parse_args()

    files, pcds = load_point_clouds(args.folder, exclude_names=[args.output])
    if len(pcds) < 2:
        raise ValueError("Need at least 2 point clouds")

    angles = parse_angles_deg(args.angles)
    if len(angles) > len(pcds):
        raise ValueError(
            f"Number of angles ({len(angles)}) cannot exceed number of point clouds ({len(pcds)})"
        )
    if len(angles) < len(pcds):
        print(
            f"Warning: found {len(pcds)} point clouds but only {len(angles)} angles; "
            f"using first {len(angles)} files after sorting."
        )
        files = files[: len(angles)]
        pcds = pcds[: len(angles)]

    voxel_size = args.voxel_size if args.voxel_size > 0 else estimate_voxel_size_auto(pcds)
    icp_threshold = args.icp_threshold if args.icp_threshold > 0 else voxel_size * 2.5
    orbit_radius = args.orbit_radius if args.orbit_radius > 0 else estimate_orbit_radius_auto(pcds)
    max_refine_translation = args.max_refine_translation if args.max_refine_translation > 0 else voxel_size * 2.0

    print(f"Input point clouds: {files}")
    print(f"Angles(deg): {angles}")
    print(
        f"Axis: {args.axis}, view_mode={args.view_mode}, "
        f"voxel_size={voxel_size:.6f}, icp_threshold={icp_threshold:.6f}, orbit_radius={orbit_radius:.6f}"
    )
    print(
        f"Refine limits: max_rot={args.max_refine_rotation_deg:.2f} deg, "
        f"max_trans={max_refine_translation:.6f}"
    )
    print(
        f"Refine mode: lock_prior_rotation={args.lock_prior_rotation}, prior_only={args.prior_only}"
    )

    ref = pcds[0]
    ref_down = prepare_icp_cloud(ref, voxel_size)

    if args.view_mode == "camera-around-object":
        base_angle_sign = 1.0
    else:
        base_angle_sign = -1.0

    if args.forward_axis == "plus-z":
        forward_candidates = [1.0]
    elif args.forward_axis == "minus-z":
        forward_candidates = [-1.0]
    else:
        forward_candidates = [1.0, -1.0]

    if args.disable_hypothesis_search:
        angle_sign_candidates = [base_angle_sign]
    else:
        angle_sign_candidates = [base_angle_sign, -base_angle_sign]

    axis_candidates = [args.axis] if args.axis != "auto" else ["x", "y", "z"]

    best_score = -1.0
    best_h = None
    best_prior_transforms = None
    best_axis_point = None

    best_axis = None
    for axis_name in axis_candidates:
        axis_point_cli = parse_point_or_auto(args.axis_point)
        if args.prior_model == "turntable-object":
            axis_point = axis_point_cli if axis_point_cli is not None else estimate_axis_point_from_centers(pcds, axis_name)
            current_forward_candidates = [1.0]
        else:
            axis_point = None
            current_forward_candidates = forward_candidates

        for angle_sign in angle_sign_candidates:
            for forward_sign in current_forward_candidates:
                if args.prior_model == "turntable-object":
                    priors = build_turntable_prior_transforms(
                        angles,
                        axis=axis_name,
                        axis_point=axis_point,
                        angle_sign=angle_sign,
                    )
                else:
                    priors = build_camera_orbit_prior_transforms(
                        angles,
                        axis=axis_name,
                        radius=orbit_radius,
                        angle_sign=angle_sign,
                        camera_forward_sign=forward_sign,
                    )
                scores = []
                for i in range(1, len(pcds)):
                    src = o3d.geometry.PointCloud(pcds[i])
                    src.transform(priors[i])
                    if args.prior_model == "camera-orbit":
                        center_t = center_alignment_transform(src, ref)
                        src.transform(center_t)
                    src_down = prepare_icp_cloud(src, voxel_size)
                    ev = o3d.pipelines.registration.evaluate_registration(
                        src_down,
                        ref_down,
                        icp_threshold,
                        np.eye(4, dtype=np.float64),
                    )
                    scores.append(float(ev.fitness))
                score = float(np.mean(scores)) if scores else 0.0
                print(
                    f"Hypothesis model={args.prior_model}, axis={axis_name}, angle_sign={angle_sign:+.0f}, "
                    f"forward_sign={forward_sign:+.0f}, "
                    f"mean_fitness={score:.4f}"
                )
                if score > best_score:
                    best_score = score
                    best_h = (angle_sign, forward_sign)
                    best_axis = axis_name
                    best_axis_point = axis_point
                    best_prior_transforms = []
                    for i, p in enumerate(priors):
                        if i == 0:
                            best_prior_transforms.append(np.eye(4, dtype=np.float64))
                            continue
                        if args.prior_model == "camera-orbit":
                            src = o3d.geometry.PointCloud(pcds[i])
                            src.transform(p)
                            c_t = center_alignment_transform(src, ref)
                            best_prior_transforms.append(c_t @ p)
                        else:
                            best_prior_transforms.append(p)

    print(
        f"Selected hypothesis: model={args.prior_model}, axis={best_axis}, angle_sign={best_h[0]:+.0f}, "
        f"forward_sign={best_h[1]:+.0f}, mean_fitness={best_score:.4f}"
    )
    if best_axis_point is not None:
        print(f"Selected axis point: [{best_axis_point[0]:.4f}, {best_axis_point[1]:.4f}, {best_axis_point[2]:.4f}]")

    transforms = [np.eye(4, dtype=np.float64)]
    for i in range(1, len(pcds)):
        prior_t = best_prior_transforms[i]
        if args.prior_only:
            refine_fitness = -1.0
            refine_rmse = -1.0
            delta_t = np.eye(4, dtype=np.float64)
        else:
            src_prior = o3d.geometry.PointCloud(pcds[i])
            src_prior.transform(prior_t)

            refine = register_with_view_prior(
                source=src_prior,
                target=ref,
                init_transform=np.eye(4, dtype=np.float64),
                icp_threshold=icp_threshold,
                voxel_size=voxel_size,
                max_iteration=args.max_iteration,
                max_translation=args.max_translation,
            )
            refine_fitness = float(refine.fitness)
            refine_rmse = float(refine.inlier_rmse)
            delta_t = clamp_transform_delta(
                refine.transformation,
                max_rotation_deg=args.max_refine_rotation_deg,
                max_translation=max_refine_translation,
            )
            if args.lock_prior_rotation:
                delta_t[:3, :3] = np.eye(3, dtype=np.float64)

        final_t = delta_t @ prior_t
        delta_rot_deg = rotation_angle_deg_from_matrix(delta_t[:3, :3])
        delta_trans = float(np.linalg.norm(delta_t[:3, 3]))
        transforms.append(final_t)
        print(
            f"[{i}] angle={angles[i]:.2f}, prior+icp fitness={refine_fitness:.4f}, "
            f"rmse={refine_rmse:.6f}, delta_rot={delta_rot_deg:.2f} deg, "
            f"delta_trans={delta_trans:.4f}"
        )

    merged = o3d.geometry.PointCloud()
    for pcd, t in zip(pcds, transforms):
        aligned = o3d.geometry.PointCloud(pcd)
        aligned.transform(t)
        merged += aligned

    merged = merged.voxel_down_sample(voxel_size * 0.6)
    out_path = os.path.join(args.folder, args.output)
    o3d.io.write_point_cloud(out_path, merged)
    print(f"Merged point cloud saved to: {out_path}")

    if args.visualize:
        o3d.visualization.draw_geometries([merged])

    return 0

if __name__ == "__main__":
    main()