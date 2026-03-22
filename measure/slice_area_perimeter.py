from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import open3d as o3d


@dataclass
class SliceFitResult:
	area: float
	perimeter: float
	hull_points_2d: np.ndarray
	hull_points_3d: np.ndarray
	plane_point: np.ndarray
	plane_normal: np.ndarray


def _unit_normal(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
	n = np.cross(p2 - p1, p3 - p1)
	norm = np.linalg.norm(n)
	if norm == 0:
		raise ValueError("Plane points are collinear; cannot define a plane.")
	return n / norm


def _pick_points_from_pcd(pcd: o3d.geometry.PointCloud) -> List[np.ndarray]:
	vis = o3d.visualization.VisualizerWithEditing()
	vis.create_window(window_name="Pick 3 points", width=1200, height=800)
	vis.add_geometry(pcd)

	print("=" * 60)
	print("Pick 3 points on the point cloud")
	print("Shift + left click: pick points")
	print("Q or ESC: close window after picking")
	print("=" * 60)

	vis.run()
	vis.destroy_window()
	indices = vis.get_picked_points()
	if len(indices) < 3:
		raise ValueError(f"Only picked {len(indices)} point(s). Need 3 points.")

	points = np.asarray(pcd.points)
	return [points[i] for i in indices[:3]]


def _extract_slice_points(
	points: np.ndarray,
	plane_point: np.ndarray,
	plane_normal: np.ndarray,
	distance_tol: float,
) -> np.ndarray:
	dists = np.dot(points - plane_point, plane_normal)
	mask = np.abs(dists) <= distance_tol
	return points[mask]


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	axis = np.array([1.0, 0.0, 0.0])
	if abs(np.dot(axis, normal)) > 0.9:
		axis = np.array([0.0, 1.0, 0.0])
	u = np.cross(normal, axis)
	u /= np.linalg.norm(u)
	v = np.cross(normal, u)
	return u, v


def _project_to_2d(
	points: np.ndarray,
	origin: np.ndarray,
	u: np.ndarray,
	v: np.ndarray,
) -> np.ndarray:
	rel = points - origin
	x = np.dot(rel, u)
	y = np.dot(rel, v)
	return np.stack([x, y], axis=1)


def _convex_hull_2d(points_2d: np.ndarray) -> np.ndarray:
	points = np.unique(points_2d, axis=0)
	if len(points) < 3:
		return points

	points = points[np.lexsort((points[:, 1], points[:, 0]))]

	def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
		return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

	lower: List[np.ndarray] = []
	for p in points:
		while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
			lower.pop()
		lower.append(p)

	upper: List[np.ndarray] = []
	for p in reversed(points):
		while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
			upper.pop()
		upper.append(p)

	hull = np.array(lower[:-1] + upper[:-1])
	return hull


def _polygon_area_perimeter(points_2d: np.ndarray) -> Tuple[float, float]:
	if len(points_2d) < 3:
		return 0.0, 0.0
	closed = np.vstack([points_2d, points_2d[0]])
	x = closed[:, 0]
	y = closed[:, 1]
	area = 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
	perimeter = float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))
	return area, perimeter


def _visualize_slice(
	pcd: o3d.geometry.PointCloud,
	slice_points: np.ndarray,
	hull_points_3d: np.ndarray,
) -> None:
	slice_pcd = o3d.geometry.PointCloud()
	slice_pcd.points = o3d.utility.Vector3dVector(slice_points)
	slice_pcd.paint_uniform_color([0.2, 0.6, 0.9])

	lines = [[i, (i + 1) % len(hull_points_3d)] for i in range(len(hull_points_3d))]
	line_set = o3d.geometry.LineSet(
		points=o3d.utility.Vector3dVector(hull_points_3d),
		lines=o3d.utility.Vector2iVector(lines),
	)
	line_set.paint_uniform_color([0.95, 0.35, 0.2])

	o3d.visualization.draw_geometries(
		[pcd, slice_pcd, line_set], window_name="Slice and Hull"
	)


def _visualize_hull_only(hull_points_3d: np.ndarray) -> None:
	lines = [[i, (i + 1) % len(hull_points_3d)] for i in range(len(hull_points_3d))]
	line_set = o3d.geometry.LineSet(
		points=o3d.utility.Vector3dVector(hull_points_3d),
		lines=o3d.utility.Vector2iVector(lines),
	)
	line_set.paint_uniform_color([0.95, 0.35, 0.2])

	vis = o3d.visualization.Visualizer()
	vis.create_window(window_name="Hull Only", width=900, height=700)
	vis.add_geometry(line_set)
	render_option = vis.get_render_option()
	render_option.background_color = np.asarray([1.0, 1.0, 1.0])
	render_option.line_width = 4.0
	vis.run()
	vis.destroy_window()


def compute_slice_area_perimeter(
	pcd_path: str,
	plane_points: Sequence[Sequence[float]],
	distance_tol: float,
	min_points: int,
	visualize: bool,
) -> SliceFitResult:
	pcd = o3d.io.read_point_cloud(pcd_path)
	if pcd.is_empty():
		raise ValueError("Point cloud is empty.")

	p1, p2, p3 = (np.asarray(p, dtype=float) for p in plane_points)
	normal = _unit_normal(p1, p2, p3)

	points = np.asarray(pcd.points)
	slice_points = _extract_slice_points(points, p1, normal, distance_tol)
	if len(slice_points) < min_points:
		raise ValueError("Not enough slice points to fit a shape.")

	u, v = _plane_basis(normal)
	points_2d = _project_to_2d(slice_points, p1, u, v)
	hull_2d = _convex_hull_2d(points_2d)
	area, perimeter = _polygon_area_perimeter(hull_2d)
	hull_3d = p1 + np.outer(hull_2d[:, 0], u) + np.outer(hull_2d[:, 1], v)

	if visualize and len(hull_2d) >= 3:
		_visualize_slice(pcd, slice_points, hull_3d)

	return SliceFitResult(area, perimeter, hull_2d, hull_3d, p1, normal)


def main() -> None:
	config = {
		"point_cloud_path": r"E:\3DProject\D3\super_wlr.ply",
		"distance_tol": 2.0,
		"min_points": 200,
		"visualize": True,
		"visualize_hull_only": True,
	}

	pcd = o3d.io.read_point_cloud(config["point_cloud_path"])
	if pcd.is_empty():
		raise ValueError("Point cloud is empty.")

	plane_points = _pick_points_from_pcd(pcd)
	result = compute_slice_area_perimeter(
		config["point_cloud_path"],
		plane_points,
		config["distance_tol"],
		config["min_points"],
		config["visualize"],
	)
	if config["visualize_hull_only"] and len(result.hull_points_3d) >= 3:
		_visualize_hull_only(result.hull_points_3d)

	print(f"Area: {result.area:.6f}")
	print(f"Perimeter: {result.perimeter:.6f}")


if __name__ == "__main__":
	main()
