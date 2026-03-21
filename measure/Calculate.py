from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d


@dataclass
class SliceResult:
	max_perimeter: Optional[float]
	perimeters: List[float]
	segment_count: int
	note: Optional[str]


def _unit_normal(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
	n = np.cross(p2 - p1, p3 - p1)
	norm = np.linalg.norm(n)
	if norm == 0:
		raise ValueError("Plane points are collinear; cannot define a plane.")
	return n / norm


def _segment_key(point: np.ndarray, tol: float) -> Tuple[int, int, int]:
	return tuple(np.round(point / tol).astype(int).tolist())


def _collect_plane_segments(
	vertices: np.ndarray,
	triangles: np.ndarray,
	p1: np.ndarray,
	normal: np.ndarray,
	eps: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
	segments: List[Tuple[np.ndarray, np.ndarray]] = []
	for tri in triangles:
		v0, v1, v2 = vertices[tri]
		d0 = float(np.dot(normal, v0 - p1))
		d1 = float(np.dot(normal, v1 - p1))
		d2 = float(np.dot(normal, v2 - p1))

		if max(d0, d1, d2) < -eps or min(d0, d1, d2) > eps:
			continue

		points: List[np.ndarray] = []
		edges = [(v0, v1, d0, d1), (v1, v2, d1, d2), (v2, v0, d2, d0)]
		for a, b, da, db in edges:
			if abs(da) <= eps and abs(db) <= eps:
				# Edge lies on plane; skip to avoid ambiguous coplanar slices.
				continue
			if abs(da) <= eps:
				points.append(a)
				continue
			if abs(db) <= eps:
				points.append(b)
				continue
			if da * db < 0:
				t = da / (da - db)
				points.append(a + t * (b - a))

		if len(points) >= 2:
			# Keep only two distinct points per triangle.
			unique = []
			for pt in points:
				if not any(np.allclose(pt, existing, atol=eps) for existing in unique):
					unique.append(pt)
			if len(unique) >= 2:
				segments.append((unique[0], unique[1]))

	return segments


def _build_graph(
	segments: Sequence[Tuple[np.ndarray, np.ndarray]], tol: float
) -> Tuple[List[np.ndarray], List[Tuple[int, int]], List[List[int]]]:
	node_index: Dict[Tuple[int, int, int], int] = {}
	nodes: List[np.ndarray] = []
	edges: List[Tuple[int, int]] = []

	def get_node(point: np.ndarray) -> int:
		key = _segment_key(point, tol)
		if key in node_index:
			return node_index[key]
		idx = len(nodes)
		node_index[key] = idx
		nodes.append(point)
		return idx

	for a, b in segments:
		i = get_node(a)
		j = get_node(b)
		if i != j:
			edges.append((i, j))

	adjacency: List[List[int]] = [[] for _ in nodes]
	for i, j in edges:
		adjacency[i].append(j)
		adjacency[j].append(i)

	return nodes, edges, adjacency


def _component_perimeters(
	nodes: List[np.ndarray],
	edges: List[Tuple[int, int]],
	adjacency: List[List[int]],
) -> List[float]:
	n = len(nodes)
	seen = [False] * n
	perimeters: List[float] = []

	edge_set = {(min(i, j), max(i, j)) for i, j in edges}

	for start in range(n):
		if seen[start] or not adjacency[start]:
			continue
		stack = [start]
		seen[start] = True
		component: List[int] = []
		while stack:
			node = stack.pop()
			component.append(node)
			for neighbor in adjacency[node]:
				if not seen[neighbor]:
					seen[neighbor] = True
					stack.append(neighbor)

		degrees_ok = all(len(adjacency[node]) == 2 for node in component)
		if not degrees_ok:
			continue

		perimeter = 0.0
		for i, j in edge_set:
			if i in component and j in component:
				perimeter += float(np.linalg.norm(nodes[i] - nodes[j]))
		if perimeter > 0:
			perimeters.append(perimeter)

	return perimeters


def _extract_closed_loops(
	nodes: List[np.ndarray],
	adjacency: List[List[int]],
) -> List[List[np.ndarray]]:
	seen = [False] * len(nodes)
	loops: List[List[np.ndarray]] = []

	for start in range(len(nodes)):
		if seen[start] or not adjacency[start]:
			continue
		stack = [start]
		seen[start] = True
		component: List[int] = []
		while stack:
			node = stack.pop()
			component.append(node)
			for neighbor in adjacency[node]:
				if not seen[neighbor]:
					seen[neighbor] = True
					stack.append(neighbor)

		if not all(len(adjacency[node]) == 2 for node in component):
			continue

		ordered: List[int] = []
		current = component[0]
		prev = -1
		while True:
			ordered.append(current)
			neighbors = adjacency[current]
			next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
			prev, current = current, next_node
			if current == ordered[0]:
				break

		loops.append([nodes[i] for i in ordered])

	return loops


def slice_mesh_perimeter(
	mesh_path: str,
	p1: Sequence[float],
	p2: Sequence[float],
	p3: Sequence[float],
	tol: float = 1e-6,
) -> SliceResult:
	mesh = o3d.io.read_triangle_mesh(mesh_path)
	if mesh.is_empty() or len(mesh.triangles) == 0:
		return SliceResult(None, [], 0, "Mesh has no triangles.")

	vertices = np.asarray(mesh.vertices)
	triangles = np.asarray(mesh.triangles)

	p1_np = np.asarray(p1, dtype=float)
	p2_np = np.asarray(p2, dtype=float)
	p3_np = np.asarray(p3, dtype=float)
	normal = _unit_normal(p1_np, p2_np, p3_np)

	segments = _collect_plane_segments(vertices, triangles, p1_np, normal, tol)
	if not segments:
		return SliceResult(None, [], 0, "No intersection segments found.")

	nodes, edges, adjacency = _build_graph(segments, tol)
	perimeters = _component_perimeters(nodes, edges, adjacency)
	max_perimeter = max(perimeters) if perimeters else None

	note = None
	if max_perimeter is None:
		note = "Intersection exists but no closed loop was detected."

	return SliceResult(max_perimeter, perimeters, len(segments), note)


def _visualize_loops(
	mesh_path: str,
	loops: List[List[np.ndarray]],
) -> None:
	if not loops:
		print("No closed loops to visualize.")
		return

	mesh = o3d.io.read_triangle_mesh(mesh_path)
	mesh.compute_vertex_normals()
	mesh.paint_uniform_color([0.85, 0.85, 0.85])
	geometries: List[o3d.geometry.Geometry] = [mesh]

	palette = [
		[0.95, 0.35, 0.2],
		[0.2, 0.75, 0.35],
		[0.2, 0.5, 0.9],
		[0.85, 0.7, 0.2],
		[0.7, 0.3, 0.8],
	]

	for idx, loop in enumerate(loops):
		points = np.asarray(loop, dtype=float)
		lines = [[i, (i + 1) % len(points)] for i in range(len(points))]
		line_set = o3d.geometry.LineSet(
			points=o3d.utility.Vector3dVector(points),
			lines=o3d.utility.Vector2iVector(lines),
		)
		color = palette[idx % len(palette)]
		line_set.paint_uniform_color(color)
		geometries.append(line_set)

	frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
	geometries.append(frame)

	vis = o3d.visualization.Visualizer()
	vis.create_window(window_name="Closed Loops", width=1200, height=900)
	for geom in geometries:
		vis.add_geometry(geom)
	render_option = vis.get_render_option()
	render_option.background_color = np.asarray([1.0, 1.0, 1.0])
	render_option.line_width = 4.0
	vis.run()
	vis.destroy_window()


def _parse_point(values: List[str]) -> Tuple[float, float, float]:
	if len(values) != 3:
		raise ValueError("Point must have exactly 3 values.")
	return float(values[0]), float(values[1]), float(values[2])


def _axis_index_from_name(axis: str) -> int:
	axis_map = {"x": 0, "y": 1, "z": 2}
	if axis not in axis_map:
		raise ValueError("axis must be 'x', 'y', 'z', or 'auto'.")
	return axis_map[axis]


def _choose_axis_from_bbox(vertices: np.ndarray) -> int:
	bbox_min = vertices.min(axis=0)
	bbox_max = vertices.max(axis=0)
	spans = bbox_max - bbox_min
	return int(np.argmax(spans))


def _plane_points_from_axis(
	center: np.ndarray,
	axis_index: int,
	height_value: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	point_on_plane = center.copy()
	point_on_plane[axis_index] = height_value
	axis_list = [0, 1, 2]
	axis_list.remove(axis_index)
	vec_a = np.zeros(3)
	vec_b = np.zeros(3)
	vec_a[axis_list[0]] = 1.0
	vec_b[axis_list[1]] = 1.0
	return point_on_plane, point_on_plane + vec_a, point_on_plane + vec_b


def _pick_points_from_mesh(mesh: o3d.geometry.TriangleMesh) -> List[np.ndarray]:
	pcd = o3d.geometry.PointCloud()
	pcd.points = mesh.vertices
	pcd.paint_uniform_color([0.5, 0.5, 0.5])

	vis = o3d.visualization.VisualizerWithEditing()
	vis.create_window(window_name="Mesh slice - pick 3 points", width=1200, height=800)
	vis.add_geometry(pcd)

	print("=" * 60)
	print("Pick 3 points on the mesh vertices")
	print("Shift + left click: pick points")
	print("Q or ESC: close window after picking")
	print("=" * 60)

	vis.run()
	vis.destroy_window()
	indices = vis.get_picked_points()
	if len(indices) < 3:
		raise ValueError(f"Only picked {len(indices)} point(s). Need 3 points.")

	points = np.asarray(mesh.vertices)
	return [points[i] for i in indices[:3]]


def _estimate_bpa_radii(pcd: o3d.geometry.PointCloud) -> List[float]:
	dists = pcd.compute_nearest_neighbor_distance()
	if len(dists) == 0:
		return [1.0]
	avg_dist = float(np.mean(dists))
	return [avg_dist * 1.5, avg_dist * 2.5, avg_dist * 3.5]


def _reconstruct_mesh_from_point_cloud(
	pcd_path: str,
	output_path: str,
	method: str = "poisson",
	poisson_depth: int = 9,
	poisson_density_quantile: float = 0.02,
	alpha: float = 5.0,
	voxel_size: Optional[float] = None,
	remove_outliers: bool = True,
	nb_neighbors: int = 30,
	std_ratio: float = 2.0,
	orient_normals: bool = True,
	normal_radius: float = 10.0,
	normal_max_nn: int = 50,
) -> str:
	print("Loading point cloud:", pcd_path)
	pcd = o3d.io.read_point_cloud(pcd_path)
	if pcd.is_empty():
		raise ValueError("Point cloud is empty.")
	print("Point cloud loaded. Points:", len(pcd.points))

	if voxel_size is not None and voxel_size > 0:
		print("Downsampling with voxel size:", voxel_size)
		pcd = pcd.voxel_down_sample(voxel_size)
		print("Points after downsample:", len(pcd.points))

	if remove_outliers:
		print("Removing outliers...")
		pcd, _ = pcd.remove_statistical_outlier(
			nb_neighbors=nb_neighbors, std_ratio=std_ratio
		)
		print("Points after outlier removal:", len(pcd.points))

	print("Estimating normals...")
	pcd.estimate_normals(
		search_param=o3d.geometry.KDTreeSearchParamHybrid(
			radius=normal_radius, max_nn=normal_max_nn
		)
	)
	if orient_normals:
		print("Orienting normals consistently...")
		pcd.orient_normals_consistent_tangent_plane(k=normal_max_nn)

	if method == "poisson":
		print("Reconstructing mesh with Poisson...")
		mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
			pcd, depth=poisson_depth
		)
		print("Poisson reconstruction done. Vertices:", len(mesh.vertices))
		if poisson_density_quantile > 0:
			densities = np.asarray(densities)
			threshold = float(np.quantile(densities, poisson_density_quantile))
			mask = densities < threshold
			print("Removing low-density vertices...")
			mesh.remove_vertices_by_mask(mask)
	elif method == "bpa":
		print("Reconstructing mesh with BPA...")
		radii = _estimate_bpa_radii(pcd)
		mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
			pcd, o3d.utility.DoubleVector(radii)
		)
	elif method == "alpha":
		print("Reconstructing mesh with Alpha Shape...")
		mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
			pcd, alpha
		)
	else:
		raise ValueError("reconstruct_method must be 'poisson', 'bpa', or 'alpha'.")

	print("Writing mesh:", output_path)
	o3d.io.write_triangle_mesh(output_path, mesh)
	print("Mesh written.")
	return output_path


def main() -> None:
	config = {
		"point_cloud_path": r"E:\3DProject\D3\super_sqz.ply",
		"reconstruct_method": "poisson",
		"reconstruct_output_path": r"E:\3DProject\D3\reconstructed_mesh.ply",
		"poisson_depth": 10,
		"poisson_density_quantile": 0.005,
		"alpha": 5.0,
		"voxel_size": None,
		"remove_outliers": True,
		"nb_neighbors": 30,
		"std_ratio": 2.0,
		"orient_normals": True,
		"normal_radius": 10.0,
		"normal_max_nn": 50,
		"interactive": False,
		"check_mesh": True,
		"check_self_intersecting": False,
		"visualize_loops": True,
		"axis": "auto",
		"waist_height_mm": None,
		"waist_height_ratio": 0.5,
		"p1": (0.0, 0.0, 0.0),
		"p2": (1.0, 0.0, 0.0),
		"p3": (0.0, 1.0, 0.0),
		"tol": 1e-6,
	}

	mesh_path = _reconstruct_mesh_from_point_cloud(
		config["point_cloud_path"],
		config["reconstruct_output_path"],
		method=config["reconstruct_method"],
		poisson_depth=config["poisson_depth"],
		poisson_density_quantile=config["poisson_density_quantile"],
		alpha=config["alpha"],
		voxel_size=config["voxel_size"],
		remove_outliers=config["remove_outliers"],
		nb_neighbors=config["nb_neighbors"],
		std_ratio=config["std_ratio"],
		orient_normals=config["orient_normals"],
		normal_radius=config["normal_radius"],
		normal_max_nn=config["normal_max_nn"],
	)

	if config["check_mesh"]:
		mesh = o3d.io.read_triangle_mesh(mesh_path)
		print("mesh_empty:", mesh.is_empty())
		print("triangles:", len(mesh.triangles))
		if not mesh.is_empty() and len(mesh.triangles) > 0:
			print("is_edge_manifold:", mesh.is_edge_manifold(allow_boundary_edges=False))
			print(
				"is_edge_manifold_allow_boundary:",
				mesh.is_edge_manifold(allow_boundary_edges=True),
			)
			print("is_vertex_manifold:", mesh.is_vertex_manifold())
			print("is_watertight:", mesh.is_watertight())
			print("is_orientable:", mesh.is_orientable())
			if config["check_self_intersecting"]:
				print("is_self_intersecting:", mesh.is_self_intersecting())

	if config["interactive"]:
		mesh = o3d.io.read_triangle_mesh(mesh_path)
		if mesh.is_empty() or len(mesh.triangles) == 0:
			raise ValueError("Mesh has no triangles.")
		print("Opening pick window...")
		p1, p2, p3 = _pick_points_from_mesh(mesh)
		print("Pick window closed.")
	else:
		mesh = o3d.io.read_triangle_mesh(mesh_path)
		vertices = np.asarray(mesh.vertices)
		if config["axis"] == "auto":
			axis_index = _choose_axis_from_bbox(vertices)
		else:
			axis_index = _axis_index_from_name(config["axis"])

		bbox_min = vertices.min(axis=0)
		bbox_max = vertices.max(axis=0)
		if config["waist_height_mm"] is None:
			height_value = float(
				bbox_min[axis_index]
				+ config["waist_height_ratio"]
				* (bbox_max[axis_index] - bbox_min[axis_index])
			)
		else:
			height_value = float(config["waist_height_mm"])

		center = (bbox_min + bbox_max) / 2.0
		p1, p2, p3 = _plane_points_from_axis(center, axis_index, height_value)
		print("Auto axis index:", axis_index)
		print("Slice height:", height_value)

	result = slice_mesh_perimeter(mesh_path, p1, p2, p3, tol=config["tol"])
	if config["visualize_loops"]:
		mesh = o3d.io.read_triangle_mesh(mesh_path)
		vertices = np.asarray(mesh.vertices)
		triangles = np.asarray(mesh.triangles)
		p1_np = np.asarray(p1, dtype=float)
		p2_np = np.asarray(p2, dtype=float)
		p3_np = np.asarray(p3, dtype=float)
		normal = _unit_normal(p1_np, p2_np, p3_np)
		segments = _collect_plane_segments(vertices, triangles, p1_np, normal, config["tol"])
		if segments:
			nodes, _, adjacency = _build_graph(segments, config["tol"])
			loops = _extract_closed_loops(nodes, adjacency)
			_visualize_loops(mesh_path, loops)
	if result.max_perimeter is None:
		print("No closed perimeter found.")
	else:
		print(f"Max perimeter: {result.max_perimeter:.6f}")
	if result.perimeters:
		print("All closed perimeters:", ", ".join(f"{p:.6f}" for p in result.perimeters))
	if result.note:
		print(result.note)


if __name__ == "__main__":
	main()
