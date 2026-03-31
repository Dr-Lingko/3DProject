import argparse
import ctypes
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import open3d as o3d


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))


FILTER_MAP = {
	"gaussian3": o3d.geometry.ImageFilterType.Gaussian3,
	"gaussian5": o3d.geometry.ImageFilterType.Gaussian5,
	"gaussian7": o3d.geometry.ImageFilterType.Gaussian7,
}


NUMPY_GAUSSIAN_KERNELS = {
	"gaussian3": np.array([1.0, 2.0, 1.0], dtype=np.float32) / 4.0,
	"gaussian5": np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0,
	"gaussian7": np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0], dtype=np.float32) / 64.0,
}


def _decode_char_array(values) -> str:
	raw = bytes(int(v) & 0xFF for v in values)
	return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()


def _matrix9_to_intrinsics(matrix9) -> Dict[str, float]:
	# Matrix layout is [fx,0,cx, 0,fy,cy, 0,0,1].
	return {
		"fx": float(matrix9[0]),
		"fy": float(matrix9[4]),
		"cx": float(matrix9[2]),
		"cy": float(matrix9[5]),
	}


def _read_intrinsics_from_sdk(index: int, serial: str) -> Dict[str, float]:
	from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd
	from Mv3dRgbdImport.Mv3dRgbdDefine import (
		CoordinateType_Depth,
		DeviceType_Ethernet,
		DeviceType_Ethernet_Vir,
		DeviceType_USB,
		DeviceType_USB_Vir,
		MV3D_RGBD_CALIB_INFO,
		MV3D_RGBD_DEVICE_INFO_LIST,
		MV3D_RGBD_FLOAT_Z_UNIT,
		MV3D_RGBD_OK,
		MV3D_RGBD_PARAM,
		ParamType_Float,
	)

	def check_ok(ret: int, action: str) -> None:
		if ret != MV3D_RGBD_OK:
			raise RuntimeError(f"{action} failed: ret=0x{ret:08X}")

	ret = Mv3dRgbd.MV3D_RGBD_Initialize()
	check_ok(ret, "MV3D_RGBD_Initialize")

	camera = Mv3dRgbd()
	try:
		device_num = ctypes.c_uint(0)
		ret = Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(
			DeviceType_Ethernet | DeviceType_USB | DeviceType_Ethernet_Vir | DeviceType_USB_Vir,
			ctypes.byref(device_num),
		)
		check_ok(ret, "MV3D_RGBD_GetDeviceNumber")
		if device_num.value == 0:
			raise RuntimeError("No MV3D RGBD device found.")

		device_list = MV3D_RGBD_DEVICE_INFO_LIST()
		ret = Mv3dRgbd.MV3D_RGBD_GetDeviceList(
			DeviceType_Ethernet | DeviceType_USB | DeviceType_Ethernet_Vir | DeviceType_USB_Vir,
			ctypes.pointer(device_list.DeviceInfo[0]),
			20,
			ctypes.byref(device_num),
		)
		check_ok(ret, "MV3D_RGBD_GetDeviceList")

		selected = index
		if serial:
			selected = -1
			for i in range(20):
				sn = _decode_char_array(device_list.DeviceInfo[i].chSerialNumber)
				if sn == serial:
					selected = i
					break
			if selected < 0:
				raise RuntimeError(f"Serial not found: {serial}")

		if selected < 0 or selected >= 20:
			raise RuntimeError(f"Invalid device index: {selected}")

		ret = camera.MV3D_RGBD_OpenDevice(ctypes.pointer(device_list.DeviceInfo[selected]))
		check_ok(ret, "MV3D_RGBD_OpenDevice")

		depth_calib = MV3D_RGBD_CALIB_INFO()
		ret = camera.MV3D_RGBD_GetCalibInfo(CoordinateType_Depth, ctypes.byref(depth_calib))
		check_ok(ret, "MV3D_RGBD_GetCalibInfo(CoordinateType_Depth)")

		z_param = MV3D_RGBD_PARAM()
		z_param.enParamType = ParamType_Float
		ret = camera.MV3D_RGBD_GetParam(MV3D_RGBD_FLOAT_Z_UNIT, ctypes.byref(z_param))
		z_unit_mm = float("nan")
		if ret == MV3D_RGBD_OK:
			z_unit_mm = float(z_param.ParamInfo.stFloatParam.fCurValue)

		intr = _matrix9_to_intrinsics(depth_calib.stIntrinsic.fData)
		intr["width"] = int(depth_calib.nWidth)
		intr["height"] = int(depth_calib.nHeight)
		if z_unit_mm > 0:
			intr["depth_scale"] = 1000.0 / z_unit_mm
		return intr
	finally:
		try:
			camera.MV3D_RGBD_CloseDevice()
		except Exception:
			pass
		Mv3dRgbd.MV3D_RGBD_Release()


def _read_intrinsics_from_json(json_path: Path) -> Dict[str, float]:
	if not json_path.exists():
		raise FileNotFoundError(f"Intrinsics JSON not found: {json_path}")
	data = json.loads(json_path.read_text(encoding="utf-8"))

	# Support the output format of detect_intrinsics_sdk.py directly.
	if "open3d_depth_args" in data and isinstance(data["open3d_depth_args"], dict):
		source = data["open3d_depth_args"]
	else:
		source = data

	if "intrinsic_matrix_3x3" in source:
		base = _matrix9_to_intrinsics(source["intrinsic_matrix_3x3"])
	else:
		base = {
			"fx": float(source["fx"]),
			"fy": float(source["fy"]),
			"cx": float(source["cx"]),
			"cy": float(source["cy"]),
		}

	if "width" in source:
		base["width"] = int(source["width"])
	if "height" in source:
		base["height"] = int(source["height"])
	if "depth_scale" in source:
		base["depth_scale"] = float(source["depth_scale"])
	return base


def _resolve_intrinsics(args, image_width: int, image_height: int) -> Dict[str, float]:
	intrinsics: Dict[str, float] = {}

	if args.use_sdk_intrinsics:
		intrinsics.update(_read_intrinsics_from_sdk(args.sdk_index, args.sdk_serial))

	if args.intrinsics_json:
		intrinsics.update(_read_intrinsics_from_json(Path(args.intrinsics_json)))

	for key in ("fx", "fy", "cx", "cy"):
		value = getattr(args, key)
		if value is not None:
			intrinsics[key] = float(value)

	if "fx" not in intrinsics or "fy" not in intrinsics or "cx" not in intrinsics or "cy" not in intrinsics:
		raise ValueError(
			"Missing camera intrinsics. Use --use-sdk-intrinsics or --intrinsics-json, "
			"or pass --fx --fy --cx --cy manually."
		)

	intrinsics["width"] = int(args.width) if args.width > 0 else int(intrinsics.get("width", image_width))
	intrinsics["height"] = int(args.height) if args.height > 0 else int(intrinsics.get("height", image_height))

	if args.depth_scale is not None:
		intrinsics["depth_scale"] = float(args.depth_scale)
	else:
		intrinsics["depth_scale"] = float(intrinsics.get("depth_scale", 1000.0))

	return intrinsics


def load_depth_image(depth_path: Path) -> o3d.geometry.Image:
	if not depth_path.exists():
		raise FileNotFoundError(f"Depth image not found: {depth_path}")

	def _read_open3d(path_obj: Path) -> Optional[o3d.geometry.Image]:
		image = o3d.io.read_image(str(path_obj))
		if np.asarray(image).size == 0:
			return None
		return image

	suffix = depth_path.suffix.lower()
	prefer_open3d_direct = suffix in {".png", ".jpg", ".jpeg"}

	if prefer_open3d_direct:
		try:
			image = _read_open3d(depth_path)
			if image is not None:
				return image
		except UnicodeDecodeError:
			pass

	# Open3D may fail on non-ASCII paths on Windows; retry from temp ASCII path.
	try:
		tmp_suffix = depth_path.suffix if depth_path.suffix else ".img"
		with tempfile.TemporaryDirectory(prefix="depth_img_") as tmpdir:
			tmp_path = Path(tmpdir) / f"input{tmp_suffix}"
			shutil.copy2(depth_path, tmp_path)
			if prefer_open3d_direct:
				image = _read_open3d(tmp_path)
				if image is not None:
					return image
	except Exception:
		pass

	# Fallback for formats not supported by current Open3D build (e.g. BMP).
	try:
		from PIL import Image
	except ImportError as exc:
		raise RuntimeError(
			"Open3D failed to read this depth image format and Pillow is not installed. "
			"Install pillow or convert depth image to PNG."
		) from exc

	with tempfile.TemporaryDirectory(prefix="depth_png_") as tmpdir:
		tmp_png = Path(tmpdir) / "input.png"
		pil_img = Image.open(depth_path)
		# Keep channel information for pseudo-color detection in normalize_depth_image.
		if pil_img.mode == "P":
			pil_img = pil_img.convert("RGB")
		elif pil_img.mode not in ("RGB", "RGBA", "L", "I;16", "I", "F"):
			pil_img = pil_img.convert("I;16")
		pil_img.save(tmp_png)
		image = _read_open3d(tmp_png)
		if image is None:
			raise RuntimeError(f"Failed to read converted PNG image: {tmp_png}")
		return image


def _apply_separable_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	pad = len(kernel) // 2
	# Horizontal pass
	h_pad = np.pad(image, ((0, 0), (pad, pad)), mode="edge")
	h_out = np.zeros_like(image, dtype=np.float32)
	for i, w in enumerate(kernel):
		h_out += w * h_pad[:, i : i + image.shape[1]]

	# Vertical pass
	v_pad = np.pad(h_out, ((pad, pad), (0, 0)), mode="edge")
	v_out = np.zeros_like(h_out, dtype=np.float32)
	for i, w in enumerate(kernel):
		v_out += w * v_pad[i : i + image.shape[0], :]
	return v_out


def _filter_depth_numpy(depth_image: o3d.geometry.Image, method: str, iterations: int) -> o3d.geometry.Image:
	depth_np = np.asarray(depth_image)
	if depth_np.ndim != 2:
		raise RuntimeError("Depth image must be single channel for numpy fallback filter.")

	original_dtype = depth_np.dtype
	work = depth_np.astype(np.float32)
	kernel = NUMPY_GAUSSIAN_KERNELS[method]
	for _ in range(iterations):
		work = _apply_separable_kernel(work, kernel)

	if np.issubdtype(original_dtype, np.integer):
		info = np.iinfo(original_dtype)
		work = np.clip(np.rint(work), info.min, info.max).astype(original_dtype)
	else:
		work = work.astype(original_dtype)
	return o3d.geometry.Image(work)


def normalize_depth_image(depth_image: o3d.geometry.Image) -> o3d.geometry.Image:
	depth_np = np.asarray(depth_image)
	if depth_np.ndim == 3:
		# RGB pseudo-color depth images cannot be converted back to metric depth.
		if depth_np.shape[2] == 3 and not (
			np.array_equal(depth_np[..., 0], depth_np[..., 1])
			and np.array_equal(depth_np[..., 1], depth_np[..., 2])
		):
			raise RuntimeError(
				"Input depth image is RGB pseudo-color (24-bit), not true metric depth. "
				"Please export raw depth (C16/16-bit PNG/TIFF) from SDK/HiViewer, "
				"or capture depth directly via SDK API."
			)
		depth_np = depth_np[..., 0]

	if depth_np.dtype == np.uint8:
		depth_np = (depth_np.astype(np.uint16) * 256)
	elif depth_np.dtype == np.float64:
		depth_np = depth_np.astype(np.float32)
	elif depth_np.dtype not in (np.uint16, np.float32):
		depth_np = depth_np.astype(np.uint16)

	return o3d.geometry.Image(depth_np)


def filter_depth_image(
	depth_image: o3d.geometry.Image,
	method: str = "gaussian5",
	iterations: int = 1,
) -> o3d.geometry.Image:
	if method == "none":
		return depth_image
	if method not in FILTER_MAP:
		raise ValueError(f"Unsupported filter method: {method}")
	if iterations < 1:
		raise ValueError("iterations must be >= 1")

	filtered = depth_image
	try:
		for _ in range(iterations):
			filtered = filtered.filter(FILTER_MAP[method])
		return filtered
	except RuntimeError:
		return _filter_depth_numpy(depth_image, method, iterations)


def build_intrinsic(
	width: int,
	height: int,
	fx: float,
	fy: float,
	cx: float,
	cy: float,
) -> o3d.camera.PinholeCameraIntrinsic:
	intrinsic = o3d.camera.PinholeCameraIntrinsic()
	intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
	return intrinsic


def depth_to_point_cloud(
	depth_image: o3d.geometry.Image,
	intrinsic: o3d.camera.PinholeCameraIntrinsic,
	depth_scale: float,
	depth_trunc: float,
	stride: int,
) -> o3d.geometry.PointCloud:
	if stride < 1:
		raise ValueError("stride must be >= 1")
	return o3d.geometry.PointCloud.create_from_depth_image(
		depth_image,
		intrinsic,
		depth_scale=depth_scale,
		depth_trunc=depth_trunc,
		stride=stride,
	)


def denoise_point_cloud(
	pcd: o3d.geometry.PointCloud,
	nb_neighbors: int,
	std_ratio: float,
	radius: float,
	min_points: int,
) -> o3d.geometry.PointCloud:
	if len(pcd.points) == 0:
		return pcd

	clean_pcd, _ = pcd.remove_statistical_outlier(
		nb_neighbors=nb_neighbors,
		std_ratio=std_ratio,
	)

	if len(clean_pcd.points) == 0:
		return clean_pcd

	clean_pcd, _ = clean_pcd.remove_radius_outlier(
		nb_points=min_points,
		radius=radius,
	)
	return clean_pcd


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Filter depth image, convert to point cloud, and export PLY."
	)
	parser.add_argument("--depth", required=True, help="Path to depth image (png/tiff)")
	parser.add_argument("--out-ply", required=True, help="Path to output PLY file")
	parser.add_argument("--out-depth", default="", help="Optional filtered depth image path")
	parser.add_argument(
		"--intrinsics-json",
		default="",
		help="Path to JSON containing intrinsics (supports detect_intrinsics_sdk output)",
	)
	parser.add_argument(
		"--use-sdk-intrinsics",
		action="store_true",
		help="Read depth intrinsics directly from MV3D SDK",
	)
	parser.add_argument("--sdk-index", type=int, default=0, help="MV3D device index for SDK mode")
	parser.add_argument("--sdk-serial", default="", help="MV3D serial number for SDK mode")

	parser.add_argument(
		"--filter",
		default="gaussian5",
		choices=["none", "gaussian3", "gaussian5", "gaussian7"],
		help="Depth filter method",
	)
	parser.add_argument("--iterations", type=int, default=1, help="Filter iterations")

	parser.add_argument("--fx", type=float, default=None, help="Camera intrinsic fx")
	parser.add_argument("--fy", type=float, default=None, help="Camera intrinsic fy")
	parser.add_argument("--cx", type=float, default=None, help="Camera intrinsic cx")
	parser.add_argument("--cy", type=float, default=None, help="Camera intrinsic cy")

	parser.add_argument(
		"--width",
		type=int,
		default=0,
		help="Intrinsic width (0 means read from depth image)",
	)
	parser.add_argument(
		"--height",
		type=int,
		default=0,
		help="Intrinsic height (0 means read from depth image)",
	)

	parser.add_argument("--depth-scale", type=float, default=None)
	parser.add_argument("--depth-trunc", type=float, default=3.0)
	parser.add_argument("--stride", type=int, default=1)

	parser.add_argument("--nb-neighbors", type=int, default=30)
	parser.add_argument("--std-ratio", type=float, default=1.5)
	parser.add_argument("--radius", type=float, default=0.01)
	parser.add_argument("--min-points", type=int, default=8)
	parser.add_argument(
		"--skip-denoise",
		action="store_true",
		help="Skip point cloud denoising and export raw depth-generated point cloud",
	)

	parser.add_argument("--visualize", action="store_true")
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	depth_path = Path(args.depth)
	out_ply = Path(args.out_ply)
	out_ply.parent.mkdir(parents=True, exist_ok=True)

	depth_image = load_depth_image(depth_path)
	depth_image = normalize_depth_image(depth_image)
	filtered_depth = filter_depth_image(depth_image, args.filter, args.iterations)

	if args.out_depth:
		out_depth = Path(args.out_depth)
		out_depth.parent.mkdir(parents=True, exist_ok=True)
		o3d.io.write_image(str(out_depth), filtered_depth)

	depth_np = np.asarray(filtered_depth)
	height, width = depth_np.shape[:2]
	intrinsics = _resolve_intrinsics(args, image_width=width, image_height=height)
	print(
		"Using intrinsics:",
		json.dumps(
			{
				"fx": intrinsics["fx"],
				"fy": intrinsics["fy"],
				"cx": intrinsics["cx"],
				"cy": intrinsics["cy"],
				"width": intrinsics["width"],
				"height": intrinsics["height"],
				"depth_scale": intrinsics["depth_scale"],
			},
			ensure_ascii=False,
		),
	)
	intrinsic = build_intrinsic(
		width=intrinsics["width"],
		height=intrinsics["height"],
		fx=intrinsics["fx"],
		fy=intrinsics["fy"],
		cx=intrinsics["cx"],
		cy=intrinsics["cy"],
	)

	pcd = depth_to_point_cloud(
		filtered_depth,
		intrinsic,
		depth_scale=intrinsics["depth_scale"],
		depth_trunc=args.depth_trunc,
		stride=args.stride,
	)

	if not args.skip_denoise:
		pcd_before_denoise = pcd
		pcd = denoise_point_cloud(
			pcd,
			nb_neighbors=args.nb_neighbors,
			std_ratio=args.std_ratio,
			radius=args.radius,
			min_points=args.min_points,
		)
		if len(pcd.points) == 0 and len(pcd_before_denoise.points) > 0:
			print(
				"Warning: denoise removed all points. "
				"Falling back to the raw point cloud. "
				"Consider --skip-denoise or tuning --radius/--std-ratio."
			)
			pcd = pcd_before_denoise

	if len(pcd.points) == 0:
		raise RuntimeError("No valid points generated. Please check depth image and camera parameters.")

	o3d.io.write_point_cloud(str(out_ply), pcd)
	print(f"Saved point cloud: {out_ply} ({len(pcd.points)} points)")

	if args.visualize:
		o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
	main()
