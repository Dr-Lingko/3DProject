import argparse
import ctypes
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd
from Mv3dRgbdImport.Mv3dRgbdDefine import (
    DeviceType_Ethernet,
    DeviceType_Ethernet_Vir,
    DeviceType_USB,
    DeviceType_USB_Vir,
    ImageType_Depth,
    MV3D_RGBD_DEVICE_INFO_LIST,
    MV3D_RGBD_FLOAT_Z_UNIT,
    MV3D_RGBD_FRAME_DATA,
    MV3D_RGBD_OK,
    MV3D_RGBD_PARAM,
    ParamType_Float,
)


def _decode_char_array(values) -> str:
    raw = bytes(int(v) & 0xFF for v in values)
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()


def _check_ok(ret: int, action: str) -> None:
    if ret != MV3D_RGBD_OK:
        raise RuntimeError(f"{action} failed: ret=0x{ret:08X}")


def _list_devices() -> tuple[MV3D_RGBD_DEVICE_INFO_LIST, list[dict]]:
    device_num = ctypes.c_uint(0)
    ret = Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(
        DeviceType_Ethernet | DeviceType_USB | DeviceType_Ethernet_Vir | DeviceType_USB_Vir,
        ctypes.byref(device_num),
    )
    _check_ok(ret, "MV3D_RGBD_GetDeviceNumber")
    if device_num.value == 0:
        raise RuntimeError("No MV3D RGBD device found.")

    device_list = MV3D_RGBD_DEVICE_INFO_LIST()
    ret = Mv3dRgbd.MV3D_RGBD_GetDeviceList(
        DeviceType_Ethernet | DeviceType_USB | DeviceType_Ethernet_Vir | DeviceType_USB_Vir,
        ctypes.pointer(device_list.DeviceInfo[0]),
        20,
        ctypes.byref(device_num),
    )
    _check_ok(ret, "MV3D_RGBD_GetDeviceList")

    devices = []
    for i in range(20):
        model = _decode_char_array(device_list.DeviceInfo[i].chModelName)
        serial = _decode_char_array(device_list.DeviceInfo[i].chSerialNumber)
        if model or serial:
            devices.append({"index": i, "model": model, "serial": serial})
    return device_list, devices


def _resolve_device_index(devices: list[dict], index: int, serial: str) -> int:
    if serial:
        for d in devices:
            if d["serial"] == serial:
                return int(d["index"])
        raise RuntimeError(f"Serial not found: {serial}")

    for d in devices:
        if int(d["index"]) == index:
            return index
    raise RuntimeError(f"Device index not available: {index}")


def _read_z_unit_mm(camera: Mv3dRgbd) -> float:
    param = MV3D_RGBD_PARAM()
    param.enParamType = ParamType_Float
    ret = camera.MV3D_RGBD_GetParam(MV3D_RGBD_FLOAT_Z_UNIT, ctypes.byref(param))
    if ret != MV3D_RGBD_OK:
        return float("nan")
    return float(param.ParamInfo.stFloatParam.fCurValue)


def _extract_depth_u16(image_data) -> np.ndarray | None:
    width = int(image_data.nWidth)
    height = int(image_data.nHeight)
    expected_pixels = width * height
    raw = ctypes.string_at(image_data.pData, int(image_data.nDataLen))

    if len(raw) >= expected_pixels * 2:
        depth = np.frombuffer(raw, dtype=np.uint16, count=expected_pixels).reshape(height, width).copy()
        return depth

    if len(raw) >= expected_pixels:
        depth8 = np.frombuffer(raw, dtype=np.uint8, count=expected_pixels).reshape(height, width)
        return (depth8.astype(np.uint16) << 8)

    return None


def _render_preview(depth_u16: np.ndarray, fixed_max_mm: int) -> np.ndarray:
    valid = depth_u16 > 0
    if not np.any(valid):
        gray = np.zeros(depth_u16.shape, dtype=np.uint8)
    else:
        max_val = fixed_max_mm if fixed_max_mm > 0 else int(np.percentile(depth_u16[valid], 99.0))
        max_val = max(max_val, 1)
        scaled = np.clip(depth_u16.astype(np.float32) / float(max_val), 0.0, 1.0)
        gray = (scaled * 255.0).astype(np.uint8)

    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    color[~valid] = (0, 0, 0)
    return color


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live preview and capture real 16-bit depth images from MV3D SDK.")
    parser.add_argument("--index", type=int, default=0, help="Device index (default: 0)")
    parser.add_argument("--serial", default="", help="Device serial number (preferred)")
    parser.add_argument("--out-dir", default="captured_depth", help="Output directory for captured depth PNG")
    parser.add_argument("--timeout-ms", type=int, default=1000, help="Fetch timeout in milliseconds")
    parser.add_argument(
        "--preview-max-mm",
        type=int,
        default=2000,
        help="Max depth for preview color scaling (0 means auto percentile)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ret = Mv3dRgbd.MV3D_RGBD_Initialize()
    _check_ok(ret, "MV3D_RGBD_Initialize")

    camera = Mv3dRgbd()
    window_name = "MV3D Depth Preview | S: save PNG16 | Q/ESC: quit"

    try:
        device_list, devices = _list_devices()
        print("Detected devices:")
        for d in devices:
            print(f"  index={d['index']} model={d['model']} serial={d['serial']}")

        selected = _resolve_device_index(devices, args.index, args.serial)
        ret = camera.MV3D_RGBD_OpenDevice(ctypes.pointer(device_list.DeviceInfo[selected]))
        _check_ok(ret, "MV3D_RGBD_OpenDevice")

        ret = camera.MV3D_RGBD_Start()
        _check_ok(ret, "MV3D_RGBD_Start")

        z_unit_mm = _read_z_unit_mm(camera)
        print(f"Connected device index={selected}, z_unit_mm={z_unit_mm}")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        last_depth = None
        last_frame_num = -1

        while True:
            frame = MV3D_RGBD_FRAME_DATA()
            ret = camera.MV3D_RGBD_FetchFrame(ctypes.pointer(frame), args.timeout_ms)
            if ret == MV3D_RGBD_OK:
                for i in range(int(frame.nImageCount)):
                    img = frame.stImageData[i]
                    if int(img.enImageType) != int(ImageType_Depth):
                        continue
                    depth = _extract_depth_u16(img)
                    if depth is None:
                        continue
                    last_depth = depth
                    last_frame_num = int(img.nFrameNum)
                    break

            if last_depth is not None:
                preview = _render_preview(last_depth, args.preview_max_mm)
                valid_mask = last_depth > 0
                valid_count = int(valid_mask.sum())
                min_mm = int(last_depth[valid_mask].min()) if valid_count > 0 else 0
                max_mm = int(last_depth[valid_mask].max()) if valid_count > 0 else 0
                text = f"frame={last_frame_num} valid={valid_count} min={min_mm} max={max_mm} mm"
                cv2.putText(preview, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(window_name, preview)
            else:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for depth frame...", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(window_name, blank)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and last_depth is not None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                png_path = out_dir / f"depth_{ts}.png"
                npy_path = out_dir / f"depth_{ts}.npy"
                ok = cv2.imwrite(str(png_path), last_depth)
                np.save(npy_path, last_depth)
                if ok:
                    print(f"Saved depth PNG16: {png_path}")
                    print(f"Saved depth NPY:   {npy_path}")
                else:
                    print(f"Failed to save PNG: {png_path}")

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(str(exc))
        return 1
    finally:
        try:
            camera.MV3D_RGBD_Stop()
        except Exception:
            pass
        try:
            camera.MV3D_RGBD_CloseDevice()
        except Exception:
            pass
        Mv3dRgbd.MV3D_RGBD_Release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
