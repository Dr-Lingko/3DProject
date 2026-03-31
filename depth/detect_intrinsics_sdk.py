import argparse
import ctypes
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd
from Mv3dRgbdImport.Mv3dRgbdDefine import (
    CoordinateType_Depth,
    CoordinateType_RGB,
    DeviceType_Ethernet,
    DeviceType_Ethernet_Vir,
    DeviceType_USB,
    DeviceType_USB_Vir,
    MV3D_RGBD_CALIB_INFO,
    MV3D_RGBD_CAMERA_PARAM,
    MV3D_RGBD_DEVICE_INFO_LIST,
    MV3D_RGBD_FLOAT_Z_UNIT,
    MV3D_RGBD_OK,
    MV3D_RGBD_PARAM,
    ParamType_Float,
)


def _decode_char_array(values) -> str:
    raw = bytes(int(v) & 0xFF for v in values)
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()


def _matrix9_to_intrinsics(matrix9: List[float]) -> Dict[str, float]:
    # Matrix layout: [fx,0,cx, 0,fy,cy, 0,0,1]
    return {
        "fx": float(matrix9[0]),
        "fy": float(matrix9[4]),
        "cx": float(matrix9[2]),
        "cy": float(matrix9[5]),
    }


def _calib_to_dict(calib: MV3D_RGBD_CALIB_INFO) -> Dict[str, object]:
    intrinsic = [float(v) for v in calib.stIntrinsic.fData]
    distortion = [float(v) for v in calib.stDistortion.fData]
    data = {
        "width": int(calib.nWidth),
        "height": int(calib.nHeight),
        "intrinsic_matrix_3x3": intrinsic,
        "distortion_12": distortion,
    }
    data.update(_matrix9_to_intrinsics(intrinsic))
    return data


def _check_ok(ret: int, action: str) -> None:
    if ret != MV3D_RGBD_OK:
        raise RuntimeError(f"{action} failed: ret=0x{ret:08X}")


def _list_devices() -> MV3D_RGBD_DEVICE_INFO_LIST:
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
    return device_list


def _pick_device_index(device_list: MV3D_RGBD_DEVICE_INFO_LIST, serial: str, index: int) -> int:
    if serial:
        for i in range(20):
            sn = _decode_char_array(device_list.DeviceInfo[i].chSerialNumber)
            if sn == serial:
                return i
        raise RuntimeError(f"Serial not found: {serial}")

    if index < 0 or index >= 20:
        raise RuntimeError(f"Invalid index: {index}")
    return index


def _read_z_unit(camera: Mv3dRgbd) -> float:
    param = MV3D_RGBD_PARAM()
    param.enParamType = ParamType_Float
    ret = camera.MV3D_RGBD_GetParam(MV3D_RGBD_FLOAT_Z_UNIT, ctypes.byref(param))
    if ret != MV3D_RGBD_OK:
        return float("nan")
    return float(param.ParamInfo.stFloatParam.fCurValue)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read MV3D camera intrinsics from SDK.")
    parser.add_argument("--index", type=int, default=0, help="Device index (default: 0)")
    parser.add_argument("--serial", type=str, default="", help="Device serial number (preferred)")
    args = parser.parse_args()

    ret = Mv3dRgbd.MV3D_RGBD_Initialize()
    if ret != MV3D_RGBD_OK:
        print(f"SDK init failed: 0x{ret:08X}")
        return 1

    camera = Mv3dRgbd()
    try:
        device_list = _list_devices()

        printable_devices = []
        for i in range(20):
            model = _decode_char_array(device_list.DeviceInfo[i].chModelName)
            serial = _decode_char_array(device_list.DeviceInfo[i].chSerialNumber)
            if model or serial:
                printable_devices.append({"index": i, "model": model, "serial": serial})

        print(json.dumps({"devices": printable_devices}, ensure_ascii=False, indent=2))

        selected = _pick_device_index(device_list, args.serial, args.index)
        ret = camera.MV3D_RGBD_OpenDevice(ctypes.pointer(device_list.DeviceInfo[selected]))
        _check_ok(ret, "MV3D_RGBD_OpenDevice")

        depth_calib = MV3D_RGBD_CALIB_INFO()
        rgb_calib = MV3D_RGBD_CALIB_INFO()
        camera_param = MV3D_RGBD_CAMERA_PARAM()

        _check_ok(
            camera.MV3D_RGBD_GetCalibInfo(CoordinateType_Depth, ctypes.byref(depth_calib)),
            "MV3D_RGBD_GetCalibInfo(CoordinateType_Depth)",
        )
        _check_ok(
            camera.MV3D_RGBD_GetCalibInfo(CoordinateType_RGB, ctypes.byref(rgb_calib)),
            "MV3D_RGBD_GetCalibInfo(CoordinateType_RGB)",
        )
        _check_ok(
            camera.MV3D_RGBD_GetCameraParam(ctypes.byref(camera_param)),
            "MV3D_RGBD_GetCameraParam",
        )

        result = {
            "selected_device_index": selected,
            "depth_calib": _calib_to_dict(depth_calib),
            "rgb_calib": _calib_to_dict(rgb_calib),
            "camera_param_depth": _calib_to_dict(camera_param.stDepthCalibInfo),
            "camera_param_rgb": _calib_to_dict(camera_param.stRgbCalibInfo),
            "depth_to_rgb_extrinsic_4x4": [float(v) for v in camera_param.stDepth2RgbExtrinsic.fData],
            "z_unit_mm": _read_z_unit(camera),
            "open3d_depth_args": {
                "fx": float(depth_calib.stIntrinsic.fData[0]),
                "fy": float(depth_calib.stIntrinsic.fData[4]),
                "cx": float(depth_calib.stIntrinsic.fData[2]),
                "cy": float(depth_calib.stIntrinsic.fData[5]),
                "width": int(depth_calib.nWidth),
                "height": int(depth_calib.nHeight),
                "depth_scale": 1000.0 / _read_z_unit(camera) if _read_z_unit(camera) > 0 else 1000.0,
            },
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as exc:
        print(str(exc))
        return 1
    finally:
        try:
            camera.MV3D_RGBD_CloseDevice()
        except Exception:
            pass
        Mv3dRgbd.MV3D_RGBD_Release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
