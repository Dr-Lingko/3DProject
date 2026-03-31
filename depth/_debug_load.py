from pathlib import Path
import traceback
import numpy as np

import fliter

p = Path(r"E:\点云测量\02\MV-DT01SNU(DA5635322)\depth\Image__Depth_20260323_22_51_28_441.bmp")
print("exists", p.exists())

try:
    img = fliter.load_depth_image(p)
    arr = np.asarray(img)
    print("shape", arr.shape, "dtype", arr.dtype, "size", arr.size)
except Exception:
    traceback.print_exc()
