import numpy as np
import tifffile as tiff

arr = np.load("image.npy")  # shape: (H,W) or (H,W,3)

# If your data is float in [0,1] and you want 16-bit integer TIFF:
if arr.dtype.kind == "f":
    arr16 = np.clip(arr, 0.0, 1.0)
    arr16 = (arr16 * 65535.0 + 0.5).astype(np.uint16)
    out = arr16
else:
    out = arr  # already integer, e.g. uint16

tiff.imwrite("image.tiff", out)