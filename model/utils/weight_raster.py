import os
import time
import numpy as np
import rasterio
from scipy.signal import convolve2d
from scipy.ndimage import generic_filter


def weight_raster(dem):

    with rasterio.open(dem, 'r') as src:
        arr = src.read()[0, :, :]
        meta = src.profile

    kernel = np.full((5, 5), 1/25)
    smoothed = convolve2d(arr, kernel, 'valid')
    resid_topog = smoothed - arr[2: -2, 2: -2]
    smoothed = None
    start = time.time()
    ri = generic_filter(resid_topog, np.std, [5, 5], mode='nearest')
    print(f'took {time.time() - start} to find RI')
    ri_max = np.max(ri)
    resid_topog = None
    weight = 1 - (ri / ri_max)

    with rasterio.open(os.path.join(os.path.dirname(dem), 'topo_weight.tif'), 'w', **meta) as dst:
        dst.write(weight, 1)

    return weight