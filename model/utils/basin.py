import numpy as np
import rasterio
import upstream_basin

flow_dir = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/connectivity/channel_upper/d8.tif'
pp = [1482, 1100]

with rasterio.open(flow_dir) as fd_src:
    fd_array = fd_src.read()[0, :, :]
    fd_arr = np.asarray(fd_array, dtype=np.int32)

bas = upstream_basin.upstream_basin(pp, fd_arr)
print('done')
