import os
import time
import rasterio
import numpy as np
from rasterstats import zonal_stats

from utils.delineate_basin import delinate_catchment


def hillslope_connectivity(network_raster, filled_dem, flow_acc, flow_dir, slope, weight):

    # all datasets need to be orthogonal

    # get array of rasterized network and all unique ID values from it
    with rasterio.open(network_raster) as src:
        network_arr = src.read()[0, :, :]
        network_nd = src.nodata
        id_vals = np.unique(network_arr)

    # set up dictionary to store connectivity values associated with each reach
    conn_vals = {val: [] for val in id_vals}

    with rasterio.open(filled_dem) as dem_src, \
        rasterio.open(flow_acc) as fa_src, \
        rasterio.open(flow_dir) as fd_src, \
        rasterio.open(weight) as weight_src, \
        rasterio.open(slope) as slope_src:

        dem_array = dem_src.read()[0, :, :]
        fa_array = fa_src.read()[0, :, :]
        fd_array = fd_src.read()[0, :, :]
        weight_array = weight_src.read()[0, :, :]
        slope_array = slope_src.read()[0, :, :]

        dem_nd = dem_src.nodata
        xres = abs(fa_src.res[0])
        yres = abs(fa_src.res[1])

    # dict for tracing flow directions and distances
    directions = {
        1: [[0, 1], xres],
        2: [[-1, 1], xres*2**0.5],
        3: [[-1, 0], yres],
        4: [[-1, -1], xres*2**0.5],
        5: [[0, -1], xres],
        6: [[1, -1], xres*2**0.5],
        7: [[1, 0], yres],
        8: [[1, 1], xres*2**0.5]
    }

    for row in range(dem_array.shape[0]):
        for col in range(dem_array.shape[1]):
            if dem_array[row, col] == dem_nd:
                continue
            if network_arr[row, col] != network_nd:
                continue
            # delineate basin upstream of cell
            basin = delinate_catchment(flow_dir, [row, col])
            # find average weight and slope within the basin
            w_ave_zs = zonal_stats(basin, weight, stats='mean')
            w_ave = w_ave_zs[0].get('mean')
            s_ave_zs = zonal_stats(basin, slope, stats='mean')
            s_ave = s_ave_zs[0].get('mean')

            d_up = w_ave * s_ave * np.sqrt(fa_array[row, col]*xres*yres)

            d_down_vals = []
            next_vals = directions[fd_array[row, col]]
            nrow, ncol = row + next_vals[0][0], col + next_vals[0][1]

            while network_arr[nrow, ncol] == network_nd:
                d_down_vals.append(next_vals[1] / (weight_array[nrow, ncol] * slope_array[nrow, ncol]))

                next_vals = directions[fd_array[nrow, ncol]]

            streamid = network_arr[nrow, ncol]
            d_down = sum(d_down_vals)

            conn_vals[streamid].append(np.log10(d_up/d_down))

    hs_connectivity = {id: np.average(vals) for id, vals in conn_vals.items()}

    return  hs_connectivity







