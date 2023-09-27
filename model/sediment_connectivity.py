import json
import os
import time
from tqdm import tqdm
import rasterio
import numpy as np
from dsp_transport.calculate_transport import transport

from numba import jit
from numba.typed import Dict

from utils.delineate_basin import delineate_catchment


@jit(nopython=True)
def basin_raster_mean(basin, raster_array):
    numcells = basin.sum()
    print(f'basin covers {numcells} cells')
    vals = []
    for row in range(raster_array.shape[0]):
        for col in range(raster_array.shape[1]):
            if basin[row, col] == 1:
                vals.append(raster_array[row, col])
    return sum(vals) / numcells


@jit(nopython=True)
def find_d_up(row, col, weight_array, slope_array, fa_array, xres, yres, basin=None):
    if basin is None:
        d_up = weight_array[row, col] * slope_array[row, col] * np.sqrt(fa_array[row, col]* xres * yres)
    else:
        w_ave = basin_raster_mean(basin, weight_array)
        s_ave = basin_raster_mean(basin, slope_array)

        d_up = w_ave * s_ave * np.sqrt(fa_array[row, col] * xres * yres)

    return d_up


@jit(nopython=True)
def find_d_down(row, col, directions, fd_array, fd_nd, network_arr, network_nd, slope_array, weight_array):

    d_down_vals = []
    outbounds = False
    next = directions[fd_array[row, col]]
    nrow, ncol, dist = row + int(next[0][0]), col + int(next[0][1]), next[1][0]
    while network_arr[nrow, ncol] == network_nd:
        if fd_array[nrow, ncol] == fd_nd:
            outbounds = True
            break
        slopeval = max(0.005, slope_array[nrow, ncol])
        slopeval = min(1., slopeval)
        d_down_vals.append(dist / (weight_array[nrow, ncol] * slopeval))

        next = directions[fd_array[nrow, ncol]]
        nrow, ncol, dist = nrow + int(next[0][0]), ncol + int(next[0][1]), next[1][0]

    if outbounds is False and len(d_down_vals) > 0:

        print(f'cells downstream {len(d_down_vals)}')
        streamid = network_arr[nrow, ncol]
        d_down = sum(d_down_vals)

        return streamid, d_down
    else:
        return None, None


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
        fd_nd = fd_src.nodata
        xres = abs(fa_src.res[0])
        yres = abs(fa_src.res[1])
        meta = dem_src.profile

        ic_array = np.full(dem_array.shape, dem_nd)

    directions = {
        1: np.array([[0, 1], [xres, 1]]),
        2: np.array([[-1, 1], [xres * 2 ** 0.5, 1]]),
        3: np.array([[-1, 0], [yres, 1]]),
        4: np.array([[-1, -1], [xres * 2 ** 0.5, 1]]),
        5: np.array([[0, -1], [xres, 1]]),
        6: np.array([[1, -1], [xres * 2 ** 0.5, 1]]),
        7: np.array([[1, 0], [yres, 1]]),
        8: np.array([[1, 1], [xres * 2 ** 0.5, 1]])
    }
    d = Dict()
    for k, v in directions.items():
        d[k] = v

    for row in tqdm(range(dem_array.shape[0])):
        for col in range(dem_array.shape[1]):
            if network_arr[row, col] != network_nd:
                continue
            if fa_array[row, col] < 1:
                continue
            if fa_array[row, col] == 1:
                d_up = find_d_up(row, col, weight_array, slope_array, fa_array, xres, yres)
            else:
                basin = delineate_catchment(flow_dir, [row, col])
                d_up = find_d_up(row, col, weight_array, slope_array, fa_array, xres, yres, basin)

            streamid, d_down = find_d_down(row, col, d, fd_array, fd_nd, network_arr, network_nd,
                                           slope_array, weight_array)
            if streamid is not None:
                if int(streamid) != 1:
                    print('checking')

            if d_down is not None:
                conn_vals[streamid].append(np.log10(d_up/d_down))
                ic_array[row, col] = np.log10(d_up/d_down)

    hs_connectivity = {id: np.average(vals) for id, vals in conn_vals.items()}

    with rasterio.open(os.path.join(os.path.dirname(filled_dem), 'IC.tif'), 'w', **meta) as dst:
        dst.write(ic_array, 1)

    return hs_connectivity


def channel_connectivity(reaches, gsds):

    # reaches must have topological identifiers from network tools

    # open the gsd json as a dict
    with open(gsds) as gsdfile:
        gsd = json.load(gsdfile)

    


nr = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/conn_test/raster_network_id.tif'
filled = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/conn_test/pitfill.tif'
fa = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/conn_test/flow_acc.tif'
fd = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/conn_test/flow_dir.tif'
sl = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/conn_test/slope.tif'
we = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/conn_test/topo_weight.tif'

hillslope_connectivity(nr, filled, fa, fd, sl, we)
