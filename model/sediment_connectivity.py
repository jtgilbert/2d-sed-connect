import json
import os
from math import pi
import time
from tqdm import tqdm
import rasterio
import numpy as np
import geopandas as gpd
from dsp_transport.calculate_transport import transport

from numba import jit
from numba.typed import Dict

from utils.delineate_basin import delineate_catchment


@jit(nopython=True)
def basin_raster_mean(basin, raster_array, res_x, res_y):
    numcells = basin.sum()
    # print(f'basin covers {numcells} cells')
    vals = []
    for row in range(raster_array.shape[0]):
        for col in range(raster_array.shape[1]):
            if basin[row, col] == 1:
                vals.append(raster_array[row, col])
    return sum(vals) / numcells, numcells * res_x * res_y


@jit(nopython=True)
def find_d_up(row, col, weight_array, slope_array, xres, yres, basin=None):
    if basin is None:
        slopeval = max(0.005, slope_array[row, col])
        slopeval = min(1., slopeval)
        d_up = weight_array[row, col] * slopeval * np.sqrt(xres * yres)
    else:
        w_ave, area = basin_raster_mean(basin, weight_array, xres, yres)
        s_ave, area = basin_raster_mean(basin, slope_array, xres, yres)

        d_up = w_ave * s_ave * np.sqrt(area)

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

    if outbounds is False and sum(d_down_vals) > 0:

        # print(f'cells downstream {len(d_down_vals)}')
        streamid = network_arr[nrow, ncol]
        d_down = sum(d_down_vals)

        return streamid, d_down
    else:
        return None, None


@jit(nopython=True)
def d8_flowdir(dinf_flow_dir, ndval):

    # create d8 flow dir for algorithm
    d8_array = np.zeros(dinf_flow_dir.shape, dtype=np.int64)
    for row in range(dinf_flow_dir.shape[0]):
        for col in range(dinf_flow_dir.shape[1]):
            if dinf_flow_dir[row, col] == ndval:
                continue
            else:
                if 15*pi/8 <= dinf_flow_dir[row, col] < pi/8:
                    d8_array[row, col] = 1
                elif pi/8 <= dinf_flow_dir[row, col] < 3*pi/8:
                    d8_array[row, col] = 2
                elif 3*pi/8 <= dinf_flow_dir[row, col] < 5*pi/8:
                    d8_array[row, col] = 3
                elif 5*pi/8 <= dinf_flow_dir[row, col] < 7*pi/8:
                    d8_array[row, col] = 4
                elif 7*pi/8 <= dinf_flow_dir[row, col] < 9*pi/8:
                    d8_array[row, col] = 5
                elif 9*pi/8 <= dinf_flow_dir[row, col] < 11*pi/8:
                    d8_array[row, col] = 6
                elif 11*pi/8 <= dinf_flow_dir[row, col] < 13*pi/8:
                    d8_array[row, col] = 7
                else:
                    d8_array[row, col] = 8

    return d8_array


def upstream_basin(pour_point, fd_array):
    def find_new_cells(point, fd_array):
        new_cells = []
        row, col = point[0], point[1]
        if fd_array[row, col + 1] == 5:
            new_cells.append([row, col + 1])
        if fd_array[row - 1, col + 1] == 6:
            new_cells.append([row - 1, col + 1])
        if fd_array[row - 1, col] == 7:
            new_cells.append([row - 1, col])
        if fd_array[row - 1, col - 1] == 8:
            new_cells.append([row - 1, col - 1])
        if fd_array[row, col - 1] == 1:
            new_cells.append([row, col - 1])
        if fd_array[row + 1, col - 1] == 2:
            new_cells.append([row + 1, col - 1])
        if fd_array[row + 1, col] == 3:
            new_cells.append([row + 1, col])
        if fd_array[row + 1, col + 1] == 4:
            new_cells.append([row + 1, col + 1])

        return new_cells

    out_array = np.zeros(fd_array.shape, dtype=np.int32)

    row, col = pour_point[0], pour_point[1]
    # set the pour point as part of the basin
    out_array[row, col] = 1

    new_cells = find_new_cells(pour_point, fd_array)
    while len(new_cells) > 0:
        for cell in new_cells:
            if out_array[cell[0], cell[1]] != 1:
                out_array[cell[0], cell[1]] = 1
                new_cells2 = find_new_cells(cell, fd_array)
                if len(new_cells2) >= 2:
                    print('checking')
                new_cells.remove(cell)
                for nc in new_cells2:
                    new_cells.append(nc)

    return out_array


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

    fd_array = d8_flowdir(fd_array, fd_nd)
    with rasterio.open(os.path.join(os.path.dirname(filled_dem), 'd8.tif'), 'w', **meta) as dst:
        dst.write(fd_array, 1)

    basintest = upstream_basin([2289, 926], fd_array)

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
                d_up = find_d_up(row, col, weight_array, slope_array, xres, yres)
            else:
                basin = upstream_basin([row, col], fd_array)
                # basin = delineate_catchment(flow_dir, [row, col])
                d_up = find_d_up(row, col, weight_array, slope_array, xres, yres, basin)

            streamid, d_down = find_d_down(row, col, d, fd_array, 0, network_arr, network_nd,
                                           slope_array, weight_array)

            if d_down not in [None, 0, 0.0] and d_up not in [None, 0, 0.0]:
                conn_vals[streamid].append(np.log10(d_up/d_down))
                ic_array[row, col] = np.log10(d_up/d_down)  # getting -inf values that I need to fix...

    hs_connectivity = {id: np.average(vals) for id, vals in conn_vals.items()}

    with rasterio.open(os.path.join(os.path.dirname(filled_dem), 'IC.tif'), 'w', **meta) as dst:
        dst.write(ic_array, 1)

    return hs_connectivity


def channel_connectivity(reaches, gsds, id_field, upstream_id, discharge, flow_scale_field, depth_hydro_geom, width_hydr_geom):

    # reaches must have topological identifiers from network tools that match keys in the gsd dict
    # reaches must also be prepared with slope and upstream-downstream topology fields, width?
    sed_yield = {}

    # open the gsd json as a dict
    with open(gsds) as gsdfile:
        gsd = json.load(gsdfile)

    network = gpd.read_file(reaches)

    for i in network.index:
        reach_id = network.loc[i, id_field]
        q = discharge * network.loc[i, flow_scale_field]
        depth = depth_hydro_geom[0] * q ** depth_hydro_geom[1]
        width = width_hydr_geom[0] * q ** width_hydr_geom[1]
        fractions = {key: val['fraction'] for key, val in gsd[reach_id]['fractions'].items()}
        qs = transport(fractions, network.loc[i, 'Slope'], discharge*network.loc[i, flow_scale_field], depth, width, 900)
        qs_kg = {key: val[1] for key, val in qs.items()}

        sed_yield[reach_id] = sum(qs_kg.values())

    sdr = {}
    for i in network.index:
        qs_in = sed_yield[network.loc[i, upstream_id]]
        qs_out = sed_yield[network.loc[i, id_field]]
        sdr[network.loc[i, id_field]] = qs_in / qs_out

    return sdr


nr = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/connectivity/channel_upper/network.tif'
filled = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/connectivity/channel_upper/pitfill.tif'
fa = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/connectivity/channel_upper/flow_acc.tif'
fd = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/connectivity/channel_upper/flow_dir.tif'
sl = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/connectivity/channel_upper/slope.tif'
we = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/connectivity/channel_upper/topo_weight.tif'

hc = hillslope_connectivity(nr, filled, fa, fd, sl, we)
print(hc)
