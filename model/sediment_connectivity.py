import json
import os
from math import pi
import time
from tqdm import tqdm
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from dsp_transport.calculate_transport import transport

from numba import jit
from numba.typed import Dict

from utils.delineate_basin import delineate_catchment
from utils.upstream_basin import upstream_basin


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


# def upstream_basin(pour_point, fd_array):
#     def find_new_cells(point, fd_array):
#         new_cells = []
#         row, col = point[0], point[1]
#         if fd_array[row, col + 1] == 5:
#             new_cells.append([row, col + 1])
#         if fd_array[row - 1, col + 1] == 6:
#             new_cells.append([row - 1, col + 1])
#         if fd_array[row - 1, col] == 7:
#             new_cells.append([row - 1, col])
#         if fd_array[row - 1, col - 1] == 8:
#             new_cells.append([row - 1, col - 1])
#         if fd_array[row, col - 1] == 1:
#             new_cells.append([row, col - 1])
#         if fd_array[row + 1, col - 1] == 2:
#             new_cells.append([row + 1, col - 1])
#         if fd_array[row + 1, col] == 3:
#             new_cells.append([row + 1, col])
#         if fd_array[row + 1, col + 1] == 4:
#             new_cells.append([row + 1, col + 1])
#
#         return new_cells
#
#     out_array = np.zeros(fd_array.shape, dtype=np.int32)
#
#     row, col = pour_point[0], pour_point[1]
#     # set the pour point as part of the basin
#     out_array[row, col] = 1
#
#     new_cells = find_new_cells(pour_point, fd_array)
#     while len(new_cells) > 0:
#         for cell in new_cells:
#             if out_array[cell[0], cell[1]] != 1:
#                 out_array[cell[0], cell[1]] = 1
#                 new_cells2 = find_new_cells(cell, fd_array)
#                 new_cells.remove(cell)
#                 for nc in new_cells2:
#                     new_cells.append(nc)
#
#     return out_array


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

    # basintest = upstream_basin([2289, 926], fd_array)

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
                # print(f'finding basin for cell {row}, {col}')
                st = time.time()
                basin = upstream_basin([row, col], np.asarray(fd_array, dtype=np.int64))
                # print(f'basin time: {time.time() - st}')
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


def channel_connectivity(reaches, gsds, id_field, reachids, upstream_id, hydrograph, q_interval, flow_scale_field,
                         depth_hydro_geom, width_hydro_geom, meas_slope, min_fr = None, min_fr_us = None):

    # reaches must have topological identifiers from network tools that match keys in the gsd dict
    # reaches must also be prepared with slope and upstream-downstream topology fields, width?
    sdr = {}

    # open the gsd json as a dict
    with open(gsds) as gsdfile:
        gsd = json.load(gsdfile)

    network = gpd.read_file(reaches)

    hydr = pd.read_csv(hydrograph)
    discharges = hydr['Q']

    for i in network.index:
        reach_id = network.loc[i, id_field]
        slope = network.loc[i, 'Slope']
        if reach_id not in reachids:
            continue
        d50 = gsd[str(i)]['d50']/1000
        d84 = gsd[str(i)]['d84']/1000
        sedyield = 0
        sedyield_us = 0
        for discharge in tqdm(discharges):
            us_reach_id = network.loc[i, upstream_id[0]]
            us_reach_id2 = network.loc[i, upstream_id[1]]
            q = discharge * network.loc[i, flow_scale_field]
            depth = (depth_hydro_geom[0] * q ** depth_hydro_geom[1]) * (meas_slope / slope)**0.75
            width = width_hydro_geom[0] * q ** width_hydro_geom[1]
            fractions_tmp = {float(key): val['fraction'] for key, val in gsd[str(i)]['fractions'].items()}
            if min_fr is not None:
                fractions = update_fracs(fractions_tmp, min_fr)
            else:
                fractions = fractions_tmp
            qs = transport(fractions, network.loc[i, 'Slope'], q, depth, width, q_interval, d50_in=d50, d84_in=d84)
            qs_kg = {key: val[1] for key, val in qs.items()}
            print(q, sum(qs_kg.values()))

            sedyield += sum(qs_kg.values())

            if us_reach_id is not None:
                for j in network.index:
                    if network.loc[j, id_field] == us_reach_id:
                        d50_us = gsd[str(j)]['d50']/1000
                        d84_us = gsd[str(j)]['d84']/1000
                        slope_us = network.loc[j, 'Slope']
                        q_us = discharge * network.loc[j, flow_scale_field]
                        depth_us = (depth_hydro_geom[0] * q ** depth_hydro_geom[1]) * (meas_slope / slope_us) ** 0.75
                        width_us = width_hydro_geom[0] * q ** width_hydro_geom[1]
                        fractions_us_tmp = {float(key): val['fraction'] for key, val in gsd[str(j)]['fractions'].items()}
                        if min_fr_us is not None:
                            fractions_us = update_fracs(fractions_us_tmp, min_fr_us)
                        else:
                            fractions_us = fractions_us_tmp
                        qs_us = transport(fractions_us, network.loc[j, 'Slope'], q_us, depth_us, width_us, q_interval, d50_in=d50_us, d84_in=d84_us)
                        qs_kg_us = {key: val[1] for key, val in qs_us.items()}

                        sedyield_us += sum(qs_kg_us.values())

            if us_reach_id2 is not None:
                for k in network.index:
                    if network.loc[k, id_field] == us_reach_id2:
                        d50_us2 = gsd[str(k)]['d50']/1000
                        d84_us2 = gsd[str(k)]['d84']/1000
                        slope_us2 = network.loc[k, 'Slope']
                        q_us2 = discharge * network.loc[k, flow_scale_field]
                        depth_us2 = (depth_hydro_geom[0] * q ** depth_hydro_geom[1]) * (meas_slope / slope_us2) ** 0.75
                        width_us2 = width_hydro_geom[0] * q ** width_hydro_geom[1]
                        fractions_us2_tmp = {float(key): val['fraction'] for key, val in gsd[str(k)]['fractions'].items()}
                        if min_fr_us is not None:
                            fractions_us2 = update_fracs(fractions_us2_tmp, min_fr_us)
                        else:
                            fractions_us2 = fractions_us2_tmp
                        qs_us2 = transport(fractions_us2, network.loc[k, 'Slope'], q_us2, depth_us2, width_us2, q_interval, d50_in=d50_us2, d84_in=d84_us2)
                        qs_kg_us2 = {key: val[1] for key, val in qs_us2.items()}

                        sedyield_us += sum(qs_kg_us2.values())

        if sedyield_us > 0:
            print(reach_id, sedyield, sedyield_us)
            sdr[reach_id] = sedyield / sedyield_us
        else:
            sdr[reach_id] = None

    return sdr


def network_sdr(reaches, gsds, id_field, upstream_id, hydrograph, q_interval, flow_scale_field, depth_hydro_geom, width_hydro_geom, min_fr=None, update_gsd=False):

    # reaches must have topological identifiers from network tools that match keys in the gsd dict
    # reaches must also be prepared with slope and upstream-downstream topology fields, width?
    sedyields = {}

    # open the gsd json as a dict
    with open(gsds) as gsdfile:
        gsd = json.load(gsdfile)

    network = gpd.read_file(reaches)

    hydr = pd.read_csv(hydrograph)
    discharges = hydr['Q']

    for i in tqdm(network.index):

        d50 = max(network.loc[i, 'D50'] / 1000, 0.001)
        d84 = max(network.loc[i, 'D84'] / 1000, 0.004)

        reach_id = network.loc[i, id_field]

        sed_yield = 0
        for discharge in discharges:
            q = discharge * network.loc[i, flow_scale_field]
            depth = depth_hydro_geom[0] * q ** depth_hydro_geom[1]
            width = width_hydro_geom[0] * q ** width_hydro_geom[1]
            fractions_tmp = {float(key): val['fraction'] for key, val in gsd[str(i)]['fractions'].items()}
            if min_fr is not None:
                fractions = update_fracs(fractions_tmp, min_fr)
            else:
                fractions = fractions_tmp
            qs = transport(fractions, max(network.loc[i, 'Slope'], 0.0001), q, depth, width, q_interval, d50_in=d50, d84_in=d84)
            qs_kg = {key: val[1] for key, val in qs.items()}
            # print(q, sum(qs_kg.values()))

            sed_yield += sum(qs_kg.values())

        sedyields[reach_id] = sed_yield

    for i in network.index:
        reach_id = network.loc[i, id_field]
        us_reach_id = network.loc[i, upstream_id[0]]
        us_reach_id2 = network.loc[i, upstream_id[1]]
        reach_yield = sedyields[reach_id]

        us_yields = 0
        if str(us_reach_id) != 'nan':
            us_yields += sedyields[us_reach_id]
        if str(us_reach_id2) != 'nan':
            us_yields += sedyields[us_reach_id2]

        if us_yields > 0:
            sdr = reach_yield / us_yields
        else:
            if reach_yield == 0:
                sdr = 0
            else:
                sdr = 1000000
        if sdr > 0:
            network.loc[i, 'SDR'] = np.log10(sdr)
        else:
            network.loc[i, 'SDR'] = np.log10(1e-6)

    network.to_file(reaches)

    return


def update_fracs(fracs, min_frac):
    to_remove = 0
    remove_from = []

    for phi, frac in fracs.items():
        if phi >= -6 and frac < min_frac:
            to_remove += min_frac - frac
            fracs[phi] = min_frac

    for phi, frac in fracs.items():
        if frac > to_remove:
            remove_from.append(phi)

    for phi, frac in fracs.items():
        if phi in remove_from:
            fracs[phi] = frac - (to_remove / len(remove_from))

    return fracs


def recking_channel_connectivity(reaches, id_field, reachids, upstream_id, hydrograph, q_interval, flow_scale_field, width_hydro_geom):
    sdr = {}

    network = gpd.read_file(reaches)

    hydr = pd.read_csv(hydrograph)
    discharges = hydr['Q']

    for i in network.index:
        reach_id = network.loc[i, id_field]
        if reach_id not in reachids:
            continue
        sedyield = 0
        sedyield_us = 0
        us_reach_id = network.loc[i, upstream_id[0]]
        us_reach_id2 = network.loc[i, upstream_id[1]]
        s = network.loc[i, 'Slope']
        d50 = network.loc[i, 'D50'] / 1000
        d84 = network.loc[i, 'D84'] / 1000
        tau_star_m = (5*s + 0.06) * (d84/d50)**(4.4*s**0.5-1.5)
        for discharge in tqdm(discharges):
            flow = discharge * network.loc[i, flow_scale_field]
            if flow > 5:
                print('checking')
            w = 9
            q = flow / w
            if q / np.sqrt(9.81*s*d84**3) < 100:
                p = 0.23
            else:
                p = 0.3
            tau_star_84 = s / (1.65 * d84 * ((2 / w) + (74 * p**2.6) * (9.81 * s)**p * q**(-2 * p) * d84**(3 * p - 1)))
            tp = 14 * tau_star_84**2.5 / (1 + (tau_star_m / tau_star_84)**4)
            qsv = np.sqrt(9.81*1.65*d84**3) * tp
            Qs = w*2650*qsv
            print(flow, Qs * q_interval)
            sedyield += Qs * q_interval

            if us_reach_id is not None:
                for j in network.index:
                    if network.loc[j, id_field] == us_reach_id:
                        s = network.loc[j, 'Slope']
                        d50 = network.loc[j, 'D50']
                        d84 = network.loc[j, 'D84']
                        tau_star_m = (5 * s + 0.06) * (d84 / d50) ** (4.4 * s ** 0.5 - 1.5)
                        flow = discharge * network.loc[j, flow_scale_field]
                        w = width_hydro_geom[0] * flow ** width_hydro_geom[1]
                        q = flow / w
                        if q / np.sqrt(9.81 * s * d84 ** 3) < 100:
                            p = 0.23
                        else:
                            p = 0.3
                        tau_star_84 = s / (1.65 * d84 * (
                                    2 / w + 74 * p ** 2.6 * (9.81 * s) ** p * q ** (-2 * p) * d84 ** (3 * p - 1)))
                        tp = 14 * tau_star_84 ** 2.5 / (1 + (tau_star_m / tau_star_84) ** 4)
                        qsv = np.sqrt(9.81 * 1.65 * d84 ** 3) * tp
                        Qs = w * 2650 * qsv
                        sedyield_us += Qs * q_interval

            if us_reach_id2 is not None:
                for k in network.index:
                    if network.loc[k, id_field] == us_reach_id:
                        s = network.loc[k, 'Slope']
                        d50 = network.loc[k, 'D50']
                        d84 = network.loc[k, 'D84']
                        tau_star_m = (5 * s + 0.06) * (d84 / d50) ** (4.4 * s ** 0.5 - 1.5)
                        flow = discharge * network.loc[k, flow_scale_field]
                        w = width_hydro_geom[0] * flow ** width_hydro_geom[1]
                        q = flow / w
                        if q / np.sqrt(9.81 * s * d84 ** 3) < 100:
                            p = 0.23
                        else:
                            p = 0.3
                        tau_star_84 = s / (1.65 * d84 * (
                                    2 / w + 74 * p ** 2.6 * (9.81 * s) ** p * q ** (-2 * p) * d84 ** (3 * p - 1)))
                        tp = 14 * tau_star_84 ** 2.5 / (1 + (tau_star_m / tau_star_84) ** 4)
                        qsv = np.sqrt(9.81 * 1.65 * d84 ** 3) * tp
                        Qs = w * 2650 * qsv
                        sedyield_us += Qs * q_interval

        if sedyield_us > 0:
            print(reach_id, sedyield, sedyield_us)
            sdr[reach_id] = sedyield / sedyield_us
        else:
            sdr[reach_id] = None

    return sdr


# nr = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/burnt_fork/connectivity/network.tif'
# filled = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/burnt_fork/connectivity/pitfill.tif'
# fa = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/burnt_fork/connectivity/flow_acc.tif'
# fd = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/burnt_fork/connectivity/flow_dir.tif'
# sl = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/burnt_fork/connectivity/slope.tif'
# we = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/burnt_fork/connectivity/topo_weight.tif'
#
# hc = hillslope_connectivity(nr, filled, fa, fd, sl, we)
# print(hc)

reaches_in = '/home/jordan/Documents/Geoscience/grain-size/Input_data/Woods_network.shp'
gsds_in = '/home/jordan/Documents/Geoscience/grain-size/Input_data/gsd_woods2.json'
id_field_in = 'rid'
reachids_in = [8.039999999999999, 1.053, 1.055, 1.066, 1.091, 1.143]
upstream_id_in = ['rid_us', 'rid_us2']
discharge_in = '/media/jordan/Elements/Geoscience/Bitterroot/Woods/flow_2021/Woods_daily_q.csv'
flow_scale_field_in = 'flow_scale'
dhg = [0.282, 0.406]
whg = [4.688, 0.281]

sdr = channel_connectivity(reaches_in, gsds_in, id_field_in, reachids_in, upstream_id_in, discharge_in, 86400, flow_scale_field_in, dhg, whg, 0.016)
print(sdr)

# sdr = recking_channel_connectivity(reaches_in, id_field_in, reachids_in, upstream_id_in, discharge_in, 1800, flow_scale_field_in, whg)
# print(sdr)

# network_sdr(reaches_in, gsds_in, id_field_in, upstream_id_in, discharge_in, 1800, flow_scale_field_in, dhg, whg, min_fr=0.005)


# Blodgett sdr params
# reaches_in = '/home/jordan/Documents/Geoscience/grain-size/Input_data/Blodgett_network.shp'
# gsds_in = '/home/jordan/Documents/Geoscience/grain-size/Input_data/gsd_blodgett2.json'
# id_field_in = 'rid'
# reachids_in = [1.212, 1.216, 1.221, 1.247]
# upstream_id_in = ['rid_us', 'rid_us2']
# discharge_in = '/media/jordan/Elements/Geoscience/Bitterroot/Blodgett/flow_2021/Blodgett_daily_q.csv'
# flow_scale_field_in = 'flow_scale'
# dhg = [0.299, 0.215]
# whg = [8.542, 0.227]

# burnt fork sdr params
# reaches_in = '/home/jordan/Documents/Geoscience/grain-size/Input_data/Burnt_Fork_network.shp'
# gsds_in = '/home/jordan/Documents/Geoscience/grain-size/Input_data/gsd_burnt_fork2.json'
# id_field_in = 'rid'
# reachids_in = [1.097]
# upstream_id_in = ['rid_us', 'rid_us2']
# discharge_in = '/media/jordan/Elements/Geoscience/Bitterroot/Burnt_Fork/BF_flow/BF_daily_q.csv'
# flow_scale_field_in = 'flow_scale'
# dhg = [0.255, 0.416]
# whg = [4.058, 0.307]

# lost horse sdr params
# reaches_in = '/home/jordan/Documents/Geoscience/grain-size/Input_data/Lost_Horse_network.shp'
# gsds_in = '/home/jordan/Documents/Geoscience/grain-size/Input_data/gsd_lost_horse2.json'
# id_field_in = 'rid'
# reachids_in = [1.022, 1.13]
# upstream_id_in = ['rid_us', 'rid_us2']
# discharge_in = '/media/jordan/Elements/Geoscience/Bitterroot/Lost_Horse/LH_flow/LH_daily_q.csv'
# flow_scale_field_in = 'flow_scale'
# dhg = [0.234, 0.411]
# whg = [4.695, 0.318]

# roaring lion sdr params
# reaches_in = '/home/jordan/Documents/Geoscience/grain-size/Input_data/Roaring_Lion_network.shp'
# gsds_in = '/home/jordan/Documents/Geoscience/grain-size/Input_data/gsd_roaring_lion2.json'
# id_field_in = 'rid'
# reachids_in = [1.105]
# upstream_id_in = ['rid_us', 'rid_us2']
# discharge_in = '/media/jordan/Elements/Geoscience/Bitterroot/Roaring_Lion/RL_flow/RL_daily_q.csv'
# flow_scale_field_in = 'flow_scale'
# dhg = [0.227, 0.387]
# whg = [5.213, 0.324]

# sleeping child sdr params
# reaches_in = '/home/jordan/Documents/Geoscience/grain-size/Input_data/Sleeping_Child_network.shp'
# gsds_in = '/home/jordan/Documents/Geoscience/grain-size/Input_data/gsd_sleeping_child2.json'
# id_field_in = 'rid'
# reachids_in = [1.114, 1.128, 1.168]
# upstream_id_in = ['rid_us', 'rid_us2']
# discharge_in = '/media/jordan/Elements/Geoscience/Bitterroot/Sleeping_Child/sc_flow/SC_daily_q.csv'
# flow_scale_field_in = 'flow_scale'
# dhg = [0.23, 0.343]
# whg = [5.072, 0.377]