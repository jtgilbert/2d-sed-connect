import json
import os
import rasterio
import numpy as np
import random
from tqdm import tqdm
from math import tan, atan, pi, sin, cos, exp

def rockfall(dem_in, slope_in, dest_raster, slope_thresh, divergence, persistence, mu_s):

    max_slope = 50 * pi / 180  # fixed rn
    slope_thresh = slope_thresh * pi / 180

    with rasterio.open(dem_in) as dem_src:
        meta = dem_src.profile
        nd = dem_src.nodata

        dist_str = dem_src.res[0]
        dist_diag = np.sqrt(dem_src.res[0]**2 + dem_src.res[1]**2)

        dem_array = dem_src.read()[0, :, :]
        rf_array = np.full(dem_array.shape, nd)  # array to store cells where rockfall ends up
        conn_array = np.full(dem_array.shape, nd)  # array to store connected rockfall source cells

    with rasterio.open(slope_in) as slope_src:
        slope_array = slope_src.read()[0, :, :]

    with rasterio.open(dest_raster) as dest_src:
        dest_array = dest_src.read()[0, :, :]

    if slope_array.shape != dem_array.shape:
        raise Exception('DEM and slope rasters have different shape')
    if dest_array.shape != dem_array.shape:
        raise Exception('DEM and sediment destination rasters have different shape')

    dists_dict = {1: dist_diag, 2: dist_str, 3: dist_diag, 4: dist_str, 5: dist_str, 6: dist_diag, 7: dist_str, 8: dist_diag}

    for r in tqdm(range(1, dem_array.shape[0]-1)):
        for c in range(1, dem_array.shape[1]-1):
            if slope_array[r, c] >= 50:
                print(f'generating rockfall at cell {[r, c]}')
                iter = 1
                while iter <= 5:
                    rf_array[r, c] = 0  # set source nodes to 0

                    # random walk to next cell
                    velocity = 10000
                    falling = True
                    outside = False
                    init_elev = dem_array[r, c]
                    prev_flow_dir = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
                    row, col = r, c
                    while velocity > 0:
                        if dem_array[row, col] == nd:
                            outside = True
                            iter += 1
                            break

                        try:
                            slopes = {
                                1: (dem_array[row,col]-dem_array[row+1,col-1])/dist_diag if dem_array[row+1,col-1] != nd else 0,
                                2: (dem_array[row,col]-dem_array[row+1,col])/dist_str if dem_array[row+1,col] != nd else 0,
                                3: (dem_array[row,col]-dem_array[row+1,col+1])/dist_diag if dem_array[row+1,col+1] != nd else 0,
                                4: (dem_array[row,col]-dem_array[row,col-1])/dist_str if dem_array[row,col-1] != nd else 0,
                                5: (dem_array[row,col]-dem_array[row,col+1])/dist_str if dem_array[row,col+1] != nd else 0,
                                6: (dem_array[row,col]-dem_array[row-1,col-1])/dist_diag if dem_array[row-1,col-1] != nd else 0,
                                7: (dem_array[row,col]-dem_array[row-1,col])/dist_str if dem_array[row-1,col] != nd else 0,
                                8: (dem_array[row,col]-dem_array[row-1,col+1])/dist_diag if dem_array[row-1,col+1] != nd else 0
                            }
                        except Exception as ex:
                            print(ex)
                            print('rockfall path exiting raster extents')
                            break

                        if max(slopes.values()) <= 0:
                            iter += 1
                            break
                        gamma_i = {key: max(0, val/tan(slope_thresh)) for key, val in slopes.items()}
                        gamma_max = max(gamma_i.values())
                        if 0 < gamma_max <= 1:
                            n_i = {key: 1 if val >= gamma_max**divergence else 0 for key, val in gamma_i.items()}
                        elif gamma_max > 1:
                            n_i = {key: 0 if val != gamma_max else 1 for key, val in gamma_i.items()}
                        sub_slopes = {key: slope if n_i[key] != 0 else 0 for key, slope in slopes.items()}
                        persists = False
                        for key, val in prev_flow_dir.items():
                            if val == 1:
                                if n_i[key] != 0:
                                    persists = True
                        if persists:
                            p_i = {key: val * persistence / sum(sub_slopes.values()) for key, val in sub_slopes.items()}
                        else:
                            p_i = {key: val / sum(sub_slopes.values()) for key, val in sub_slopes.items()}

                        direction = random.choices(list(p_i.keys()), list(p_i.values()))[0]
                        new_row_col = {
                            1: [row + 1, col - 1],
                            2: [row + 1, col],
                            3: [row + 1, col + 1],
                            4: [row, col - 1],
                            5: [row, col + 1],
                            6: [row - 1, col - 1],
                            7: [row - 1, col],
                            8: [row - 1, col + 1]
                        }
                        h_dict = {key: dem_array[row, col] - dem_array[new_row_col[key][0], new_row_col[key][1]] for key in new_row_col.keys()}
                        h = h_dict[direction]
                        h_f = init_elev - dem_array[new_row_col[direction][0], new_row_col[direction][1]]

                        if falling is True and sub_slopes[direction] >= max_slope:
                            v_i = np.sqrt(2*9.81*h_f)
                        elif falling is True and sub_slopes[direction] < max_slope:
                            v_i = np.sqrt(2*9.81*h_f) - 0.75 * np.sqrt(2*9.81*h_f)
                            falling = False
                        else:
                            v_i = np.sqrt(velocity**2 + 2*9.81*(h - mu_s * dists_dict[direction]))

                        velocity = v_i
                        # print(f'velocity: {velocity}')
                        prev_flow_dir = {key: 0 if key != direction else 1 for key in p_i.keys()}
                        row = new_row_col[direction][0]
                        col = new_row_col[direction][1]

                    if outside:
                        rf_array[row, col] = nd
                    else:
                        if rf_array[row, col] == nd:
                            rf_array[row, col] = 1
                        else:
                            rf_array[row, col] = rf_array[row, col] + 1
                        if dest_array[row, col] == 1:
                            conn_array[r, c] = 1
                    iter += 1

    with rasterio.open(os.path.join(os.path.dirname(dem_in), 'rockfall.tif'), 'w', **meta) as dst:
        dst.write(rf_array, 1)

    with rasterio.open(os.path.join(os.path.dirname(dem_in), 'connected_sources.tif'), 'w', **meta) as dst2:
        dst2.write(conn_array, 1)


def next_row_col(fd_val, row, col, processed):

    if 15 * pi / 8 < fd_val <= pi / 8:
        r = row
        c = col + 1
    elif pi / 8 < fd_val <= 3 * pi / 8:
        r = row - 1
        c = col + 1
    elif 3 * pi / 8 < fd_val <= 5 * pi / 8:
        r = row - 1
        c = col
    elif 5 * pi / 8 < fd_val <= 7 * pi / 8:
        r = row - 1
        c = col - 1
    elif 7 * pi / 8 < fd_val <= 9 * pi / 8:
        r = row
        c = col - 1
    elif 9 * pi / 8 < fd_val <= 11 * pi / 8:
        r = row + 1
        c = col - 1
    elif 11 * pi / 8 < fd_val <= 13 * pi / 8:
        r = row + 1
        c = col
    else:
        r = row + 1
        c = col + 1

    while [r, c] in processed:
        fd_val += pi/8
        if 15 * pi / 8 < fd_val <= pi / 8:
            r = row
            c = col + 1
        elif pi / 8 < fd_val <= 3 * pi / 8:
            r = row - 1
            c = col + 1
        elif 3 * pi / 8 < fd_val <= 5 * pi / 8:
            r = row - 1
            c = col
        elif 5 * pi / 8 < fd_val <= 7 * pi / 8:
            r = row - 1
            c = col - 1
        elif 7 * pi / 8 < fd_val <= 9 * pi / 8:
            r = row
            c = col - 1
        elif 9 * pi / 8 < fd_val <= 11 * pi / 8:
            r = row + 1
            c = col - 1
        elif 11 * pi / 8 < fd_val <= 13 * pi / 8:
            r = row + 1
            c = col
        else:
            r = row + 1
            c = col + 1

    return r, c


def df_source_cells(slope_array, da_array, fd_array, fd_nodata, chan_array):

    source_array = np.zeros(slope_array.shape)
    print('finding sources')
    for row in tqdm(range(slope_array.shape[0])):
        for col in range(slope_array.shape[1]):
            if tan(slope_array[row, col]*180/pi) * 100 > 0:
                crit_da = 4.3 * (tan(slope_array[row, col]*pi/180) * 100) ** -1.3
                if da_array[row, col] > crit_da and chan_array[row, col] != 1:
                    source_array[row, col] = 1

    # filter out cells downstream of other cells
    print('filtering sources (first pass)')
    for row in tqdm(range(source_array.shape[0])):
        for col in range(source_array.shape[1]):
            if source_array[row, col] == 1:
                processed = [[row, col]]
                next_r, next_c = next_row_col(fd_array[row, col], row, col, processed)
                while chan_array[next_r, next_c] != 1 and fd_array[next_r, next_c] != fd_nodata:
                    if source_array[next_r, next_c] == 1:
                        source_array[next_r, next_c] = 0

                    r, c = next_r, next_c
                    next_r, next_c = next_row_col(fd_array[r, c], r, c, processed)
                    processed.append([next_r, next_c])
    print('filtering sources (second pass)')
    for row in tqdm(range(source_array.shape[0])):
        for col in range(source_array.shape[1]):
            if source_array[row, col] == 1:
                processed = [[row, col]]
                next_r, next_c = next_row_col(fd_array[row, col], row, col, processed)
                while chan_array[next_r, next_c] != 1 and fd_array[next_r, next_c] != fd_nodata:
                    if len(processed) > 3:
                        if source_array[next_r, next_c] == 1:
                            source_array[next_r, next_c] = 0
                        if source_array[next_r, next_c+1] == 1:
                            source_array[next_r, next_c+1] = 0
                        if source_array[next_r-1, next_c+1] == 1:
                            source_array[next_r-1, next_c+1] = 0
                        if source_array[next_r-1, next_c] == 1:
                            source_array[next_r - 1, next_c] = 0
                        if source_array[next_r-1, next_c-1] == 1:
                            source_array[next_r - 1, next_c - 1] = 0
                        if source_array[next_r, next_c-1] == 1:
                            source_array[next_r, next_c - 1] = 0
                        if source_array[next_r+1, next_c-1] == 1:
                            source_array[next_r + 1, next_c - 1] = 0
                        if source_array[next_r + 1, next_c] == 1:
                            source_array[next_r + 1, next_c] = 0
                        if source_array[next_r + 1, next_c + 1] == 1:
                            source_array[next_r + 1, next_c + 1] = 0

                    r, c = next_r, next_c
                    next_r, next_c = next_row_col(fd_array[r, c], r, c, processed)
                    processed.append([next_r, next_c])

    return source_array


def debris_flow(dem_in, slope_in, fd_in, da_in, chan_in, slope_thresh, divergence, persistence, m_d):

    slope_thresh = slope_thresh * pi / 180

    with rasterio.open(dem_in) as dem_src:
        profile = dem_src.profile
        dist_str = dem_src.res[0]
        dist_diag = np.sqrt(dem_src.res[0] ** 2 + dem_src.res[1] ** 2)
        # profile.update(
        #     dtype=rasterio.uint8,
        #     count=1,
        #     compress='lzw')
        nd = dem_src.nodata
        dem_arr = dem_src.read()[0, :, :]
        df_arr = np.full(dem_arr.shape, nd)

    with rasterio.open(slope_in) as slope_src, rasterio.open(fd_in) as fd_src, rasterio.open(da_in) as da_src, rasterio.open(chan_in) as chan_src:
        slope_arr = slope_src.read()[0, :, :]
        fd_arr = fd_src.read()[0, :, :]
        fd_nd = fd_src.nodata
        da_arr = da_src.read()[0, :, :]
        chan_arr = chan_src.read()[0, :, :]

    if dem_arr.shape != slope_arr.shape or dem_arr.shape != fd_arr.shape or dem_arr.shape != da_arr.shape:
        raise Exception('Input rasters must have same shape')

    dists_dict = {1: dist_diag, 2: dist_str, 3: dist_diag, 4: dist_str, 5: dist_str, 6: dist_diag, 7: dist_str,
                  8: dist_diag}

    source_cells = df_source_cells(slope_arr, da_arr, fd_arr, fd_nd, chan_arr)
    # print('copying to new array')
    # sc = np.full(source_cells.shape, nd)
    # for row in range(source_cells.shape[0]):
    #     for col in range(source_cells.shape[1]):
    #         if source_cells[row, col] == 1:
    #             sc[row, col] = 1

    # delete later
    # with rasterio.open(os.path.join(os.path.dirname(dem), 'df_head.tif'), 'w', **profile) as dst:
    #     dst.write(sc.astype(rasterio.uint8), 1)

    for r in range(source_cells.shape[0]):
        for c in range(source_cells.shape[1]):
            if source_cells[r, c] == 1:
                print(f'generating debris flow at {[r, c]}')
                iter = 1
                while iter <= 3:
                    processed = [[r, c]]
                    df_arr[r, c] = 1

                    # random walk to next cell
                    velocity = 0.001
                    prev_flow_dir = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
                    row, col = r, c
                    while velocity > 0:
                        if dem_arr[row, col] == nd or fd_arr[row, col] == fd_nd:
                            velocity = 0
                            iter = 10
                            break

                        slopes = {
                            1: (dem_arr[row, col] - dem_arr[row + 1, col - 1]) / dist_diag if dem_arr[
                                                                                                      row + 1, col - 1] != nd else 0,
                            2: (dem_arr[row, col] - dem_arr[row + 1, col]) / dist_str if dem_arr[
                                                                                                 row + 1, col] != nd else 0,
                            3: (dem_arr[row, col] - dem_arr[row + 1, col + 1]) / dist_diag if dem_arr[
                                                                                                      row + 1, col + 1] != nd else 0,
                            4: (dem_arr[row, col] - dem_arr[row, col - 1]) / dist_str if dem_arr[
                                                                                                 row, col - 1] != nd else 0,
                            5: (dem_arr[row, col] - dem_arr[row, col + 1]) / dist_str if dem_arr[
                                                                                                 row, col + 1] != nd else 0,
                            6: (dem_arr[row, col] - dem_arr[row - 1, col - 1]) / dist_diag if dem_arr[
                                                                                                      row - 1, col - 1] != nd else 0,
                            7: (dem_arr[row, col] - dem_arr[row - 1, col]) / dist_str if dem_arr[
                                                                                                 row - 1, col] != nd else 0,
                            8: (dem_arr[row, col] - dem_arr[row - 1, col + 1]) / dist_diag if dem_arr[
                                                                                                      row - 1, col + 1] != nd else 0
                        }
                        if max(slopes.values()) <= 0:
                            for key, val in slopes.items():
                                if val == 0:
                                    slopes[key] = 0.001
                        gamma_i = {key: max(0, val / tan(slope_thresh)) for key, val in slopes.items()}
                        gamma_max = max(gamma_i.values())
                        if 0 < gamma_max <= 1:
                            n_i = {key: 1 if val >= gamma_max ** divergence else 0 for key, val in gamma_i.items()}
                        elif gamma_max > 1:
                            n_i = {key: 0 if val != gamma_max else 1 for key, val in gamma_i.items()}
                        sub_slopes = {key: slope if n_i[key] != 0 else 0 for key, slope in slopes.items()}
                        persists = False
                        for key, val in prev_flow_dir.items():
                            if val == 1:
                                if n_i[key] != 0:
                                    persists = True
                        if persists:
                            p_i = {key: val * persistence / sum(sub_slopes.values()) for key, val in sub_slopes.items()}
                        else:
                            p_i = {key: val / sum(sub_slopes.values()) for key, val in sub_slopes.items()}

                        direction = random.choices(list(p_i.keys()), list(p_i.values()))[0]
                        new_row_col = {
                            1: [row + 1, col - 1],
                            2: [row + 1, col],
                            3: [row + 1, col + 1],
                            4: [row, col - 1],
                            5: [row, col + 1],
                            6: [row - 1, col - 1],
                            7: [row - 1, col],
                            8: [row - 1, col + 1]
                        }

                        area = {1: da_arr[new_row_col[1][0], new_row_col[1][1]], 2: da_arr[new_row_col[2][0], new_row_col[2][1]], 3: da_arr[new_row_col[3][0], new_row_col[3][1]],
                                         4: da_arr[new_row_col[4][0], new_row_col[4][1]], 5: da_arr[row, col], 6: da_arr[new_row_col[5][0], new_row_col[5][1]],
                                         7: da_arr[new_row_col[6][0], new_row_col[6][1]], 8: da_arr[new_row_col[7][0], new_row_col[7][1]], 9: da_arr[new_row_col[8][0], new_row_col[8][1]]}
                        mu_i = 0.13 * (max(area.values())*1000000)**-0.4
                        #mu_i = 1e-10
                        s_percent = slopes[direction]
                        s_rad = atan(s_percent)
                        slope_len = np.sqrt((dem_arr[r, c] - dem_arr[new_row_col[direction][0], new_row_col[direction][1]])**2 + dists_dict[direction]**2)
                        alpha_i = 9.81*(sin(s_rad) - mu_i*cos(s_rad))
                        beta_i = (-2 * slope_len) / m_d
                        v_i = np.sqrt(alpha_i * m_d * (1 - exp(beta_i)) + velocity**2*exp(beta_i))

                        velocity = v_i
                        prev_flow_dir = {key: 0 if key != direction else 1 for key in p_i.keys()}
                        processed.append([row, col])
                        row = new_row_col[direction][0]
                        col = new_row_col[direction][1]
                        df_arr[row, col] = 1
                        if [row, col] in processed:
                            iter = 5
                            velocity = 0
                            break

                        if chan_arr[row, col] == 1:
                            velocity = 0

                    iter += 1

    with rasterio.open(os.path.join(os.path.dirname(dem), 'debris_flow.tif'), 'w', **profile) as dst:
        dst.write(df_arr, 1)


def hsca(thresh_slope, flow_dir, destinations):

    with rasterio.open(thresh_slope) as src, rasterio.open(flow_dir) as fd_src, rasterio.open(destinations) as dest_src:
        profile = src.profile
        nd = src.nodata
        slope_arr = src.read()[0, :, :]
        hsca_array = np.full(slope_arr.shape, nd)
        fd_arr = fd_src.read()[0, :, :]
        fd_nd = fd_src.nodata
        dest_arr = dest_src.read()[0, :, :]
        dest_nd = dest_src.nodata

    processed = []
    for row in tqdm(range(1, slope_arr.shape[0]-1)):
        for col in range(1, slope_arr.shape[1]-1):
            if slope_arr[row, col] == 1:
                subprocessed = []
                subprocessed.append([row, col])
                next_r, next_c = next_row_col(fd_arr[row, col], row, col, subprocessed)
                try:
                    while slope_arr[next_r, next_c] == 1 and dest_arr[next_r, next_c] == dest_nd and fd_arr[next_r, next_c] != fd_nd:
                        if [next_r, next_c] not in processed:
                            subprocessed.append([next_r, next_c])
                            next_r, next_c = next_row_col(fd_arr[next_r, next_c], next_r, next_c, subprocessed)
                        else:
                            break
                    if dest_arr[next_r, next_c] == 1 or hsca_array[next_r, next_c] == 1:
                        for coords in subprocessed:
                            hsca_array[coords[0], coords[1]] = 1
                    else:
                        for coords in subprocessed:
                            hsca_array[coords[0], coords[1]] = 0
                    for coords in subprocessed:
                        processed.append(coords)
                except Exception as ex:
                    print(ex)
                    continue

    with rasterio.open(os.path.join(os.path.dirname(thresh_slope), 'hsca.tif'), 'w', **profile) as dst:
        dst.write(hsca_array, 1)


def hs_channel_connectivity(connected_slopes, flow_dir, channel):
    out_vals = {}
    out_area = {}
    with rasterio.open(connected_slopes) as src, rasterio.open(flow_dir) as fd_src, rasterio.open(channel) as dest_src:
        profile = src.profile
        xres = src.transform[0]
        yres = src.transform[4]
        nd = src.nodata
        slope_arr = src.read()[0, :, :]
        out_array = np.full(slope_arr.shape, nd)
        fd_arr = fd_src.read()[0, :, :]
        fd_nd = fd_src.nodata
        dest_arr = dest_src.read()[0, :, :]
        dest_nd = dest_src.nodata

    processed = []
    for row in tqdm(range(1, slope_arr.shape[0]-1)):
        for col in range(1, slope_arr.shape[1]-1):
            if slope_arr[row, col] == 1 and dest_arr[row, col] == dest_nd:
                subprocessed = []
                subprocessed.append([row, col])
                next_r, next_c = next_row_col(fd_arr[row, col], row, col, subprocessed)
                try:
                    while dest_arr[next_r, next_c] == dest_nd and fd_arr[next_r, next_c] != fd_nd and out_array[next_r, next_c] == nd:
                        if [next_r, next_c] not in processed:
                            subprocessed.append([next_r, next_c])
                            next_r, next_c = next_row_col(fd_arr[next_r, next_c], next_r, next_c, subprocessed)
                        else:
                            break
                    if dest_arr[next_r, next_c] != dest_nd or out_array[next_r, next_c] != nd:
                        assign_val = max(dest_arr[next_r, next_c], out_array[next_r, next_c])
                        for coords in subprocessed:
                            out_array[coords[0], coords[1]] = assign_val
                        if assign_val not in out_vals.keys():
                            out_vals[assign_val] = len(subprocessed)
                        else:
                            out_vals[assign_val] += len(subprocessed)
                    else:
                        for coords in subprocessed:
                            out_array[coords[0], coords[1]] = 0
                    for coords in subprocessed:
                        processed.append(coords)
                except Exception as ex:
                    print(ex)
                    continue

    for key, val in out_vals.items():
        if key is not None:
            out_area[str(int(key))] = val * abs(xres) * abs(yres)

    with rasterio.open(os.path.join(os.path.dirname(connected_slopes), 'hillslope_conn.tif'), 'w', **profile) as dst:
        dst.write(out_array, 1)

    with open(os.path.join(os.path.dirname(connected_slopes), 'connected_areas.json'), 'w') as jsondst:
        json.dump(out_area, jsondst, indent=2)



# dem = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/lost_horse/Lost_Horse_DEM_10m.tif'
# slope = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/lost_horse/slope_10m.tif'
# sed_dest = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/lost_horse/sed_dest_10m.tif'
# rockfall(dem, slope, sed_dest, 40, 5, 3, 0.7)

# dem = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/roaring_lion/Roaring_Lion_DEM_10m.tif'
# slope = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/roaring_lion/slope_10m.tif'
# fd = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/roaring_lion/flow_dir_10m.tif'
# da = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/roaring_lion/dr_area_10m.tif'
# chan = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/roaring_lion/network_10m.tif'
#
# debris_flow(dem, slope, fd, da, chan, 20, 1.3, 1.5, 75)

# thresholded_slope = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/burnt_fork/thresh_slope.tif'
# flow_direction = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/burnt_fork/flow_dir_10m.tif'
# destination_raster = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/burnt_fork/dest_raster.tif'
# hsca(thresholded_slope, flow_direction, destination_raster)

connected_slope = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/woods/connected_hillslopes.tif'
flow_direction = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/woods/flow_dir_10m.tif'
stream_raster = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/woods/network_id.tif'
hs_channel_connectivity(connected_slope, flow_direction, stream_raster)
