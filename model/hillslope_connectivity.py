import os
import rasterio
import numpy as np
import random
from tqdm import tqdm
from math import tan, atan, pi

def rockfall(dem_in, slope_in, slope_thresh, divergence, persistence, mu_s):

    max_slope = 60 * pi / 180  # fixed rn
    slope_thresh = slope_thresh * pi / 180

    with rasterio.open(dem_in) as dem_src:
        meta = dem_src.profile
        nd = dem_src.nodata

        dist_str = dem_src.res[0]
        dist_diag = np.sqrt(dem_src.res[0]**2 + dem_src.res[1]**2)

        dem_array = dem_src.read()[0, :, :]
        out_array = np.full(dem_array.shape, nd)

    with rasterio.open(slope_in) as slope_src:
        slope_array = slope_src.read()[0, :, :]

    if slope_array.shape != dem_array.shape:
        raise Exception('DEM and slope rasters have different shape')

    dists_dict = {1: dist_diag, 2: dist_str, 3: dist_diag, 4: dist_str, 5: dist_str, 6: dist_diag, 7: dist_str, 8: dist_diag}

    for r in tqdm(range(dem_array.shape[0])):
        for c in range(dem_array.shape[1]):
            if slope_array[r, c] >= 60:
                print(f'generating rockfall at cell {[r, c]}')
                iter = 1
                while iter <= 10:
                    out_array[r, c] = 0  # set source nodes to 0

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
                        iter += 1

                    if outside:
                        out_array[row, col] = nd
                    else:
                        if out_array[row, col] == nd:
                            out_array[row, col] = 1
                        else:
                            out_array[row, col] = out_array[row, col] + 1

    with rasterio.open(os.path.join(os.path.dirname(dem_in), 'rockfall.tif'), 'w', **meta) as dst:
        dst.write(out_array, 1)


dem = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/connectivity/talus_fan/dem_sub.tif'
slope = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/connectivity/talus_fan/slope_sub.tif'
rockfall(dem, slope, 30, 6, 5, 0.6)
