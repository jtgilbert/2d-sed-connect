from pysheds.grid import Grid
import rasterio
from rasterio.features import shapes
import numpy as np
import geopandas as gpd
import os
import geojson


def delineate_catchment(flowdir, pixel):

    directions = (3, 2, 1, 8, 7, 6, 5, 4)
    grid = Grid.from_raster(flowdir)
    fdir = grid.read_raster(flowdir)
    catch = grid.catchment(pixel[1], pixel[0], fdir, dirmap=directions, xytype='index')

    catch_arr = np.where(catch, 1.0, np.nan).astype(np.uint16)
    #
    # results = (
    #     {'properties': {'raster_val': v}, 'geometry': s}
    #     for i, (s, v)
    #     in enumerate(
    #     shapes(catch_arr, mask=catch_arr == 1., transform=transform)))
    #
    # geoms = list(results)
    # if len(geoms) == 0:
    #     raise Exception('no geoms')
    # shps = [g['geometry'] for g in geoms]

    return catch_arr

def util_delineate_basin(flowdir, coord):
    with rasterio.open(flowdir) as src:
        meta = src.profile

    directions = (3, 2, 1, 8, 7, 6, 5, 4)
    grid = Grid.from_raster(flowdir)
    fdir = grid.read_raster(flowdir)
    catch = grid.catchment(coord[0], coord[1], fdir, dirmap=directions, xytype='coordinate')

    catch_arr = np.where(catch, 1.0, np.nan).astype(np.uint16)

    with rasterio.open(os.path.join(os.path.dirname(flowdir), 'basin.tif'), 'w', **meta) as dst:
        dst.write(catch_arr, 1)

    return catch_arr


fd_in = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/conn_test/flow_dir.tif'
coord_in = [218489, 235313]
util_delineate_basin(fd_in, coord_in)
