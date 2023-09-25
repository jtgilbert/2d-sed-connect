from pysheds.grid import Grid
import rasterio
from rasterio.features import shapes
import numpy as np


def delinate_catchment(flowdir, pixel):

    with rasterio.open(flowdir) as src:
        transform = src.transform

    directions = [3, 2, 1, 8, 7, 6, 5, 4]
    grid = Grid.from_raster(flowdir)
    fdir = grid.read_raster(flowdir) # does this need different direction map?
    catch = grid.catchment(pixel[0], pixel[0], fdir, dirmap=directions)

    catch_arr = np.where(catch, 1, np.nan).astype(np.int16)

    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v)
        in enumerate(
        shapes(catch_arr, mask=catch_arr == 1., transform=transform)))

    geoms = list(results)
    if len(geoms) == 0:
        raise Exception('no geoms')
    shps = [g['geometry'] for g in geoms]

    return shps
