import os
from run_subprocess import run_subprocess


def wats_hydro(dem):

    pit_fill = os.path.join(os.path.dirname(dem), 'pitfill.tif')
    pitfill_status = run_subprocess(os.path.dirname(dem), ["mpiexec", "-n", "2", "pitremove", "-z", dem, "-fel", pit_fill])
    if pitfill_status != 0 or not os.path.isfile(pit_fill):
        raise Exception('TauDEM pitfill failed')

    flow_dir = os.path.join(os.path.dirname(dem), 'flow_dir.tif')
    slope = os.path.join(os.path.dirname(dem), 'slope.tif')
    flow_dir_status = run_subprocess(os.path.dirname(dem), ["mpiexec", "-n", "2", "d8flowdir", "-fel", pit_fill, "-p", flow_dir, "-sd8", slope])
    if flow_dir_status != 0 or not os.path.isfile(flow_dir):
        raise Exception('TauDEM flow directions failed')

    flow_acc = os.path.join(os.path.dirname(dem), 'flow_acc.tif')
    flow_acc_status = run_subprocess(os.path.dirname(dem), ["mpiexec", "-n", "2", "aread8", "-p", flow_dir, "-ad8", flow_acc, "-nc"])
    if flow_acc_status != 0 or not os.path.isfile(flow_acc):
        raise Exception('TauDEM flow accumulation failed')


dem_in = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/blodgett/Blodgett_DEM_2m_sub.tif'
wats_hydro(dem_in)