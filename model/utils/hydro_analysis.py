import os
from run_subprocess import run_subprocess


def wats_hydro(dem, type, pf=False):

    if pf is False:
        pit_fill = os.path.join(os.path.dirname(dem), 'pitfill_10m.tif')
        pitfill_status = run_subprocess(os.path.dirname(dem), ["mpiexec", "-n", "2", "pitremove", "-z", dem, "-fel", pit_fill])
        if pitfill_status != 0 or not os.path.isfile(pit_fill):
            raise Exception('TauDEM pitfill failed')
    else:
        pit_fill = dem

    if type == 'dinf':
        flow_dir = os.path.join(os.path.dirname(dem), 'flow_dir_10m.tif')
        slope = os.path.join(os.path.dirname(dem), 'slope_dinf.tif')
        flow_dir_status = run_subprocess(os.path.dirname(dem), ["mpiexec", "-n", "2", "dinfflowdir", "-fel", pit_fill, "-ang", flow_dir, "-slp", slope])
        if flow_dir_status != 0 or not os.path.isfile(flow_dir):
            raise Exception('TauDEM flow directions failed')
    else:
        flow_dir = os.path.join(os.path.dirname(dem), 'flow_dir_d8.tif')
        slope = os.path.join(os.path.dirname(dem), 'slope_d8.tif')
        flow_dir_status = run_subprocess(os.path.dirname(dem), ["mpiexec", "-n", "2", "d8flowdir", "-fel", pit_fill, "-p", flow_dir, "-sd8", slope])
        if flow_dir_status != 0 or not os.path.isfile(flow_dir):
            raise Exception('TauDEM flow directions failed')

    if type == 'dinf':
        flow_acc = os.path.join(os.path.dirname(dem), 'flow_acc_10m.tif')
        flow_acc_status = run_subprocess(os.path.dirname(dem), ["mpiexec", "-n", "2", "areadinf", "-ang", flow_dir, "-sca", flow_acc, "-nc"])
        if flow_acc_status != 0 or not os.path.isfile(flow_acc):
            raise Exception('TauDEM flow accumulation failed')
    else:
        flow_acc = os.path.join(os.path.dirname(dem), 'flow_acc_d8.tif')
        flow_acc_status = run_subprocess(os.path.dirname(dem), ["mpiexec", "-n", "2", "aread8", "-p", flow_dir, "-ad8", flow_acc, "-nc"])


dem_in = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/roaring_lion/pitfill_10m.tif'
wats_hydro(dem_in, type='d8', pf=True)
