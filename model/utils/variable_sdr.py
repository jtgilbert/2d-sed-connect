import pandas as pd
import geopandas as gpd
import numpy as np
import json
from tqdm import tqdm
from dsp_transport.calculate_transport import transport


def network_sdr(reaches, gsds, id_field, upstream_id, hydrograph, q_interval, flow_scale_field, depth_hydro_geom, width_hydro_geom, min_fr=None, ud_gsd=False):

    # reaches must have topological identifiers from network tools that match keys in the gsd dict
    # reaches must also be prepared with slope and upstream-downstream topology fields, width?
    sedyields = {}

    # open the gsd json as a dict
    with open(gsds) as gsdfile:
        gsd = json.load(gsdfile)

    reach_gsds = {}

    network = gpd.read_file(reaches)

    hydr = pd.read_csv(hydrograph)
    discharges = hydr['Q']

    for discharge in tqdm(discharges):

        ts_yields = {}  # keep track of fractions transported at each time step for updating gsds
        for i in network.index:

            d50 = max(network.loc[i, 'D50'] / 1000, 0.001)  # change to dynamic?
            d84 = max(network.loc[i, 'D84'] / 1000, 0.004)

            reach_id = network.loc[i, id_field]

            q = discharge * network.loc[i, flow_scale_field]
            depth = depth_hydro_geom[0] * q ** depth_hydro_geom[1]
            width = width_hydro_geom[0] * q ** width_hydro_geom[1]
            if i in reach_gsds.keys():
                fractions_tmp = reach_gsds[i]
            else:
                fractions_tmp = {float(key): val['fraction'] for key, val in gsd[str(i)]['fractions'].items()}
                reach_gsds[i] = fractions_tmp
            if min_fr is not None:
                fractions = update_fracs(fractions_tmp, min_fr)
            else:
                fractions = fractions_tmp
            qs = transport(fractions, max(network.loc[i, 'Slope'], 0.0001), q, depth, width, q_interval, d50_in=d50, d84_in=d84)
            qs_kg = {key: val[1] for key, val in qs.items()}
            # print(q, sum(qs_kg.values()))
            ts_yields[reach_id] = qs_kg
            if reach_id in sedyields.keys():
                sedyields[reach_id] += sum(qs_kg.values())
            else:
                sedyields[reach_id] = sum(qs_kg.values())

        if ud_gsd:
            for i in network.index:
                reach_id = network.loc[i, id_field]
                us_reach_id = network.loc[i, upstream_id[0]]
                us_reach_id2 = network.loc[i, upstream_id[1]]
                if str(us_reach_id) == 'nan':
                    continue
                if str(us_reach_id2) != 'nan':
                    us_yield1 = ts_yields[us_reach_id]
                    us_yield2 = ts_yields[us_reach_id2]
                    us_yield = {key: val + us_yield2[key] for key, val in us_yield1.items()}
                    new_gsd = update_gsd(reach_gsds[reach_id], us_yield, ts_yields[reach_id])
                    reach_gsds[reach_id] = new_gsd
                else:
                    new_gsd = update_gsd(reach_gsds[i], ts_yields[us_reach_id], ts_yields[reach_id])
                    reach_gsds[i] = new_gsd

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


def update_gsd(gsd_in, qs_in, qs_out):

    d_qs = {size: val - qs_out[size] for size, val in qs_in.items()}
    total = sum([abs(val) for val in d_qs.values()])
    if total == 0:
        return gsd_in
    else:
        d_prop = {size: val / total for size, val in d_qs.items()}
        print(f'combined proportions should equal 1: {sum([val for val in d_prop.values()])}')

        new_props_tmp = {size: val * (1 + d_prop[size]) for size, val in gsd_in.items()}
        new_props = {size: val / sum(new_props_tmp.values()) for size, val in new_props_tmp.items()}

        return new_props


reaches_in = '/home/jordan/Documents/Geoscience/grain-size/Input_data/Blodgett_network_100m.shp'
gsds_in = '/home/jordan/Documents/Geoscience/grain-size/Input_data/gsd_blodgett.json'
id_field_in = 'rid'
reachids_in = [1.053]
upstream_id_in = ['rid_us', 'rid_us2']
discharge_in = '/media/jordan/Elements/Geoscience/Bitterroot/Blodgett/flow_2021/Blodgett_daily_q.csv'
flow_scale_field_in = 'flow_scale'
dhg = [0.299, 0.215]
whg = [6.504, 0.348]

network_sdr(reaches_in, gsds_in, id_field_in, upstream_id_in, discharge_in, 1800, flow_scale_field_in, dhg, whg, min_fr=0.005, ud_gsd=True)