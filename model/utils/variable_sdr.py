import pandas as pd
import geopandas as gpd
import numpy as np
import json
from tqdm import tqdm
from dsp_transport.calculate_transport import transport


def network_sdr(reaches, gsds, id_field, upstream_id, hydrograph, q_interval, flow_scale_field, depth_hydro_geom,
                width_hydro_geom, min_fr=None, ud_gsd=False, dist_reach=None, dist_vol=None, dist_gsd=None):

    # reaches must have topological identifiers from network tools that match keys in the gsd dict
    # reaches must also be prepared with slope and upstream-downstream topology fields, width?
    sedyields = {}

    # open the gsd json as a dict
    with open(gsds) as gsdfile:
        gsd = json.load(gsdfile)

    reach_gsds = {}
    if dist_gsd is not None:
        reach_gsds[dist_reach] = dist_gsd
        dist_mass = {key: (val * dist_vol) * (2650 * 0.79) for key, val in dist_gsd.items()}

    network = gpd.read_file(reaches)

    hydr = pd.read_csv(hydrograph)
    discharges = hydr['Q']

    if ud_gsd:  # if introducing disturbance, update gsds for reaches downstream of it ...
        print('getting all reaches downstream of disturbance')
        ds_reaches = []
        for i in network.index:
            if network.loc[i, id_field] == dist_reach:
                next_ds = network.loc[i, 'rid_ds']
                while str(next_ds) != 'nan':
                    ds_reaches.append(next_ds)
                    for j in network.index:
                        if network.loc[j, id_field] == next_ds:
                            next_ds = network.loc[j, 'rid_ds']

    for discharge in tqdm(discharges):

        ts_yields = {}  # keep track of fractions transported at each time step for updating gsds
        for i in network.index:

            d50 = max(network.loc[i, 'D50'] / 1000, 0.001)  # change to dynamic?
            if network.loc[i, 'Slope'] < 0.05:
                d84 = max(network.loc[i, 'D84'] / 1000, 0.004)
            else:
                d84 = max(network.loc[i, 'D84_high'] / 1000, 0.004)

            reach_id = network.loc[i, id_field]

            q = discharge * network.loc[i, flow_scale_field]
            depth = (depth_hydro_geom[0] * q ** depth_hydro_geom[1]) * (0.011 / network.loc[i, 'Slope'])**0.75  # correction for steeper or less steep channels... based on relationship from Mannings
            width = width_hydro_geom[0] * q ** width_hydro_geom[1]
            if i in reach_gsds.keys():
                fractions_tmp = reach_gsds[i]
            else:
                fractions_tmp = {float(key): val['fraction'] for key, val in gsd[str(i)]['fractions'].items()}
                reach_gsds[i] = fractions_tmp
            if min_fr is not None:
                if network.loc[i, 'Slope'] <= 0.08:  # make sure there's sand in less steep reaches...
                    fractions = update_fracs(fractions_tmp, min_fr)
                else:
                    fractions = fractions_tmp
            else:
                fractions = fractions_tmp
            # if reach_id == 5.12:
            #     print('checking')
            qs = transport(fractions, max(network.loc[i, 'Slope'], 0.0001), q, depth, width, q_interval, d50_in=d50, d84_in=d84)
            qs_kg = {key: val[1] for key, val in qs.items()}

            if network.loc[i, id_field] == dist_reach:
                for key, val in qs_kg.items():
                    if val < dist_mass[key]:
                        dist_mass[key] -= val
                    else:
                        qs_kg[key] = dist_mass[key]
                        dist_mass[key] = 0.

            ts_yields[reach_id] = qs_kg
            if reach_id in sedyields.keys():
                sedyields[reach_id] += sum(qs_kg.values())
            else:
                sedyields[reach_id] = sum(qs_kg.values())

        if ud_gsd:  # maybe add logic here to only apply this while dist_vol is still > 0 then return to orig gsds...
            for i in network.index:
                reach_id = network.loc[i, id_field]
                if reach_id in ds_reaches:
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
            network.loc[i, 'SDR_v'] = np.log10(sdr)
        else:
            network.loc[i, 'SDR_v'] = np.log10(1e-6)

        network.loc[i, 'sp_yield'] = (reach_yield/1000) / network.loc[i, 'Drain_Area']

    network.to_file(reaches)
    with open('/media/jordan/Elements/Geoscience/Bitterroot/Blodgett/flow_2021/reach_yield.json', 'w') as f:
        json.dump(sedyields, f, indent=4)

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

        new_props_tmp = {size: val * (1 + d_prop[size]) for size, val in gsd_in.items()}
        new_props = {size: val / sum(new_props_tmp.values()) for size, val in new_props_tmp.items()}
        if not 0.99 < sum(new_props.values()) < 1.1:
            raise Exception('does not equal 1')

        return new_props


def sdr_notransport(network_in):
    network = gpd.read_file(network_in)

    for i in network.index:
        us_yield = 0
        ds_yield = 0
        if network.loc[i, 'sp_yield'] == 0:
            us_id1 = network.loc[i, 'rid_us']
            us_id2 = network.loc[i, 'rid_us2']
            ds_id = network.loc[i, 'rid_ds']
            for j in network.index:
                if network.loc[j, 'rid'] == us_id1:
                    us_yield += network.loc[j, 'sp_yield']
                if network.loc[j, 'rid'] == us_id2:
                    us_yield += network.loc[j, 'sp_yield']
                if network.loc[j, 'rid'] == ds_id:
                    ds_yield += network.loc[j, 'sp_yield']

            if us_yield == 0 and ds_yield == 0:
                network.loc[i, 'SDR_v'] = 0

    network.to_file(network_in)



reaches_in = '/home/jordan/Documents/Geoscience/grain-size/Input_data/Blodgett_network.shp'
gsds_in = '/home/jordan/Documents/Geoscience/grain-size/Input_data/gsd_blodgett2.json'
id_field_in = 'rid'
reachids_in = [1.053]
upstream_id_in = ['rid_us', 'rid_us2']
discharge_in = '/media/jordan/Elements/Geoscience/Bitterroot/Blodgett/flow_2021/Blodgett_daily_q.csv'
flow_scale_field_in = 'flow_scale'
dhg = [0.299, 0.215]
whg = [8.542, 0.227]

disturbance_gsd = {
    1.: 0.01,
    0.: 0.01,
    -1.: 0.02,
    -2.: 0.02,
    -2.5: 0.02,
    -3.: 0.02,
    -3.5: 0.03,
    -4.: 0.03,
    -4.5: 0.05,
    -5.: 0.06,
    -5.5: 0.06,
    -6.: 0.09,
    -6.5: 0.1,
    -7.: 0.17,
    -7.5: 0.15,
    -8.: 0.09,
    -8.5: 0.06,
    -9.: 0.01
}

network_sdr(reaches_in, gsds_in, id_field_in, upstream_id_in, discharge_in, 86400, flow_scale_field_in,
            dhg, whg, min_fr=0.005)  #, ud_gsd=True, dist_reach=1.234, dist_vol=930, dist_gsd=disturbance_gsd)

# sdr_notransport(reaches_in)
