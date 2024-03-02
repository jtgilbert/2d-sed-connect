import geopandas as gpd
import json

network = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/woods/woods_connectivity_network.shp'
conn_vals = '/media/jordan/Elements/Geoscience/Bitterroot/lidar/woods/connected_areas.json'

dn = gpd.read_file(network)
with open(conn_vals, 'r') as src:
    vals = json.load(src)

for i in dn.index:
    if str(int(i)) in vals.keys():
        dn.loc[i, 'hs_conn'] = vals[str(int(i))]
    else:
        dn.loc[i, 'hs_conn'] = 0

dn.to_file(network)
