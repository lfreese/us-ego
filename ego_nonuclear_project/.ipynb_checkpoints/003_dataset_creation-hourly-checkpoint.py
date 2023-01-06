#!/home/emfreese/anaconda3/envs/grid_mod/bin/python



import numpy as np

import xarray as xr

import regionmask

import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cartopy.feature as cfeat
import matplotlib.patches as mpatches
import datetime

import xesmf as xe

import geopandas
from shapely.geometry import Point, Polygon

import feather

import glob

import calendar

from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from math import radians, cos, sin, asin, sqrt

import sys
sys.path.append('../')
import utils
import plotting


# Create Hourly Datasets
all_paths = ['renew_nonuc_NA','normal_NA','nei_NA','nonuc_NA','nonuc_coal_NA','renew_nonuc_NA','egrid_NA','epa_NA']

ds = {}

for poll in ['merged_O3']:
    ds[poll] = {}
    for path in all_paths:
        print(poll,path)
        dsb = xr.open_mfdataset(f'/net/fs11/d0/emfreese/GCrundirs/nuclearproj/{path}/merged_data/{poll}*.nc', chunks='auto')
        print('complete!')
        ds[poll][path] = dsb
        
print('combine and convert function')
ds = utils.combine_and_convert_O3([ds['SpeciesConc_O3'] for ds in list(ds['merged_O3'].values())], 
                    [nm for nm in ds['merged_O3'].keys()], 'all_models_ozone')

print('select summer months')
ds = ds.sel(time = slice('2016-04-01','2016-09-30'))

print('modify time zone')
#modify time zone
### https://maps.princeton.edu/catalog/stanford-nt016bt0491
tz = geopandas.read_file('final_data/Time_Zones.geojson')
#mask based on timezone
mask = {}

for loc in ['Central','Mountain','Pacific','Eastern']:
    mask[loc] = regionmask.Regions(tz.loc[tz['Zone'] == loc]['geometry']).mask(ds, lon_name = 'lon', lat_name = 'lat')

    
dsO3_masked = {}
    
for loc in ['Central','Mountain','Pacific','Eastern']:
    dsO3_masked[loc] = ds.where(~np.isnan(mask[loc]))

#create the change in time zone
z_time = np.arange(0,24)
z_time_convert = {'Central':5,'Mountain':6,'Pacific':7,'Eastern':4}

#apply and merge the different timezones
ds_summer = xr.merge([dsO3_masked['Central'],dsO3_masked['Mountain'],dsO3_masked['Pacific'],dsO3_masked['Eastern']])
ds_summer['time'] = pd.to_datetime(ds_summer['time'])
ds_summer.attrs = {'units':'ppb'}

#save hourly dataset out
ds_summer.to_netcdf('final_data/ds_O3_hourly.nc4')

print('mda8 calculation')
# ## MDA8 calculation
#ds_summer = xr.open_dataset('final_data/ds_O3_hourly.nc4')
ds_mda8 = utils.calc_mda8(ds_summer)
ds_mda8.to_netcdf('final_data/ds_O3_mda8.nc4')
