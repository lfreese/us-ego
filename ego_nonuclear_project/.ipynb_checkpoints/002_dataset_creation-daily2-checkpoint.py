#!/home/emfreese/anaconda3/envs/grid_mod/bin/python
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH -p fdr


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



all_paths = ['renew_nonuc_NA','normal_NA','nei_NA','nonuc_NA','nonuc_coal_NA','renew_nonuc_NA','egrid_NA','epa_NA']

#combine all models of daily datasets
ds = {}

ds['merged_PM'] = {}
ds['merged_O3'] = {}
for poll in ['merged_O3','merged_PM']:
    for path in all_paths:
        dsb = xr.open_dataset(f'/net/fs11/d0/emfreese/GCrundirs/nuclearproj/{path}/merged_data/daily_mean/{poll}_daily_mean.nc')
        #dsb = dsb.rename({'date':'time'})
        print(poll,path)
        ds[poll][path] = dsb.groupby('time.date').mean()
ds = utils.combine_and_convert_PM_O3([ds['SpeciesConc_O3'] for ds in list(ds['merged_O3'].values())], 
                    [ds['PM25'] for ds in list(ds['merged_PM'].values())],
                    [nm for nm in ds['merged_O3'].keys()],
                    [nm for nm in ds['merged_PM'].keys()], 'all_models')

ds = ds.rename({'date':'time'})
ds['time'] = pd.to_datetime(ds['time'])

ds['O3'].attrs = {'units':'ppb'}
ds['PM25'].attrs = {'units':r'$u$g/m3'}
ds.to_netcdf('final_data/ds_PM_O3_daily.nc4')

