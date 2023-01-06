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

#create our daily datasets
for poll in ['merged_O3','merged_PM']:
    for path in all_paths:
        ds = xr.open_mfdataset(f'/net/fs11/d0/emfreese/GCrundirs/nuclearproj/{path}/merged_data/{poll}*.nc')
        ds = ds.groupby('time.date').mean(dim = 'time')
        ds['date'] = pd.to_datetime(ds['date'])
        ds = ds.rename({'date':'time'})
        ds.to_netcdf(f'/net/fs11/d0/emfreese/GCrundirs/nuclearproj/{path}/merged_data/daily_mean/{poll}_daily_mean.nc')
        print(poll, path, 'done')

