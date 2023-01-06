#!/usr/bin/env python
# coding: utf-8

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


all_paths = ['renew_nonuc_NA', 'normal_NA','nei_NA','nonuc_NA','nonuc_coal_NA','renew_nonuc_NA','egrid_NA','epa_NA']



# # Prep NOx, SO2, VOC data

##module load cdo
##create specific file and merge each of them: 
#for file in GEOSChem.SpeciesConc.2016*; do cdo -selvar,SpeciesConc_CH2O $file CH2O_$file; done && for month in {01,02,03,04,05,06,07,08,09,10,11,12}; do cdo mergetime CH2O*2016$month*.nc4 ../merged_data/merged_CH2O_$month.nc ; done && for file in GEOSChem.SpeciesConc.2016*; do cdo -selvar,SpeciesConc_NO $file NO_$file; done && for month in {01,02,03,04,05,06,07,08,09,10,11,12}; do cdo mergetime NO*2016$month*.nc4 ../merged_data/merged_NO_$month.nc ; done && for file in GEOSChem.SpeciesConc.2016*; do cdo -selvar,SpeciesConc_NO2 $file NO2_$file; done && for month in {01,02,03,04,05,06,07,08,09,10,11,12}; do cdo mergetime NO2*2016$month*.nc4 ../merged_data/merged_NO2_$month.nc ; done && for file in GEOSChem.SpeciesConc.2016*; do cdo -selvar,SpeciesConc_SO2 $file SO2_$file; done && for month in {01,02,03,04,05,06,07,08,09,10,11,12}; do cdo mergetime SO2*2016$month*.nc4 ../merged_data/merged_SO2_$month.nc ; done


ds = {}

ds['merged_NO'] = {}
ds['merged_NO2'] = {}
ds['merged_CH2O'] = {}
ds['merged_SO2'] = {}

for poll in ['merged_NO','merged_NO2','merged_CH2O','merged_SO2']:
    for path in all_paths:
        ds[poll][path] = xr.open_mfdataset(f'/net/fs11/d0/emfreese/GCrundirs/nuclearproj/{path}/merged_data/{poll}_*.nc', combine = 'by_coords')

ds_gas = {}

for gas in ['NO2','NO','SO2','CH2O']:
    ds_gas[gas] = utils.convert_gases([ds for ds in list(ds[f'merged_{gas}'].values())], [nm for nm in ds[f'merged_{gas}'].keys()], gas)

ds_out = xr.merge([ds_gas[gas] for gas in ['NO2','NO','SO2','CH2O']])

ds_out = ds_out.groupby('time.date').mean(dim = 'time').rename({'date':'time'})

ds_out['time'] = pd.to_datetime(ds_out['time'])

#sum our NO2 and NO to get NOx
ds_out['NOx'] = (ds_out['NO'] + ds_out['NO2'])

ds_out.to_netcdf('final_data/ds_NOX_SO2_CH2O_daily.nc4')