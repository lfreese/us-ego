#!/home/emfreese/anaconda3/envs/grid_mod/bin/python
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
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

## EPA Observational data
#data is from https://aqs.epa.gov/aqsweb/airdata/download_files.html
#for the year 2016
#choosing O3, NO, SO2, PM25 (FEM/FRM)


all_paths = ['normal_NA','nei_NA','nonuc_NA','nonuc_coal_NA','renew_nonuc_NA','egrid_NA','epa_NA']

ds = {}
ds['merged_NIT'] = {}
ds['merged_SO4'] = {}
for poll in ['merged_NIT','merged_SO4']:
    for path in all_paths:
        ds[poll][path] = xr.open_mfdataset(f'/net/fs11/d0/emfreese/GCrundirs/nuclearproj/{path}/merged_data/daily_mean/{poll}_*.nc', combine = 'by_coords')
        
for path in all_paths:
    print(path)
    ds['merged_NIT'][path] = ds['merged_NIT'][path]['AerMassNIT'].to_dataset()
    ds['merged_SO4'][path] = ds['merged_SO4'][path]['SpeciesConc_SO4'].to_dataset()
    
ds_gas = {}
ds_gas['SO4'] = utils.convert_gases([ds for ds in list(ds[f'merged_SO4'].values())], [nm for nm in ds[f'merged_SO4'].keys()], 'SO4')
ds_gas['SO4'] = ds_gas['SO4'].rename({f'SpeciesConc_SO4':f'SO4'})
ds_gas['NIT'] = utils.convert_aerosol([ds for ds in list(ds[f'merged_NIT'].values())], [nm for nm in ds[f'merged_NIT'].keys()], 'NIT')



ds_out = xr.merge([ds_gas[gas] for gas in ['NIT','SO4']])
ds_out = ds_out.groupby('time.date').mean(dim = 'time').rename({'date':'time'})
ds_out['time'] = pd.to_datetime(ds_out['time'])

ds2 = xr.open_dataset('final_data/ds_NOX_SO2_CH2O_daily.nc4')
ds3 = xr.open_dataset('final_data/ds_PM_O3_daily.nc4')

poll_ds = ds_out.merge(ds2)
poll_ds = poll_ds.merge(ds3)

EPA_obs_df = utils.import_and_edit_EPAobs('../../GEOS_CHEM/obs_data/daily*.csv')


pm_df = utils.import_IMPROVE('../../GEOS_CHEM/obs_data/IMPROVE_2016_PM.txt', 'PM25', 'MF')
s_df = utils.import_IMPROVE('../../GEOS_CHEM/obs_data/IMPROVE_2016_Sulfate.txt', 'SO4', 'SO4f')
n_df = utils.import_IMPROVE('../../GEOS_CHEM/obs_data/IMPROVE_2016_Nitrate.txt', 'NIT', 'NO3f')
ammon_df = utils.import_IMPROVE('../../GEOS_CHEM/obs_data/IMPROVE_2016_ammonia.txt', 'NH4', 'NH4f')
oc_df = utils.import_IMPROVE('../../GEOS_CHEM/obs_data/IMPROVE_2016_OC.txt', 'OC', 'ECf')

IMPROVE_df = pd.concat([pm_df, s_df, n_df,oc_df, ammon_df], axis = 0) #concatenate all dataframes and reset the index
IMPROVE_df['Date'] = pd.to_datetime(IMPROVE_df['Date']) #change to datetime
IMPROVE_df = IMPROVE_df.loc[IMPROVE_df['Arithmetic Mean'] >= 0] #get rid of -999 readings where there is no data
IMPROVE_df = IMPROVE_df.loc[(IMPROVE_df['Latitude'].between(24,50,inclusive = True)) & (IMPROVE_df['Longitude'].between(-130,-60,inclusive = True))]


#define Lat and Lon of the nested grid US
levels_dict = {'PM25':np.arange(0., 20., .5), 'SO2':np.arange(0., 5., .1), 
               'NO2':np.arange(0., 5., .1), 'NOx':np.arange(0., 5., .1), 'O3':np.arange(0., 70., 1.),
               'dif':np.arange(-.3, .31, .01), 'regional_dif':np.arange(-1.5, 1.51, .01)}


### interpolate data for EPA
interp_EPA_df = pd.DataFrame(columns=['Arithmetic Mean', 'Longitude', 'Latitude','model_name','species','date'])

for model in ['egrid_NA', 'nei_NA', 'epa_NA', 'normal_NA']:
    print(model, end = ', ')
    for species in ['PM25', 'SO2', 'NO2', 'O3']:
        print(species, end = ', ')
        for month in np.arange(1,13):
            print(month, end = ', ')
            #data selected for date
            data = poll_ds.sel(model_name = model)[f'{species}'].groupby('time.month').mean().sel(month = month)
            
            #new lat and lon in radians
            lats_new = EPA_obs_df.loc[(EPA_obs_df['species'] == species)]['Latitude'].unique()
            lons_new = EPA_obs_df.loc[(EPA_obs_df['species'] == species)]['Longitude'].unique()
            
            #interpolation function
            interp_data = []
            for idx in range(lats_new.size):
                interp_data.append(data.sel(lat=lats_new[idx], lon=lons_new[idx], method='nearest').values.item())
            tmp_df = pd.DataFrame({'Arithmetic Mean':interp_data, 'Longitude':lons_new, 'Latitude':lats_new, 'model_name': model, 'species': species, 'date': month})
            interp_EPA_df = interp_EPA_df.append(tmp_df, sort=False, ignore_index=True)

for i in range(0, len(interp_EPA_df)):
    interp_EPA_df.loc[i,('date')] = datetime.datetime(2016,interp_EPA_df['date'][i],calendar.monthrange(2016,interp_EPA_df['date'][i])[1])
    
### interpolate data for IMPROVE
interp_IMPROVE_df = pd.DataFrame(columns=['Arithmetic Mean', 'Longitude', 'Latitude','model_name','species','date'])

for model in ['egrid_NA', 'nei_NA', 'epa_NA', 'normal_NA']:
    print(model, end = ', ')
    for species in ['PM25','NIT','SO4']:
        print(species, end = ', ')
        for month in np.arange(1,13):
            print(month, end = ', ')
            #data selected for date
            data = poll_ds.sel(model_name = model)[f'{species}'].groupby('time.month').mean().sel(month = month)
            
            #new lat and lon in radians
            if species == 'NH3': #interpolating NH3 to get it for our ISORROPIA total NH4 and NH3
                lats_new = IMPROVE_df.loc[(IMPROVE_df['species'] == 'NH4')]['Latitude'].unique()
                lons_new = IMPROVE_df.loc[(IMPROVE_df['species'] == 'NH4')]['Longitude'].unique()
            if species == 'HNO3': #interpolating NH3 to get it for our ISORROPIA total HNO3 and NIT
                lats_new = IMPROVE_df.loc[(IMPROVE_df['species'] == 'NIT')]['Latitude'].unique()
                lons_new = IMPROVE_df.loc[(IMPROVE_df['species'] == 'NIT')]['Longitude'].unique()
            if species == 'TotalOC':
                lats_new = IMPROVE_df.loc[(IMPROVE_df['species'] == 'OC')]['Latitude'].unique()
                lons_new = IMPROVE_df.loc[(IMPROVE_df['species'] == 'OC')]['Longitude'].unique()
            else:
                lats_new = IMPROVE_df.loc[(IMPROVE_df['species'] == species)]['Latitude'].unique()
                lons_new = IMPROVE_df.loc[(IMPROVE_df['species'] == species)]['Longitude'].unique()
            #interpolation function
            interp_data = []
            for idx in range(lats_new.size):
                interp_data.append(data.sel(lat=lats_new[idx], lon=lons_new[idx], method='nearest').values.item())
            tmp_df = pd.DataFrame({'Arithmetic Mean':interp_data, 'Longitude':lons_new, 'Latitude':lats_new, 'model_name': model, 'species': species, 'date': month})
            interp_IMPROVE_df = interp_IMPROVE_df.append(tmp_df, sort=False, ignore_index=True)

            
for i in range(0, len(interp_IMPROVE_df)):
    interp_IMPROVE_df.loc[i,('date')] = datetime.datetime(2016,interp_IMPROVE_df['date'][i],calendar.monthrange(2016,interp_IMPROVE_df['date'][i])[1])

    
#make monthly

IMPROVE_df = IMPROVE_df.rename(columns = {'Date':'date'})

def create_monthly_obs_df(obs_df):
    
    #create the 'geometries' for each lat and lon
    gdf = geopandas.GeoDataFrame(
    obs_df, geometry=geopandas.points_from_xy(obs_df.Longitude, obs_df.Latitude))
    geometries = gdf['geometry'].apply(lambda x: x.wkt).values
    #add to the dataset
    obs_df['geometry'] = geometries
    obs_df.index = obs_df['date']

    #group by month and geometry
    monthly_obs_df = pd.DataFrame(columns = ['Arithmetic Mean','Latitude','Longitude', 'geometry','species', 'date'])
    geometry = geometries[0]
    for geometry in np.unique(np.unique(obs_df['geometry'])):
        for species in np.unique(obs_df['species'].values):
            lat = obs_df.loc[(obs_df['geometry'] == geometry) & (obs_df['species'] == species)].groupby(pd.Grouper(freq='M'))['Latitude'].first().values
            lon = obs_df.loc[(obs_df['geometry'] == geometry) & (obs_df['species'] == species)].groupby(pd.Grouper(freq='M'))['Longitude'].first().values
            data = obs_df.loc[(obs_df['geometry'] == geometry) & (obs_df['species'] == species)].groupby(pd.Grouper(freq='M'))['Arithmetic Mean'].mean()
            tmp_df = pd.DataFrame({'Arithmetic Mean': data.values, 'Latitude':lat, 'Longitude':lon, 
                                   'geometry':geometry, 'species': species, 'date': data.index})
            monthly_obs_df = monthly_obs_df.append(tmp_df, sort=False, ignore_index=True)
    return(monthly_obs_df)

monthly_IMPROVE_df = create_monthly_obs_df(IMPROVE_df)
monthly_EPA_df = create_monthly_obs_df(EPA_obs_df)

#add region to the dataframes based on lat_lon dictionary
for df in [monthly_EPA_df, EPA_obs_df, interp_EPA_df, IMPROVE_df, interp_IMPROVE_df,monthly_IMPROVE_df]: 
    df['Region'] = 'a'
    for region in ['SE_lat_lon', 'NW_lat_lon', 'NE_lat_lon', 'MW_lat_lon', 'SW_lat_lon']:
        df.loc[(df['Longitude'].between(utils.lat_lon_dict[region][0], utils.lat_lon_dict[region][1], inclusive = True)) & 
            (df['Latitude'].between(utils.lat_lon_dict[region][2], utils.lat_lon_dict[region][3], inclusive = True)), 'Region'] = region
        
#save out

interp_EPA_df.to_csv('./final_data/interp_EPA_df.csv', date_format='%Y%m%d', index=False)
EPA_obs_df.to_csv('./final_data/EPA_obs_df.csv', date_format='%Y%m%d', index=False)
monthly_EPA_df.to_csv('./final_data/EPA_monthly_obs_df.csv', date_format='%Y%m%d', index=False)
IMPROVE_df.to_csv('./final_data/IMPROVE_df.csv', date_format='%Y%m%d', index=False)
interp_IMPROVE_df.to_csv('./final_data/interp_IMPROVE_df.csv', date_format='%Y%m%d', index=False)
monthly_IMPROVE_df.to_csv('./final_data/IMPROVE_monthly_obs_df.csv', date_format='%Y%m%d', index=False)