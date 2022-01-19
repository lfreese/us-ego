#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%matplotlib inline

import pandas as pd
import numpy as np

import xarray as xr

import regionmask

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cartopy.feature as cfeat
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import feather, h5py, sys, pickle
from shapely.geometry import Point, Polygon
from geocube.api.core import make_geocube
import xagg as xa
import netCDF4
import geopandas

np.seterr(invalid='ignore'); # disable a warning from matplotlib and cartopy


# county_mean = pd.read_csv('data/county_mean.csv')

# In[2]:


poll_ds = xr.open_zarr('final_data/ds_PM_O3_daily.nc4')

print('imported')
# In[3]:


counties = geopandas.read_file('data/cb_2018_us_county_500k.shx')
counties = counties.rename(columns = {'NAME':'CountyName'})

print('imported')

# In[4]:


states = geopandas.read_file('data/cb_2018_us_state_500k.shx')
states = states.rename(columns = {'NAME':'StateName'})


# states_counties = states.merge(counties)

# In[5]:


counties = counties.reset_index()


# In[6]:


counties['index'] = counties['index'].astype('int32')


# In[7]:


counties['STATEFP'] = counties['STATEFP'].astype('int32')


# In[8]:


weightmap = xa.pixel_overlaps(poll_ds,counties)

print('weightmap complete')

# In[9]:


poll_ds = poll_ds.drop([
 'county',
 'state'])


# In[ ]:


aggregated= xa.aggregate(poll_ds,weightmap)
print('aggregate complete')

aggregated.to_netcdf('aggregated_counties.nc')
print('aggregate saved')
# In[ ]:


ds_out = aggregated.to_dataset()


# In[ ]:


xr.Dataset.to_zarr(ds_out, './data/GC_counties.zarr', mode = 'w') #save the dataset 

print('zarr saved')
