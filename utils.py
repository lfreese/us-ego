import xarray as xr
from scipy import stats
import pandas as pd
import numpy as np
import glob
from datetime import datetime, timedelta 
import geopandas
from math import radians, cos, sin, asin, sqrt


lat_lon_dict = {
'US_lat_lon':[-130.0, -60.0, 24.0, 45.0],
'SE_lat_lon':[-90.,-75.,25.,38.],
'NW_lat_lon':[-125.,-109.,40.,50.],
'NE_lat_lon':[-90.,-65.,38.,50.],
'MW_lat_lon':[-110.,-89.,25.,50.],
'SW_lat_lon':[-125.,-109.,25.,40.]

}
month_string = ['01','02','03','04','05','06','07','08','09','10','11','12']
species_dict = {'PM25':'PM2.5 - Local Conditions', 'SO2':'Sulfur dioxide', 'NO2':'Nitrogen dioxide (NO2)', 'O3':'Ozone', 'NOx':'Nitrogen Oxides (NO2+NO)', 'NH3':'Ammonia'}


#list of all gas species
gas_species_list = [ 
    'NO',
    'NO2',
    'SO2',
    'O3',
    'CH2O',
    'NOx',
    'NH3', 
    'NO3',
    'HNO3',
    'H2O2',
    #'OH'
]
#dict of all aerosol/PM contributing species and their mw
aerosol_species_dict = {
    'NH4': 18.039,
    'NIT': 62.0049,
    'SO4': 96.06,
    #'BCPI': 12.,
    #'OCPI': 12.,
    #'BCPO': 12.,
    #'OCPO': 12.,
    #'DST1': 29.,
    #'DST2': 29.,
    #'SALA': 31.4,

}

def import_GC_runs_general(path, output, preprocessor, species_list):
    '''Import and name a Geos Chem run based on the path and the aerosol and species concentration files'''
    ds = xr.open_mfdataset(path+output, preprocess = preprocessor)
    return (ds)

def combine_and_convert_PM_O3(O3_ds_combine, PM_ds_combine, O3_index_names,PM_index_names, ds_final_name):
    '''Combine geos chem datasets, and convert mol/mol to ppbv for O3 only'''

    #combine our two datasets into one, with model as an index
    dsO3 = xr.concat(O3_ds_combine, pd.Index(O3_index_names, name='model_name'))
    dsPM = xr.concat(PM_ds_combine, pd.Index(PM_index_names, name='model_name'))
    ds = xr.merge([dsO3['SpeciesConc_O3'], dsPM['PM25']])
    ds = ds.isel(lev = 0)

    ds[f'SpeciesConc_O3'] *= 1e9 #convert from mol/mol to ppbv
    ds[f'SpeciesConc_O3'].attrs['units'] = 'ppbv'
    ds = ds.rename({'SpeciesConc_O3':'O3'})

    #name dataset
    ds.attrs['name'] = ds_final_name
    return(ds)

def convert_gases(gas_ds, index_model_names, gas_name):
    '''Combine geos chem datasets, and convert mol/mol to ppbv for O3 only'''

    #combine our two datasets into one, with model as an index
    ds_gas = xr.concat(gas_ds, pd.Index(index_model_names, name='model_name'))
    ds_gas = ds_gas.isel(lev = 0)

    ds_gas[f'SpeciesConc_{gas_name}'] *= 1e9 #convert from mol/mol to ppbv
    ds_gas[f'SpeciesConc_{gas_name}'].attrs['units'] = 'ppbv'
    ds_gas = ds_gas.rename({f'SpeciesConc_{gas_name}':f'{gas_name}'})

    return(ds_gas)

def convert_aerosol(aerosol_ds, index_model_names, aerosol_name):
    '''Combine geos chem datasets, and convert mol/mol to ppbv for O3 only'''

    #combine our two datasets into one, with model as an index
    aerosol_ds = xr.concat(aerosol_ds, pd.Index(index_model_names, name='model_name'))
    aerosol_ds = aerosol_ds.isel(lev = 0)

    aerosol_ds[f'AerMass{aerosol_name}'].attrs['units'] = 'ug/m3'
    aerosol_ds = aerosol_ds.rename({f'AerMass{aerosol_name}':f'{aerosol_name}'})

    return(aerosol_ds)

def combine_egrid_generation(oris_ds, gen_ds, egrid_ds):   
    #create a capacity, fueltype, and regionname grouped by ORISCode
    capacity = oris_ds.groupby('ORISCode').sum()['Capacity']
    fueltype = oris_ds.to_dataframe().groupby('ORISCode').first()['FuelType']
    regionname = oris_ds.to_dataframe().groupby('ORISCode').first()['RegionName']
    #group by ORIS code and take the mean of everything but capacity
    oris_ds = oris_ds.groupby('ORISCode').mean().drop('Capacity')
    #rename our generation variable
    gen_ds = gen_ds.rename('modelgeneration')

    ###concatenate the generation and ORIS files
    gmodel_oris_ds = xr.merge([gen_ds, oris_ds])
    #add in the capacity
    gmodel_oris_ds['Capacity'] = capacity
    #create a column for capacity factors
    gmodel_oris_ds['model_capafactor'] = 100 * gmodel_oris_ds['modelgeneration'] / (gmodel_oris_ds['Capacity'] * 8760) # % generation for each year's total capacity
    
    ###concatenate our model/oris and egrid emissions dataframes into one, grouped by ORIS code
    gmodel_egrid_ds = xr.merge([gmodel_oris_ds, egrid_ds])
    #turn all zeroes (just in the modelgeneration) to NAN
    gmodel_egrid_ds.where('modelgeneration' == 0)['modelgeneration'] = np.nan
    #rename the egrid data column for ease
    gmodel_egrid_ds = gmodel_egrid_ds.rename({'Generator annual net generation (MWh)':'annual_egridgeneration'})
    #add in fueltype
    gmodel_egrid_ds['fueltype'] = fueltype
    gmodel_egrid_ds = gmodel_egrid_ds.set_coords('fueltype')
    #add in region name
    gmodel_egrid_ds['regionname'] = regionname
    gmodel_egrid_ds = gmodel_egrid_ds.set_coords('regionname')
    return(gmodel_egrid_ds)
    

    
def season_mean(ds, calendar="standard"):
    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = ds.time.dt.days_in_month

    # Calculate the weights by grouping by 'time.season'
    weights = (
        month_length.groupby("time.season") / month_length.groupby("time.season").sum()
    )

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))

    # Calculate the weighted average
    return (ds * weights).groupby("time.season").sum(dim="time")


#### function to find area of a grid cell from lat/lon ####
def find_area(ds, R = 6378.1):
    """ ds is the dataset, i is the number of longitudes to assess, j is the number of latitudes, and R is the radius of the earth in km. 
    Must have the ds['lat'] in descending order (90...-90)
    Returns Area of Grid cell in km"""
    
    dy = (ds['lat_b']- ds['lat_b'].roll({'lat_b':-1}, roll_coords = False))[:-1]*2*np.pi*R/360 

    dx1 = (ds['lon_b'].roll({'lon_b':-1}, roll_coords = False) - 
           ds['lon_b'])[:-1]*2*np.pi*R*np.cos(np.deg2rad(ds['lat_b']))
    
    dx2 = (ds['lon_b'].roll({'lon_b':-1}, roll_coords = False) - 
           ds['lon_b'])[:-1]*2*np.pi*R*np.cos(np.deg2rad(ds['lat_b'].roll({'lat_b':-1}, roll_coords = False)[:-1]))
    
    A = .5*(dx1+dx2)*dy
    
    #### assign new lat and lon coords based on the center of the grid box instead of edges ####
    A = A.assign_coords(lon_b = ds.lon.values,
                    lat_b = ds.lat.values)
    A = A.rename({'lon_b':'lon','lat_b':'lat'})

    A = A.transpose()
    
    return(A)


def make_2d_grid(lon_b1, lon_b2, lon_step, lat_b1, lat_b2, lat_step):
    lon_bounds = np.arange(lon_b1, lon_b2+lon_step, lon_step)
    lon_centers = (lon_bounds[:-1] + lon_bounds[1:])/2
    
    lat_bounds = np.arange(lat_b1, lat_b2+lat_step, lat_step)[::-1]
    lat_centers = (lat_bounds[:-1] + lat_bounds[1:])/2
    
    ds = xr.Dataset({'lat': (['lat'], lat_centers),
                     'lon': (['lon'], lon_centers),
                     'lat_b': (['lat_b'], lat_bounds),
                     'lon_b': (['lon_b'], lon_bounds),
                    }
                   )
    return(ds)



#bootstrap function    
def draw_bs(x, size):
    '''performs pairs bootstrap for a linear regression for ozone health impact assessment'''
    #Perform pairs bootstrap for linear regression

    # Set up array of indices to sample from: inds
    inds = np.arange(0,len(x))

    # Generate replicates
    bs_inds = np.random.choice(inds, size=size)
    bs_val= x[bs_inds]
        
    return bs_val

    
def linregress_data(obs_df, interp_df, model_names, month_string, species_list):
    result = [stats.linregress(
            x = obs_df.loc[
            (obs_df['species'] == species)
            ].groupby(['Latitude','Longitude']).mean()['Arithmetic Mean'],
            y = interp_df.loc[
            (interp_df['model_name'] == model) & 
            (interp_df['species'] == species)
            ].groupby(['Latitude','Longitude']).mean()['Arithmetic Mean']
            ) for species in species_list for model in model_names]

    lin_regress_df = pd.merge(pd.DataFrame([(species,model) for species in species_list for model in model_names], columns = ('species','model_name')),
             pd.DataFrame(result), 
             right_index = True, 
             left_index = True)
    return lin_regress_df


def interp_obs_differences(EPA_obs_df, interp_df, month_string, model_names, species_list):
    EPA_interp_dif = {}
    for idx_s, species in enumerate(species_list):
        EPA_interp_dif[species] = {}
        for idx_m, model in enumerate(model_names):
            EPA_interp_dif[species][model] = (
                EPA_obs_df.loc[EPA_obs_df['species'] == species].groupby([pd.Grouper('Latitude'), pd.Grouper('Longitude')]).mean()['Arithmetic Mean'] -
                interp_df.loc[(interp_df['model_name'] == model) & (interp_df['species'] == species)].groupby(['Latitude','Longitude']).mean()['Arithmetic Mean']            )
    return(EPA_interp_dif)


def ppb_to_ug(ds, species_to_convert, mw_species_list, stp_p = 101325, stp_t = 298.):
    '''Convert species to ug/m3 from ppb'''
    R = 8.314 #J/K/mol
    ppb_ugm3 = (stp_p * 1e6 / stp_t / R) #Pa/K/(J/K/mol) = g/m^3

    for spec in species_to_convert:
        attrs = ds[spec].attrs
        ds[spec] = ds[spec]*mw_species_list[spec]*ppb_ugm3 #ppb*g/mol*g/m^3
        ds[spec].attrs['units'] = 'μg m-3'
    return(ds)

def import_and_edit_EPAobs(path):
    '''Import EPA observational data and drop un-needed columns, round Latitude and Longitude, convert to datetime, select only one of the SO2 standards, convert ozone to ppb'''
    EPA_obs_df = pd.concat(map(pd.read_csv, glob.glob(path)))
    EPA_obs_df['date'] = pd.to_datetime(EPA_obs_df['Date Local'])
    EPA_obs_df['date'] = EPA_obs_df['date'].dt.normalize() + timedelta(hours=12)
    EPA_obs_df['Longitude'] = np.round(EPA_obs_df['Longitude'], decimals = 8)
    EPA_obs_df['Latitude'] = np.round(EPA_obs_df['Latitude'], decimals = 8)

    EPA_obs_df = EPA_obs_df.drop(columns = ['State Code','County Code','Site Num','Parameter Code','POC','Datum','Sample Duration','Date Local', 'Event Type',
           'Observation Count', 'Observation Percent','1st Max Value', '1st Max Hour', 'Address', 'County Name', 'City Name',
           'CBSA Name', 'Date of Last Change', 'Method Name'])

    #only use the SO2 1 hour 2010 pollutant standard arithmetic means and drop the 3-hour 1971 arithmetic means
    EPA_obs_df = EPA_obs_df.loc[~(EPA_obs_df['Pollutant Standard'] == 'SO2 3-hour 1971')]
    EPA_obs_df = EPA_obs_df.rename(columns = {'Parameter Name':'species'})

    #convert Ozone to ppb
    EPA_obs_df.loc[EPA_obs_df['species'] == 'Ozone','Arithmetic Mean'] *= 1e3 #ppb
    EPA_obs_df = EPA_obs_df.loc[~(EPA_obs_df['Arithmetic Mean'] <= 0)]
    
    #rename pollutants to shortened names
    for species in ['PM25', 'SO2', 'NO2', 'O3']:
        EPA_obs_df.loc[(EPA_obs_df['species'] == species_dict[species]), 'species'] = species
    
    #limit to just contiguous US (leaving out HI and AL)
    EPA_obs_df = EPA_obs_df.loc[(EPA_obs_df['Latitude'].between(24,50,inclusive = True)) & (EPA_obs_df['Longitude'].between(-130,-60,inclusive = True))]

    return(EPA_obs_df)
        
def import_IMPROVE(path, species, short_name):
    df = pd.read_fwf(path)
    df['species'] = species
    if species == 'NH4':
        df['Arithmetic Mean'] = df['ammNO3f:Value'] + df['ammSO4f:Value']
        df = df.rename(columns = {'ammNO3f:Unit': 'Unit'})
        df = df.drop(columns = {'NH4f:Value','NH4f:Unc','NH4f:Unit','ammNO3f:Value','ammNO3f:Unc','ammSO4f:Value', 'ammSO4f:Unit',
                                            'ammSO4f:Unc'})
    elif species == 'OC_EC':
        df['Arithmetic Mean'] = df['OCf:Value']
        df['Unit'] = df['OCf:Unit']
        df['species'] = 'OC_EC'
        df = df.drop(columns = {'ECf:Value','ECf:Unit','OCf:Value','OCf:Unit'})
    else:
        df = df.rename(columns = {f'{short_name}:Value':'Arithmetic Mean', f'{short_name}:Unc': 'Uncertainty', f'{short_name}:Unit': 'Unit'})
    return(df)
    
def open_ISORROPIA(season1, season2, season_names, region_name, two_seasons = True):
    season1_ds = xr.open_dataset(season1)
    if two_seasons == True:
        season2_ds = xr.open_dataset(season2)
        ds_out = xr.concat([season1_ds, season2_ds], pd.Index(season_names, name='season'))
    else:
        ds_out = season1_ds
    ds_out.attrs['region_name'] = region_name
    return(ds_out)

def haversine(lon1, lat1, lon2, lat2): #adjusted from https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    Calculate distance in km between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers, have to convert to miles when out
    return c * r

def grouped_weighted_avg(values, weights):
    return (values * weights).sum() / weights.sum()