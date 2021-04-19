import xarray as xr
from scipy import stats
import pandas as pd
import numpy as np
import glob
from datetime import datetime, timedelta 
import geopandas

lat_lon_dict = {
'US_lat_lon':[-130.0, -60.0, 24.0, 45.0],
'SE_lat_lon':[-90.,-75.,25.,38.],
'NW_lat_lon':[-125.,-109.,40.,50.],
'NE_lat_lon':[-90.,-65.,38.,50.],
'MW_lat_lon':[-110.,-89.,25.,50.],
'SW_lat_lon':[-125.,-109.,25.,40.]

}
month_string = ['01','02','03','04','05','06','07','08','09','10','11','12']
species_dict = {'PM25':'PM2.5 - Local Conditions', 'SO2':'Sulfur dioxide', 'NO2':'Nitrogen dioxide (NO2)', 'O3':'Ozone', 'NOx':'Nitrogen Oxides (NO2+NO)'}
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
    'H2O2'
]
#dict of all aerosol/PM contributing species and their mw
aerosol_species_dict = {
    'NH4': 18.039,
    'NIT': 62.0049,
    'SO4': 96.06
    #'BCPI': 12.,
    #'OCPI': 12.,
    #'BCPO': 12.,
    #'OCPO': 12.,
    #'DST1': 29.,
    #'DST2': 29.,
    #'SALA': 31.4,

}

def import_GC_runs_general(path, speciesconc_output, aerosol_output, name):
    '''Import and name a Geos Chem run based on the path and the aerosol and species concentration files'''
    ds_speciesconc = xr.open_mfdataset(path+speciesconc_output, combine='by_coords')
    ds_aerosolmass = xr.open_mfdataset(path+aerosol_output,combine='by_coords')
    ds_name = xr.merge([ds_aerosolmass, ds_speciesconc])
    ds_name.attrs['name'] = name
    return (ds_name)

def combine_and_convert_ds(gas_species_list, aerosol_species_list, datasets_to_combine, index_names, model_names_no_reference, reference_model_name, ds_final_name):
    '''Combine geos chem datasets, choose which species will be in the dataset, and convert mol/mol to ppbv
    Includes create a H2O2/HNO3 and CH2O/NO2 ratio
    returns a pollution dataset with all species listed in proper units'''
    #list of all names of species we will keep
    all_aerosol_species_list = list(aerosol_species_list) + ['PM25', 'TotalOC','TotalOA','AerMassBC']
    all_species = list(gas_species_list) + list(all_aerosol_species_list)
    
    #combine our two datasets into one, with model as an index
    ds = xr.concat(datasets_to_combine, pd.Index(index_names, name='model_name'))
    
    #sum our NO2 and NO to get NOx
    ds['SpeciesConc_NOx'] = (
        ds['SpeciesConc_NO'] + ds['SpeciesConc_NO2']
                   )
    
    #drop anything not in our list of species, and rename to drop the 'speciesconc' and 'aermass'
    poll_ds = ds.rename({'SpeciesConc_' + spec: spec for spec in gas_species_list})
    poll_ds = poll_ds.rename({'AerMass' + spec: spec for spec in aerosol_species_list})

    poll_ds = poll_ds.drop_vars([species for species in poll_ds.data_vars if species not in all_species])

    #convert all but PM to ppbv
    for species in gas_species_list:
        poll_ds[f'{species}'] *= 1e9 #convert from mol/mol to ppbv
        poll_ds[f'{species}'].attrs['units'] = 'ppbv'

    #calculate the NO2/CH2O ratio    
    poll_ds[f'CH2O_NO2'] = poll_ds['CH2O']/poll_ds['NO2']
    poll_ds[f'CH2O_NO2'].attrs['units'] = 'Ratio CH2O/NO2'

    #calculate the H2O2/HNO3 ratio    
    poll_ds[f'H2O2_HNO3'] = poll_ds['H2O2']/poll_ds['HNO3']
    poll_ds[f'H2O2_HNO3'].attrs['units'] = 'Ratio H2O2/HNO3'

    for new_model_name in model_names_no_reference:
        poll_ds = dif_between_models(poll_ds, new_model_name, reference_model_name, all_species)
    #select ground level
    poll_ds = poll_ds.isel(lev = 0)

    #name dataset
    poll_ds.attrs['name'] = ds_final_name
    return(poll_ds)

def dif_between_models(poll_ds, new_model_name, reference_model_name, species_list):
    
    for species in species_list:
        #calculate the differences for species
            poll_ds[f'dif_{new_model_name}-{reference_model_name}_{species}'] = poll_ds.sel(model_name = new_model_name)[f'{species}'] - poll_ds.sel(model_name = reference_model_name)[f'{species}']
            poll_ds[f'dif_{new_model_name}-{reference_model_name}_{species}'].attrs['units'] = poll_ds.sel(model_name = new_model_name)[f'{species}'].attrs['units']

            #calculate the percent differences for species
            poll_ds[f'percent_dif_{new_model_name}-{reference_model_name}_{species}'] = (poll_ds[f'dif_{new_model_name}-{reference_model_name}_{species}']/poll_ds.sel(model_name = reference_model_name)[f'{species}'])*100
            poll_ds[f'percent_dif_{new_model_name}-{reference_model_name}_{species}'].attrs['units'] = 'Percent Difference'
    return(poll_ds)

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
    
            
def linregress_data(obs_df, interp_df, model_names, month_string, species_list):
    result = [stats.linregress(
            x = obs_df.loc[
            (obs_df['species'] == species)
            ].groupby(['Latitude','Longitude']).mean()['Arithmetic Mean'],
            y = interp_df.loc[
            (interp_df['model'] == model) & 
            (interp_df['species'] == species)
            ].groupby(['Latitude','Longitude']).mean()['Arithmetic Mean']
            ) for species in species_list for model in model_names]

    lin_regress_df = pd.merge(pd.DataFrame([(species,model) for species in species_list for model in model_names], columns = ('species','model')),
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
                interp_df.loc[(interp_df['model'] == model) & (interp_df['species'] == species)].groupby(['Latitude','Longitude']).mean()['Arithmetic Mean']            )
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
