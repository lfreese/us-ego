import xarray as xr
from scipy import stats
import pandas as pd

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

def import_GC_runs_general(ds_name, path, aerosol_output, speciesconc_output, name):
    ds_speciesconc = xr.open_mfdataset(path+speciesconc_output, combine='by_coords')
    ds_aerosolmass = xr.open_mfdataset(path+aerosol_output,combine='by_coords')
    ds_name = xr.merge([ds_aerosolmass, ds_speciesconc])
    ds_name.attrs['name'] = name
    return (ds_name)

def import_GC_runs(egrid_path, NEI_path, MODEL_path, aerosol_output, speciesconc_output):
    ds_egrid_speciesconc = xr.open_mfdataset(egrid_path+speciesconc_output, combine='by_coords')
    ds_egrid_aerosolmass = xr.open_mfdataset(egrid_path+aerosol_output,combine='by_coords')
    ds_egrid = xr.merge([ds_egrid_aerosolmass, ds_egrid_speciesconc])
    ds_egrid.attrs['name'] = 'egrid'

    ds_NEI_speciesconc = xr.open_mfdataset(NEI_path+speciesconc_output,combine='by_coords')
    ds_NEI_aerosolmass = xr.open_mfdataset(NEI_path+aerosol_output, combine = 'by_coords')
    ds_NEI = xr.merge([ds_NEI_aerosolmass, ds_NEI_speciesconc])
    ds_NEI.attrs['name'] = 'NEI'

    ds_MODEL_speciesconc = xr.open_mfdataset(MODEL_path+speciesconc_output,combine='by_coords')
    ds_MODEL_aerosolmass = xr.open_mfdataset(MODEL_path+aerosol_output,combine='by_coords')
    ds_MODEL = xr.merge([ds_MODEL_aerosolmass, ds_MODEL_speciesconc])
    ds_MODEL.attrs['name'] = 'MODEL'
    
    return(ds_egrid, ds_NEI, ds_MODEL)



def linregress_data(obs_df, interp_df, model_names, month_string):
    result = [stats.linregress(
            x = obs_df.loc[
            (obs_df['species'] == species_dict[species])
            ].groupby(['Latitude','Longitude']).mean()['Arithmetic Mean'],
            y = interp_df.loc[
            (interp_df['model'] == model) & 
            (interp_df['species'] == species_dict[species])
            ].groupby(['Latitude','Longitude']).mean()['Arithmetic Mean']
            ) for species in ['PM25', 'SO2', 'NO2', 'O3'] for model in model_names]

    lin_regress_df = pd.merge(pd.DataFrame([(species,model) for species in ['PM25', 'SO2', 'NO2', 'O3'] for model in model_names], columns = ('species','model')),
             pd.DataFrame(result), 
             right_index = True, 
             left_index = True)
    return lin_regress_df


def interp_obs_differences(EPA_obs_df, interp_df, month_string, model_names):
    EPA_interp_dif = {}
    for idx_s, species in enumerate(['PM25', 'SO2', 'NO2', 'O3']):
        EPA_interp_dif[species] = {}
        for idx_m, model in enumerate(model_names):
            EPA_interp_dif[species][model] = (
                EPA_obs_df.loc[EPA_obs_df['species'] == species_dict[species]].groupby([pd.Grouper('Latitude'), pd.Grouper('Longitude')]).mean()['Arithmetic Mean'] -
                interp_df.loc[(interp_df['model'] == model) & (interp_df['species'] == species_dict[species])].groupby(['Latitude','Longitude']).mean()['Arithmetic Mean']            )
    return(EPA_interp_dif)

def ppb_to_ug(ds, species_to_convert, mw_species_list, stp_p = 101325, stp_t = 298.):
    R = 8.314 #J/K/mol
    ppb_ugm3 = (stp_p * 1e6 / stp_t / R) #Pa/K/(J/K/mol) = g/m^3

    for spec in species_to_convert:
        attrs = ds[spec].attrs
        ds[spec] = ds[spec]*mw_species_list[spec]*ppb_ugm3 #ppb*g/mol*g/m^3
        attrs['units'] = 'μg m-3'
        ds[spec].attrs.update(attrs)
        
def open_ISORROPIA(DJF_path, JJA_path, region_name):
    DJF_ds = xr.open_dataset(DJF_path)
    JJA_ds = xr.open_dataset(JJA_path)
    ds_out = xr.concat([DJF_ds, JJA_ds], pd.Index(['DJF', 'JJA'], name='season'))
    ds_out.attrs['region_name'] = region_name
    return(ds_out)