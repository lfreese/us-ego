import matplotlib.pyplot as plt
import cartopy.feature as cfeat
import cartopy.crs as ccrs
import numpy as np

def concentration_plot(ds, species, model_names, rows, 
                       columns, figsize, levels,
                       season, cmap,shrink_cbar,
                       lat_lon, extension = 'max'):
    fig = plt.figure(figsize=figsize)
    for idx_m, model in enumerate(model_names):
            ####### plot our first day of NOx ######
            ax = fig.add_subplot(rows,columns,idx_m+1, projection=ccrs.PlateCarree())
        
            #make the plot
            ds[f'{model}_{species}'].groupby('time.season').mean(dim = 'time').sel(season = season).plot(ax=ax, #set the axis
                                    levels = np.squeeze(levels), #set the levels for our colorbars
                                   extend=extension,#extend the colorbar in both directions
                                   transform=ccrs.PlateCarree(), #fit data into map
                                   cbar_kwargs={'label':ds[f'{model}_{species}'].attrs['units'],'shrink':shrink_cbar}, #label our colorbar
                                    cmap=cmap)  #choose color for our colorbar
            
            ax.add_feature(cfeat.STATES)
            ax.coastlines() #add coastlines
            ax.set_extent(lat_lon) #set a limit on the plot lat and lon
            plt.title(f'{model} {species}'); #title
    #plt.suptitle(species_dict[species], fontsize = 24)
    #plt.subplots_adjust(bottom=0, top=.5)
def ratio_plot(ds, species, model_names, rows, 
                       columns, figsize, levels,
                       season, cmap,shrink_cbar,
                       lat_lon, extension = 'max'):
    fig = plt.figure(figsize=figsize)
    for idx_m, model in enumerate(model_names):
            ####### plot our first day of NOx ######
            ax = fig.add_subplot(rows,columns,idx_m+1, projection=ccrs.PlateCarree())
        
            #make the plot
            ds[f'{model}_{species}'].groupby('time.season').mean(dim = 'time').sel(season = season).plot(ax=ax, #set the axis
                                    levels = np.squeeze(levels), #set the levels for our colorbars
                                   extend=extension,#extend the colorbar in both directions
                                   transform=ccrs.PlateCarree(), #fit data into map
                                   cbar_kwargs={'label':ds[f'{model}_{species}'].attrs['units'],'shrink':shrink_cbar,'spacing':'uniform'}, #label our colorbar
                                    cmap=cmap)  #choose color for our colorbar
            
            ax.add_feature(cfeat.STATES)
            ax.coastlines() #add coastlines
            ax.set_extent(lat_lon) #set a limit on the plot lat and lon
            plt.title(f'{model} {species}'); #title
    #plt.suptitle(species_dict[species], fontsize = 24)
    #plt.subplots_adjust(bottom=0, top=.5)
    
    
#define a plot for observations and model
def obs_model_plot(ds, df, species,model_names, 
                   vmin, vmax, rows, columns, cmap, figsize, month,
                   lat_lon, lat_spacing=47,lon_spacing=73 
                   ):
    fig = plt.figure(figsize=figsize)
    species_dict = {'PM25':'PM2.5 - Local Conditions', 'SO2':'Sulfur dioxide', 'NO2':'Nitrogen dioxide (NO2)', 'O3':'Ozone', 'NOx':'Nitrogen Oxides (NO2+NO)'}

    for idx, model in enumerate(model_names):
        ###### Create axes ######
        ax=fig.add_subplot(rows,columns, idx +1, projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeat.STATES)
        
        ####### GEOS-CHEM output #######
        #PCM parameters and plot for model
        PCM_m=ax.pcolormesh(ds['lon'], ds['lat'], ds.groupby('time.month').mean(dim = 'time').sel(month = month)[f'{model}_{species}'], 
                            cmap=cmap,vmin=vmin, vmax=vmax)
    
        ###### observations #######
        #create lat and lon for observations
        lat_o = df.loc[df['species'] == species_dict[species]]['Latitude'].unique()
        lon_o = df.loc[df['species'] == species_dict[species]]['Longitude'].unique()
        #define the concentrations for observations
        mean_conc=df.loc[(df['species'] == species_dict[species])].groupby(['Latitude', 'Longitude']).mean()['Arithmetic Mean']
        #PCM parameters and plot for observations
        PCM_o=ax.scatter(lon_o, lat_o, c=mean_conc, transform=ccrs.PlateCarree(),cmap=cmap,edgecolors='k',linewidth=.3,vmin=vmin, vmax=vmax)
        plt.colorbar(PCM_o, ax=ax,extend='max', shrink=.3) 

        ###### adjustments and labels ########
        #adjust lat&lon being mapped
        ax.set_extent(lat_lon)
        plt.title(f'{model} {species}'); #title
        
def EPA_interp_dif_plot(df, species, model_names, difference_parameter,
                       vmin, vmax, rows, columns, cmap, figsize, 
                        lat_lon):
    fig = plt.figure(figsize=figsize)
    species_dict = {'PM25':'PM2.5 - Local Conditions', 'SO2':'Sulfur dioxide', 'NO2':'Nitrogen dioxide (NO2)', 'O3':'Ozone', 'NOx':'Nitrogen Oxides (NO2+NO)'}
    for idx, model in enumerate(model_names):
        ###### Create axes ######
        ax=fig.add_subplot(rows,columns, idx +1, projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cfeat.STATES)
        
        ##### select lon and lat #####
        lat=df['Latitude'].loc[
            (df['model'] == model) 
            & (df['species'] == species_dict[species])
        ].unique()
        lon=df['Longitude'].loc[
            (df['model'] == model) 
            & (df['species'] == species_dict[species])
        ].unique()
        
        ##### Define the Concentrations #####
        conc = df.loc[
            (df['model'] == model) 
            & (df['species'] == species_dict[species])
        ].groupby(['Latitude','Longitude']).mean()[difference_parameter]

        ##### Colorbar Parameters #####
        PCM=ax.scatter(lon, lat,c=conc,transform=ccrs.PlateCarree(),
                       cmap=cmap,edgecolors='k', linewidth = .3, vmin = vmin, vmax = vmax)
        plt.colorbar(PCM, ax=ax,extend='both') 
        
        ##### Adjustments and Labels #####
        ax.set_extent(lat_lon)
        plt.title(f'{difference_parameter} {species}')

        
def loc_mean_plot(df, species_list, model_names, subset, 
                  percent_dif):
    species_dict = {'PM25':'PM2.5 - Local Conditions', 'SO2':'Sulfur dioxide', 'NO2':'Nitrogen dioxide (NO2)', 'O3':'Ozone', 'NOx':'Nitrogen Oxides (NO2+NO)'}
    for species in species_list:
        fig = plt.figure(figsize = [20,5])
        for model in model_names:
            #plot anything greater than our percent_dif we input
            if subset == True:
                plt.plot(df.loc[
                    np.abs(df['GC-EPA Daily Mean Percent Difference']) > percent_dif
                ].loc[species_dict[species],model].index.values,
                         df.loc[
                             np.abs(df['GC-EPA Daily Mean Percent Difference']) > percent_dif
                         ].loc[species_dict[species],model]['GC-EPA Daily Mean Percent Difference'], 'o',
                        label = {model})
                plt.xticks(rotation = 45, fontsize = 8);
            #plot all percent differences
            else:
                plt.plot(df.loc[species_dict[species],model].index.values,
                     df.loc[species_dict[species],model]['GC-EPA Daily Mean Percent Difference'], '.',
                    label = {model})
                plt.xticks(rotation = 45, fontsize = 3);
            plt.xlabel('Station')
            plt.ylabel('% Difference')
            plt.title(f'{species} Average % Difference between Models and EPA Observations')