import matplotlib.pyplot as plt
import cartopy.feature as cfeat
import cartopy.crs as ccrs
import numpy as np
import utils
from matplotlib.lines import Line2D
from matplotlib import cm

proper_names_dict = {'PM25':r'PM$_{2.5}$ ($\mu g/m^3$)', 'NOx':r'NO$_x$ (ppbv)', 'SO2':r'SO$_2$ (ppbv)','O3':r'O$_3$ (ppbv)', 'NIT':r'Nitrate $(\mu g/m^3)$', 'NO2':r'NO$_2$ (ppbv)','SO4':r'SO$_4\ (\mu g/m^3)$','NH3':'Ammonia','NH4':'Ammonium'}
proper_model_names_dict = {'nonuc_NA':'No Nuclear','normal_NA': 'Base','egrid_NA':'eGRID','epa_NA':'NEI 2016','nei_NA':'NEI 2011', 'nonuc_coal_NA':'No Nuclear \n or Coal', 'EPA':'AQS','IMPROVE':'IMPROVE'}
###set color for each type
nonuc_color = 'C1'
normal_color = 'C0'
egrid_color = 'C7'
nocoal_color = 'c'
nox_lim_color = 'C5'
renew_color = 'C3'

cmap_dif = 'PRGn_r'
cmap_conc = 'PuBu'
cmap_discrete = cm.get_cmap('Paired', 3) 


lat_lon = [-120,-70,20,50]

import sys
sys.path.append('/model_validation')
species_dict = utils.species_dict

levels_dict = {'PM25':np.arange(0., 20., .5), 'SO2':np.arange(0., 3., .1), 
               'NO2':np.arange(0., 5., .1), 'NOx':np.arange(0., 5., .1), 'O3':np.arange(0., 80., 1.),
               'dif':np.arange(-1., 1.01, .01), 'regional_dif':np.arange(-1.5, 1.51, .01), 'regional_dif_tight':np.arange(-.3, .31, .01),
              'percent_dif_full':np.arange(-100, 101, 1), 'percent_dif_tight':np.arange(-10,10.1,.1),'NH3':np.arange(0,3,.1),'NH4':np.arange(0,3,.1)}



def concentration_plot_annual(ds, species_names, model_names, rows, 
                       columns, figsize, levels,
                     cmap,shrink_cbar,
                       lat_lon, extension = 'max'):
    fig, axes =  plt.subplots(len(species_names), len(model_names), figsize=figsize,subplot_kw={'projection':ccrs.LambertConformal()})
    for idx_m, model in enumerate(model_names):
        for idx_s, species in enumerate(species_names):
            ax = axes[idx_s,idx_m]        
            #make the plot
            ds[f'{species}'].sel(model_name = model).mean(dim = 'time').plot(ax=ax, #set the axis
                                   levels = np.squeeze(levels_dict[species]), #set the levels for our colorbars
                                   extend=extension,#extend the colorbar in both directions
                                   transform=ccrs.PlateCarree(), #fit data into map
                                   cbar_kwargs={'label':ds.sel(model_name = model)[f'{species}'].attrs['units'],'shrink':shrink_cbar}, #label our colorbar
                                    cmap=cmap)  #choose color for our colorbar
            
            ax.add_feature(cfeat.STATES)
            ax.coastlines() #add coastlines
            ax.set_extent(lat_lon) #set a limit on the plot lat and lon
            ax.set_title(f'{model} {species}', fontsize = 16); #title
    plt.tight_layout()

def concentration_plot_seasonal(ds_seasonal, species_names, season, model_names, figsize,
                       cmap,shrink_cbar,
                       lat_lon, extension = 'max'):
    fig, axes =  plt.subplots(len(species_names), len(model_names), figsize=figsize,subplot_kw={'projection':ccrs.LambertConformal()})
    for idx_m, model in enumerate(model_names):
        for idx_s, species in enumerate(species_names):
            ax = axes[idx_s,idx_m]        
            #make the plot
            ds_seasonal[f'{species}'].sel(model_name = model, season = season).plot(ax=ax, #set the axis
                                   levels = np.squeeze(levels_dict[species]), #set the levels for our colorbars
                                   extend=extension,#extend the colorbar in both directions
                                   transform=ccrs.PlateCarree(), #fit data into map
                                   cbar_kwargs={'label':ds_seasonal.sel(model_name = model)[f'{species}'].attrs['units'],'shrink':shrink_cbar}, #label our colorbar
                                    cmap=cmap)  #choose color for our colorbar
            
            ax.add_feature(cfeat.STATES)
            ax.coastlines() #add coastlines
            ax.set_extent(lat_lon) #set a limit on the plot lat and lon
            ax.set_title(f'{proper_model_names_dict[model]} {species}', fontsize = 16); #title

def concentration_plot_seasonal_dif(ds_seasonal, species_names, seasons, mod_base, mod_delta, rows, 
                       columns, figsize, levels,
                     cmap,
                       lat_lon, extension = 'both'):
    fig, axes =  plt.subplots(len(species_names), len(seasons), figsize=figsize,subplot_kw={'projection':ccrs.LambertConformal()})
    for idx_seas, season in enumerate(seasons):
        for idx_spec, species in enumerate(species_names):

            ax = axes[idx_spec, idx_seas]

        #make the plot
            q = (ds_seasonal[f'{species}'].sel(season = season, model_name = mod_delta)-ds_seasonal[f'{species}'].sel(season = season, model_name = mod_base)).plot(ax=ax, #set the axis
                                       levels = np.squeeze(levels), #set the levels for our colorbars
                                       extend=extension,#extend the colorbar in both directions
                                       transform=ccrs.PlateCarree(), #fit data into map
                                        cmap=cmap, add_colorbar = False)  #choose color for our colorbar

            ax.add_feature(cfeat.STATES)
            ax.coastlines() #add coastlines
            ax.set_extent(lat_lon) #set a limit on the plot lat and lon
            ax.set_title(''); #title
    for idx_spec, species in enumerate(species_names):
        axes[idx_spec, 0].annotate(f'{proper_names_dict[species]}', xy=(-.1, 0.2), xycoords = 'axes fraction', fontsize = 14, rotation = 90)
    axes[0,0].set_title('JJA', fontsize = 14)
    axes[0,1].set_title('DJF', fontsize = 14)
    fig.subplots_adjust(right=0.8)
    # put colorbar at desire position
    cbar_ax = fig.add_axes([0.2, 0.06, 0.5, 0.03]) # [left, bottom, width, height]
    fig.colorbar(q, cax=cbar_ax, orientation="horizontal")

    
def concentration_plot_monthly_dif(ds, species_names, times, rows, 
                       columns, figsize, levels,
                     cmap,
                       lat_lon, extension = 'both'):
    fig, axes =  plt.subplots(len(species_names), len(months), figsize=figsize,subplot_kw={'projection':ccrs.LambertConformal()})
    for idx_mon, month in enumerate(months):
        for idx_spec, species in enumerate(species_names):

            ax = axes[idx_spec, idx_mon]

        #make the plot
            q = ds[f'dif_nonuc_model-normal_model_{species}'].sel(time = time).plot(ax=ax, #set the axis
                                       levels = np.squeeze(levels), #set the levels for our colorbars
                                       extend=extension,#extend the colorbar in both directions
                                       transform=ccrs.PlateCarree(), #fit data into map
                                        cmap=cmap, add_colorbar = False)  #choose color for our colorbar

            ax.add_feature(cfeat.STATES)
            ax.coastlines() #add coastlines
            ax.set_extent(lat_lon) #set a limit on the plot lat and lon
            ax.set_title(''); #title
    for idx_spec, species in enumerate(species_names):
        axes[idx_spec, 0].annotate(f'{proper_names_dict[species]}', xy=(-.1, 0.2), xycoords = 'axes fraction', fontsize = 14, rotation = 90)
    #axes[0,0].set_title('JJA', fontsize = 14)
    #axes[0,1].set_title('DJF', fontsize = 14)
    fig.subplots_adjust(right=0.8)
    # put colorbar at desire position
    cbar_ax = fig.add_axes([0.2, 0.06, 0.5, 0.03]) # [left, bottom, width, height]
    fig.colorbar(q, cax=cbar_ax, orientation="horizontal")
    
def concentration_plot_seasonal_dif_models(ds_seasonal, species_name, seasons, mod_base, mod_deltas, rows, 
                       columns, figsize, levels,
                     cmap,
                       lat_lon, extension = 'both'):
    fig, axes =  plt.subplots(len(mod_deltas), len(seasons), figsize=figsize,subplot_kw={'projection':ccrs.LambertConformal()})
    for idx_seas, season in enumerate(seasons):
        for idx_mod, mod in enumerate(mod_deltas):

            ax = axes[idx_mod, idx_seas]

        #make the plot
            q = (ds_seasonal[f'{species_name}'].sel(season = season, model_name = mod)-ds_seasonal[f'{species_name}'].sel(season = season, model_name = mod_base)).plot(ax=ax, #set the axis
                                       levels = np.squeeze(levels), #set the levels for our colorbars
                                       extend=extension,#extend the colorbar in both directions
                                       transform=ccrs.PlateCarree(), #fit data into map
                                        cmap=cmap, add_colorbar = False)  #choose color for our colorbar

            ax.add_feature(cfeat.STATES)
            ax.coastlines() #add coastlines
            ax.set_extent(lat_lon) #set a limit on the plot lat and lon
            ax.set_title(''); #title
    for idx_mod, mod in enumerate(mod_deltas):
        axes[idx_mod, 0].annotate(f'{proper_model_names_dict[mod]}', xy=(-.15, 0.2), xycoords = 'axes fraction', fontsize = 14, rotation = 90)
    axes[0,0].set_title(seasons[0], fontsize = 14)
    axes[0,1].set_title(seasons[1], fontsize = 14)
    fig.subplots_adjust(right=0.8)
    # put colorbar at desire position
    cbar_ax = fig.add_axes([0.2, 0.06, 0.5, 0.03]) # [left, bottom, width, height]
    fig.colorbar(q, cax=cbar_ax, orientation="horizontal")

def concentration_plot_dif_models(ds_seasonal, species_name, season, mod_base, mod_deltas, rows, 
                       columns, figsize, levels,
                     cmap,
                       lat_lon, extension = 'both'):
    fig, axes =  plt.subplots(len(mod_deltas), len(season), figsize=figsize,subplot_kw={'projection':ccrs.LambertConformal()})
    for idx_mod, mod in enumerate(mod_deltas):

        ax = axes[idx_mod]

    #make the plot
        q = (ds_seasonal[f'{species_name}'].sel(season = season, model_name = mod)-ds_seasonal[f'{species_name}'].sel(season = season, model_name = mod_base)).plot(ax=ax, #set the axis
                                   levels = np.squeeze(levels), #set the levels for our colorbars
                                   extend=extension,#extend the colorbar in both directions
                                   transform=ccrs.PlateCarree(), #fit data into map
                                    cmap=cmap, add_colorbar = False)  #choose color for our colorbar

        ax.add_feature(cfeat.STATES)
        ax.coastlines() #add coastlines
        ax.set_extent(lat_lon) #set a limit on the plot lat and lon
        ax.set_title(''); #title
    for idx_mod, mod in enumerate(mod_deltas):
        axes[idx_mod].annotate(f'{proper_model_names_dict[mod]}', xy=(-.15, 0.2), xycoords = 'axes fraction', fontsize = 14, rotation = 90)
    axes[0].set_title('JJA', fontsize = 14)
    fig.subplots_adjust(right=0.8)
    # put colorbar at desire position
    cbar_ax = fig.add_axes([0.2, 0.06, 0.5, 0.03]) # [left, bottom, width, height]
    fig.colorbar(q, cax=cbar_ax, orientation="horizontal")
    
def ratio_plot(ds_seasonal, seasons, species1, species2, model, rows, 
                       columns, figsize, levels,
                       cmap,shrink_cbar,
                       lat_lon, extension = 'max'):
    fig = plt.figure(figsize=figsize)
    for idx_s, season in enumerate(seasons):
            ax = fig.add_subplot(rows,columns,idx_s+1, projection=ccrs.LambertConformal())
        
            #make the plot
            q = (ds_seasonal[f'{species1}']/ds_seasonal[f'{species2}']).sel(model_name = model, season = season).plot(ax=ax, #set the axis
                                    levels = np.squeeze(levels), #set the levels for our colorbars
                                   extend=extension,#extend the colorbar in both directions
                                   transform=ccrs.PlateCarree(), #fit data into map
                                    cmap=cmap, add_colorbar = False)  #choose color for our colorbar
            
            ax.add_feature(cfeat.STATES)
            ax.coastlines() #add coastlines
            ax.set_extent(lat_lon) #set a limit on the plot lat and lon
            plt.title(f'{season}', fontsize = 16); #title
    
    fig.subplots_adjust(right=0.8)
    # put colorbar at desire position
    cbar_ax = fig.add_axes([0.2, 0.06, 0.5, 0.03]) # [left, bottom, width, height]
    fig.colorbar(q, cax=cbar_ax, orientation="horizontal")
    cbar_ax.set_xlabel(r'$\frac{CH_2O}{NO_2}$ ratio', fontsize = 14)
    
    
#define a plot for observations and model
def obs_model_plot(ds, df, species,model_names, 
                   vmin, vmax, rows, columns, cmap, figsize, month,
                   lat_lon, lat_spacing=47,lon_spacing=73 
                   ):
    fig = plt.figure(figsize=figsize)

    for idx, model in enumerate(model_names):
        ###### Create axes ######
        ax=fig.add_subplot(rows,columns, idx +1, projection=ccrs.LambertConformal())
        ax.coastlines()
        ax.add_feature(cfeat.STATES)
        
        ####### GEOS-CHEM output #######
        #PCM parameters and plot for model
        PCM_m=ax.pcolormesh(ds.sel(model_name = model).groupby('time.month').mean().sel(month = month)['lon'], ds.sel(model_name = model).groupby('time.month').mean().sel(month = month)['lat'], ds.sel(model_name = model).groupby('time.month').mean().sel(month = month)[f'{species}'], 
                            cmap=cmap,vmin=vmin, vmax=vmax)
    
        ###### observations #######
        #create lat and lon for observations
        lat_o = df.loc[(df['species'] == species) & (df.date.dt.month == month)]['Latitude'].unique()
        lon_o = df.loc[(df['species'] == species) & (df.date.dt.month == month)]['Longitude'].unique()
        #define the concentrations for observations
        mean_conc=df.loc[(df['species'] == species) & (df.date.dt.month == month)].groupby(['Latitude','Longitude']).mean()['Arithmetic Mean']
        #PCM parameters and plot for observations
        PCM_o=ax.scatter(lon_o, lat_o, c=mean_conc, transform=ccrs.PlateCarree(),cmap=cmap,edgecolors='k',linewidth=.3,vmin=vmin, vmax=vmax)
        plt.colorbar(PCM_o, ax=ax,extend='max', shrink=.3) 

        ###### adjustments and labels ########
        #adjust lat&lon being mapped
        ax.set_extent(lat_lon)
        plt.title(f'{model} {month} {species}'); #title
        
def obs_plot(df,species, month,
            vmin, vmax, cmap, figsize,
            lat_lon, lat_spacing=47,lon_spacing=73 
                   ):
    fig = plt.figure(figsize=figsize)

    ###### Create axes ######
    ax=fig.add_subplot(1,1,1, projection=ccrs.LambertConformal())
    ax.coastlines()
    ax.add_feature(cfeat.STATES)
        
    ###### observations #######
    #create lat and lon for observations
    lat_o = df.loc[(df['species'] == species) & (df.date.dt.month == month)]['Latitude'].unique()
    lon_o = df.loc[(df['species'] == species) & (df.date.dt.month == month)]['Longitude'].unique()
    #define the concentrations for observations
    mean_conc=df.loc[(df['species'] == species) & (df.date.dt.month == month)].groupby(['Latitude','Longitude']).mean()['Arithmetic Mean']
    #PCM parameters and plot for observations
    PCM_o=ax.scatter(lon_o, lat_o, c=mean_conc, transform=ccrs.PlateCarree(),cmap=cmap,edgecolors='k',linewidth=.3,vmin=vmin, vmax=vmax)
    plt.colorbar(PCM_o, ax=ax,extend='max', shrink=.3) 

    ###### adjustments and labels ########
    #adjust lat&lon being mapped
    ax.set_extent(lat_lon)
    plt.title(f'{month} {species}'); #title

        
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
            
def interp_scatterplot(interp_df, obs_df, lin_regress_df,species_list, model_names, month_string, colors_dict, rows,columns):
    fig, axes = plt.subplots(rows,columns,figsize = [9,12])
    for idx_s, species in enumerate(species_list):
        for idx_m, model in enumerate(model_names):
            ax = axes[idx_s,idx_m]
            x = obs_df.loc[(obs_df['species'] == species)].groupby(['Latitude','Longitude']).mean()['Arithmetic Mean']
            y = interp_df.loc[(interp_df['model_name'] == model) & (interp_df['species'] == species)].groupby(['Latitude','Longitude']).mean()['Arithmetic Mean']
            abline = lin_regress_df.loc[(lin_regress_df['model_name'] == model) & (lin_regress_df['species'] == species)]['slope'].values * x + lin_regress_df.loc[(lin_regress_df['model_name'] == model) & (lin_regress_df['species'] == species)]['intercept'].values
            ax.scatter(x, y, c = colors_dict[model], marker = '.')
            ax.plot(x, x + 0, 'xkcd:grey', label = '1:1 Line')
            #ax.plot(x,abline,'xkcd:almost black' , label = 'Linear Regression')
            ax.set_xlim([-1,x.max()+1])
            ax.set_ylim([-1,x.max()+1])
            r_val = np.round(lin_regress_df.loc[(lin_regress_df['species'] == species) & (lin_regress_df['model_name'] == model), 'rvalue'].values[0], 2)
            std_err = np.round(lin_regress_df.loc[(lin_regress_df['species'] == species) & (lin_regress_df['model_name'] == model), 'stderr'].values[0], 2)
            ax.set_title(f'{species} {proper_model_names_dict[model]} \n R-value: {r_val} \n Standard error: {std_err}')
    custom_lines = [Line2D([0], [0], color='xkcd:grey', lw=4),]
                #Line2D([0], [0], color='xkcd:almost black', lw=4)]
    fig.text(0.5, -0.01, 'Observational Annual Mean', ha='center', fontsize = 16)
    fig.text(-0.01, 0.5, 'GC Annual Mean', va='center', rotation='vertical', fontsize = 16)

    fig.legend(custom_lines, ['1:1 Line'], loc = 'lower right')
    plt.tight_layout()

    
def hist_obs_interp(df, model_names, colors_dict, bins, species_list, rows, columns):
    fig, axes = plt.subplots(rows,columns,figsize = [12,12])
    for idx_s, species in enumerate(species_list):
        for idx_m, model in enumerate(model_names):
            ax = axes[idx_s,idx_m]
            n, bins, patches = ax.hist(df[species][model], bins, color = colors_dict[model])
            ax.set_title(f'{species} {proper_model_names_dict[model]}  Observations - \n Interpolated GEOS-Chem Data')
    plt.tight_layout()
    

def monthly_mean(ds, species, lat_lon, cmap):
    fig = plt.figure(figsize=[24,18])
    for idx, m in enumerate(range(1,13)):
        ax = fig.add_subplot(4,3,idx+1,projection=ccrs.LambertConformal())
        ds[f'{species}'].groupby('time.month').mean().sel(month = m).plot(ax=ax, #set the axis
                                        #levels = np.squeeze(levels), #set the levels for our colorbars
                                       extend='max',#extend the colorbar in both directions
                                       transform=ccrs.PlateCarree(), #fit data into map
                                       #cbar_kwargs={'label':ds[f'{species}'].attrs['units'],'shrink':.8}, #label our colorbar
                                        cmap=cmap)  #choose color for our colorbar

        ax.add_feature(cfeat.STATES)
        ax.coastlines() #add coastlines
        ax.set_extent(lat_lon) #set a limit on the plot lat and lon)
        plt.title(f'{species} in {m}')

def seasonal_mean(ds, species, lat_lon, cmap):
    fig = plt.figure(figsize=[24,18])
    for idx, s in enumerate(['DJF','JJA']):#,'MAM','SON']):
        ax = fig.add_subplot(4,3,idx+1,projection=ccrs.LambertConformal())
        ds[f'{species}'].groupby('time.season').mean().sel(season = s).plot(ax=ax, #set the axis
                                        #levels = np.squeeze(levels), #set the levels for our colorbars
                                       extend='max',#extend the colorbar in both directions
                                       transform=ccrs.PlateCarree(), #fit data into map
                                       #cbar_kwargs={'label':ds[f'{species}'].attrs['units'],'shrink':.8}, #label our colorbar
                                        cmap=cmap)  #choose color for our colorbar

        ax.add_feature(cfeat.STATES)
        ax.coastlines() #add coastlines
        ax.set_extent(lat_lon) #set a limit on the plot lat and lon)
        plt.title(f'{species} in {s}')
        
        
def plot_emissions(ds, emission, season, levels, lat_lon, figsize = [12,9]):
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(projection=ccrs.LambertConformal())
    ds[emission].groupby('time.season').mean().sel(season = season).plot(ax=ax, #set the axis
                                           levels = np.squeeze(levels), #set the levels for our colorbars
                                           extend='max',#extend the colorbar in both directions
                                          transform=ccrs.PlateCarree(), #fit data into map
                                           cbar_kwargs={'label':ds[f'{emission}'].attrs['units'],'shrink':.8}, #label our colorbar
                                            cmap='pink_r')  #choose color for our colorbar

    ax.add_feature(cfeat.STATES)
    ax.coastlines() #add coastlines
    ax.set_extent(lat_lon) #set a limit on the plot lat and lon)

def plot_emissions_dif(ds1, ds2, emissions, seasons, levels, lat_lon, figsize):
    fig, axes = plt.subplots(len(seasons), len(emissions), figsize = figsize, subplot_kw={'projection':ccrs.LambertConformal()})
    for idx_spec, emission in enumerate(emissions):
        for idx_season, season_val in enumerate(seasons):
            ax = axes[idx_season, idx_spec]
            q = (ds1[emission].groupby('time.season').mean().sel(season = season_val) - ds2[emission].groupby('time.season').mean().sel(season = season_val)).plot(
                                                ax=ax, #set the axis
                                                   levels = np.squeeze(levels), #set the levels for our colorbars
                                                   extend='both',#extend the colorbar in both directions
                                                  transform=ccrs.PlateCarree(), #fit data into map
                                                add_colorbar = False,
                                                    cmap='BrBG_r' #choose color for our colorbar
            )  
            ax.set_title(f'')
            ax.add_feature(cfeat.STATES)
            ax.coastlines() #add coastlines
            ax.set_extent(lat_lon) #set a limit on the plot lat and lon)
    axes[1,0].set_title(r'$NO_x$', fontsize = 14)
    axes[1,1].set_title(r'$SO_2$', fontsize = 14)
    pad = 5
    axes[0,0].annotate('JJA', xy=(0.07, 0.65), xycoords = 'figure fraction', fontsize = 14)
    axes[0,1].annotate('DJF', xy=(0.07, 0.25), xycoords = 'figure fraction', fontsize = 14)
    fig.subplots_adjust(right=0.8)
    # put colorbar at desire position
    cbar_ax = fig.add_axes([0.2, 0.06, 0.5, 0.03]) # [left, bottom, width, height]
    fig.colorbar(q, cax=cbar_ax, orientation="horizontal")
    cbar_ax.set_xlabel(r'$\frac{kg}{m^2s}$', fontsize = 14)
    
def plot_percent_emissions_dif(ds_seasonal, m1, m2, emissions, seasons, levels, lat_lon, figsize = [16,5]):
    fig, axes = plt.subplots(len(seasons), len(emissions), figsize = figsize, subplot_kw={'projection':ccrs.LambertConformal()})
    for idx_spec, emission in enumerate(emissions):
        for idx_season, season_val in enumerate(seasons):
            ax = axes[idx_season, idx_spec]
            q = (((ds_seasonal[emission].sel(model_name = m2, season = season_val) - ds_seasonal[emission].sel(model_name =m1, season = season_val))/(ds_seasonal[emission].sel(model_name =m1, season = season_val)))*100).plot(
                                                ax=ax, #set the axis
                                                   levels = np.squeeze(levels), #set the levels for our colorbars
                                                   extend='both',#extend the colorbar in both directions
                                                  transform=ccrs.PlateCarree(), #fit data into map
                                                add_colorbar = False,
                                                    cmap='BrBG_r' #choose color for our colorbar
            )  
            ax.set_title(f'')
            ax.add_feature(cfeat.STATES)
            ax.coastlines() #add coastlines
            ax.set_extent(lat_lon) #set a limit on the plot lat and lon)
    axes[1,0].set_title(r'$NO_x$', fontsize = 14)
    axes[1,1].set_title(r'$SO_2$', fontsize = 14)
    pad = 5
    axes[0,0].annotate('JJA', xy=(0.06, 0.65), xycoords = 'figure fraction', fontsize = 14)
    axes[0,1].annotate('DJF', xy=(0.06, 0.25), xycoords = 'figure fraction', fontsize = 14)
    fig.subplots_adjust(right=0.8)
    # put colorbar at desire position
    cbar_ax = fig.add_axes([0.2, 0.06, 0.5, 0.03]) # [left, bottom, width, height]
    fig.colorbar(q, cax=cbar_ax, orientation="horizontal")
    cbar_ax.set_xlabel('% change', fontsize = 14)
    
def plant_region_plot(ds, xvariable, yvariable1, egrid, egrid_yvariable, figsize, nocoal,  noxlim = True, normal = True, nonuc = True, renew = True):
    fig, ax = plt.subplots(figsize=figsize)
    width = 0.3
    if noxlim == True:
        plt.bar(ds.sel(model_name = 'nox_lim_model')[xvariable], ds.sel(model_name = 'nox_lim_model')[yvariable1], color = nox_lim_color, width = width, align="edge", label = 'NOx Emissions Limited Model')
    if nocoal == True:
        plt.bar(ds.sel(model_name = 'nonuc_nocoal_model')[xvariable], ds.sel(model_name = 'nonuc_nocoal_model')[yvariable1], color = nocoal_color, width = width, align="center", label = 'No Coal or Nuclear Model')
    if nonuc == True:
        plt.bar(ds.sel(model_name = 'nonuc_model')[xvariable], ds.sel(model_name = 'nonuc_model')[yvariable1], color = nonuc_color, width = width, align="edge", label = 'No Nuclear')
    if egrid == True:
        plt.bar(ds.sel(model_name = 'normal_model')[xvariable], ds.sel(model_name = 'normal_model')[egrid_yvariable], color = egrid_color, width = -width, align="edge", label = 'Egrid')
    if normal == True:
        plt.bar(ds.sel(model_name = 'normal_model')[xvariable], ds.sel(model_name = 'normal_model')[yvariable1], color = normal_color, width = width, align="center", label = 'Base Model')
    if renew == True:
        plt.bar(ds.sel(model_name = 'renewables_model')[xvariable], ds.sel(model_name = 'renewables_model')[yvariable1], color = renew_color, width = width, align="edge", label = 'Renewables Model')

    plt.xticks(rotation = 45)
    ax.legend();
    
def fossil_fuel_plot(ds, sci_names, xvariable, pollutants, figsize, nonuc_color, normal_color,egrid_color, nocoal, egrid, noxlim = True, normal = True, renew = True):
    fig,axes = plt.subplots(2, 2, figsize=figsize, sharex = True)
    for ax, pollutant in zip(axes.flatten(), pollutants):
        width = 0.3
        ax.bar(ds.sel(model_name = 'nonuc_model').sel(fueltype = ['Coal', 'NaturalGas'])[xvariable], ds.sel(model_name = 'nonuc_model').sel(fueltype = ['Coal', 'NaturalGas'])[f'model_annual_{pollutant}_conc']/1000, color = nonuc_color, width = width, align="edge", label = 'No Nuclear')
        if nocoal == True:
            ax.bar(ds.sel(model_name = 'nonuc_nocoal_model').sel(fueltype = ['Coal', 'NaturalGas'])[xvariable], ds.sel(model_name = 'nonuc_nocoal_model').sel(fueltype = ['Coal', 'NaturalGas'])[f'model_annual_{pollutant}_conc']/1000, color = nocoal_color, width = width, align="center", label = 'No Nuclear or Coal Model')
        if normal == True:
            ax.bar(ds.sel(model_name = 'normal_model').sel(fueltype = ['Coal', 'NaturalGas'])[xvariable], ds.sel(model_name = 'normal_model').sel(fueltype = ['Coal', 'NaturalGas'])[f'model_annual_{pollutant}_conc']/1000, color = normal_color, width = width, align="center", label = 'Base Model')
        if egrid == True:
            ax.bar(ds.sel(model_name = 'normal_model').sel(fueltype = ['Coal', 'NaturalGas'])[xvariable], ds.sel(model_name = 'normal_model').sel(fueltype = ['Coal', 'NaturalGas'])[f'egrid_annual_{pollutant}_conc']/1000, color = egrid_color, width = -width, align="edge", label = 'Egrid')
        if noxlim == True:
            ax.bar(ds.sel(model_name = 'nox_lim_model').sel(fueltype = ['Coal', 'NaturalGas'])[xvariable], ds.sel(model_name = 'nox_lim_model').sel(fueltype = ['Coal', 'NaturalGas'])[f'model_annual_{pollutant}_conc']/1000, color = nox_lim_color, width = -width, align="edge", label = 'NOx Limited Model')
        if renew == True:
            ax.bar(ds.sel(model_name = 'renewables_model').sel(fueltype = ['Coal', 'NaturalGas'])[xvariable], ds.sel(model_name = 'renewables_model').sel(fueltype = ['Coal', 'NaturalGas'])[f'model_annual_{pollutant}_conc']/1000, color = renew_color, width = -width, align="center", label = 'NOx Limited Model')
        ax.set_title(f'{sci_names[pollutant]}', fontsize = 20);
        ax.set_title(f'{sci_names[pollutant]}', fontsize = 20);
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    fig.text(0.35, -0.01, f'Fuel Type', fontsize = 14, ha='center')
    fig.text(-0.01, 0.38, f'Emissions (metric tons)', fontsize = 14, ha='center', rotation='vertical')
    custom_lines = [Line2D([0], [0], color=nonuc_color, lw=4),
                    Line2D([0], [0], color=normal_color, lw=4),
                   Line2D([0], [0], color=nocoal_color, lw=4),
                   Line2D([0], [0], color=nox_lim_color, lw=4),
                   Line2D([0], [0], color=renew_color, lw=4),
                   Line2D([0], [0], color=egrid_color, lw=4)]
    plt.legend(custom_lines, ['No Nuclear', 'Base Model', 'No Nuclear or Coal Model', 'NOx Limited Model', 'Renewables Model', 'Egrid Model'], bbox_to_anchor=(.95,.5),  bbox_transform=fig.transFigure)
    plt.tight_layout()
        
def isorropia_obs_model_plot(cdf, ds_isorropia, vmin, vmax, spacing, figsize = [8,20]):
    fig, axes = plt.subplots(5, 2, figsize=figsize)
    season_dict = {'DJF':[12,1,2],'JJA':[6,7,8]}
    region_dict = {'NE_lat_lon':'NE', 'SW_lat_lon':'SW', 'MW_lat_lon':'MW', 'SE_lat_lon':'SE',
           'NW_lat_lon':'NW'}
    for idx_s, season in enumerate(['JJA','DJF']):
        for idx_r, region in enumerate(region_dict.keys()):
            ax = axes[idx_r, idx_s]
            #plot isorropia data
            q = ds_isorropia.sel(region_name = region_dict[region], season = season).PM.plot(ax = ax, cmap = 'BrBG_r', levels = np.arange(vmin,vmax,spacing), extend = 'max')

            #plot model
            x = cdf.loc[(cdf['species'] == 'NH4') & (cdf['Region'] == region) & (cdf['model_name'] == 'normal_model') & (cdf['date'].dt.month.isin(season_dict[season]))]['Arithmetic Mean']
            y = cdf.loc[(cdf['species'] == 'NIT') & (cdf['Region'] == region) & (cdf['model_name'] == 'normal_model') & (cdf['date'].dt.month.isin(season_dict[season]))]['Arithmetic Mean']
            z = cdf.loc[(cdf['species'] == 'PM25') & (cdf['Region'] == region) & (cdf['model_name'] == 'normal_model') & (cdf['date'].dt.month.isin(season_dict[season]))]['Arithmetic Mean']

            ax.scatter(x, y, c = 'C1', cmap = 'Accent', marker = ">", vmin = vmin, vmax = vmax, label = 'Model');
            
            #plot observations
            x2 = cdf.loc[(cdf['species'] == 'NH4') & (cdf['Region'] == region) & (cdf['model_name'] == 'IMPROVE') & (cdf['date'].dt.month.isin(season_dict[season]))]['Arithmetic Mean']
            y2 = cdf.loc[(cdf['species'] == 'NIT') & (cdf['Region'] == region) & (cdf['model_name'] == 'IMPROVE') & (cdf['date'].dt.month.isin(season_dict[season]))]['Arithmetic Mean']
            z2 = cdf.loc[(cdf['species'] == 'inorganic_PM') & (cdf['Region'] == region) & (cdf['model_name'] == 'IMPROVE') & (cdf['date'].dt.month.isin(season_dict[season]))]['Arithmetic Mean']
            ax.scatter(x2, y2, c = 'C7', cmap = 'Accent',marker = 'v', vmin = vmin, vmax = vmax, label = 'IMPROVE Observations');
            
            #plot nonuc
           # x = cdf.loc[(cdf['species'] == 'NH4') & (cdf['Region'] == region) & (cdf['model'] == 'nonuc_model') & (cdf['date'].dt.month.isin(season_dict[season]))]['Arithmetic Mean']
           # y = cdf.loc[(cdf['species'] == 'NIT') & (cdf['Region'] == region) & (cdf['model'] == 'nonuc_model') & (cdf['date'].dt.month.isin(season_dict[season]))]['Arithmetic Mean']
           # z = cdf.loc[(cdf['species'] == 'PM25') & (cdf['Region'] == region) & (cdf['model'] == 'nonuc_model') & (cdf['date'].dt.month.isin(season_dict[season]))]['Arithmetic Mean']

            #ax.scatter(x, y, c = 'C0', cmap = 'Accent', marker = '<', vmin = vmin, vmax = vmax, label = 'Model');
            #adjust x and y labels and limits
            ax.set_xlabel(r'Total $NH_{3} (\mu g/m^3)$');
            ax.set_ylabel('Total $NO_{3} (\mu g/m^3)$');
            ax.set_xlim([0,6])
            ax.set_ylim([0,14])
            ax.set_title(' ')
    for idx_s, season in enumerate(['JJA','DJF']):
        axes[0,idx_s].set_title(f'{season}', fontsize = 20, pad = 15)
    for idx_r, region in enumerate(region_dict.keys()):
        axes[idx_r, 0].annotate(f'{region_dict[region]}', xy=(-0.35, 0.8), xycoords = 'axes fraction', fontsize = 20)
    fig.subplots_adjust(right=0.8)
    # put colorbar at desire position
    cbar_ax = fig.add_axes([0.2, -0.01, 0.5, 0.01]) # [left, bottom, width, height]
    fig.colorbar(q, cax=cbar_ax, orientation="horizontal")
    cbar_ax.set_xlabel(r'$PM_{2.5} (\mu g/m^3)$', fontsize = 14)
    #make custom legend
    plt.scatter([], [], c='C1', marker = '>',
                    label= 'Model')
    plt.scatter([], [], c='C7', marker = 'v',
                    label= 'IMPROVE')
   # plt.scatter([], [], c='C0', marker = '<',
       #             label= 'No Nuclear')
    plt.legend(bbox_to_anchor=(1.4, 2))
    #layout
    plt.tight_layout()

def scatter_nitrate_plots(figsize, poll_ds, season, x_species, y_species):
    fig,ax = plt.subplots(figsize = figsize)
    x = poll_ds.groupby('time.season').mean().sel(season = season, model_name = 'nonuc_model')[x_species]
    y = poll_ds.groupby('time.season').mean().sel(season = season, model_name = 'nonuc_model')[y_species]
    plt.plot(x,y, 'C0.', label = 'No Nuclear')
    x1 = poll_ds.groupby('time.season').mean().sel(season = season, model_name = 'normal_model')[x_species]
    y1 = poll_ds.groupby('time.season').mean().sel(season = season, model_name = 'normal_model')[y_species]
    plt.plot(x1,y1, 'C1.', label = 'normal_model')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel(f'{x_species} {poll_ds[x_species].units}', fontsize = 16)
    plt.ylabel(f'{y_species} {poll_ds[y_species].units}', fontsize = 16)