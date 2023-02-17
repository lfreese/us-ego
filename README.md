# us-ego
## Optimization_model:

### All notes in readme inside the optimization_model folder as well 

Warning: the input csv files are not included here. They are stored in our current Dropbox folder on Guillaume's server (https://www.dropbox.com/sh/edxgc11umqw4y3b/AABlT1BPtHBfTTVNdsR5dA5da?dl=0)

1. Run modify_inputs, which loads Alan's input data from ./good_model_inputs/ and adjusts the costs and capacities. This gives you a new generation and transmission file.

2. Run check_feasibility.ipynb on the new generation and transmission files if you want to check whether or not there is a time where demand exceeds generation capacity+import capacity.

3. Launch your terminal and create a directory called 'outputs'

4. Run ./launch.sh, which will run the optimization (sliced into 8 different runs)
    - To edit the limit for time and CPUs for the run, edit runopt.sh. This is what launches runopt.jl, your optimization script, and is launched by "sbatch runopt.sh t1 t2" where t1 is the first hour of the simulation and t2 is the last one
    - To edit the time slices for the run, edit launch.sh
    - Your run will take anywhere from 1-2 days, after which you can look at the output
    - Edit the input files and output file names for runopt.jl (based on the output from 1) 

6. Run combined.ipynb to merge and format the outputs

7. Validate your output against eGrid generation and emissions with postproc_opt.ipynb, which takes your combined outputs

8. If you want to compare your data to NEI 2016 data, use NEI_validation.ipynb for a monthly validation (data from ftp://newftp.epa.gov/DMDnLoad/emissions/daily/quarterly/2016/)

9. To get the data in a format readable by GEOS-Chem, with access to a slurm run sbatch opt_output_to_inventory.py. You can then run remove_nans.ipynb in order to remove any nans that appear in the .nc files to make sure they work with GEOS-Chem. If you want methane data use opt_output_to_inventory-GHG_only.py, but a warning that the methane data is not reliable, and this is only to be used to show the gaps in methane data.



Additional notes:
1. mask_us_neigrid.pkl masks the US grid, and is used in the opt_output_to_inventory


## ego_nonuclear_project

Notebooks for the paper comparing no nuclear, base, no nuclear + no coal, no nuclear + renewables cases.

The beginning of each notebook or script has a number, most of this indicates the order in which they should likely be run. Data prep is labelled in the 0, energy modeling output evaluation is in 1-3, pollution/emissions output is 5-8, social costs 9, health impacts 10, systems analysis 11. Some specifics are discussed below:

0) Data preparation. 

The first step for data preparation/cleaning is to select individual variables and merge them, using the code below:

``` module load cdo
cd Outputdir
mkdir ../merged_data
mkdir ../merged_data/daily_mean
mkdir ../merged_data/hrly_summer_ozone
```
Ozone:
```
for file in GEOSChem.SpeciesConc.2016*; do cdo -selvar,SpeciesConc_O3 $file O3_$file; done && for month in {01,02,03,04,05,06,07,08,09,10,11,12}; do cdo mergetime O3*2016$month*.nc4 ../merged_data/merged_O3_$month.nc ; done
```
PM2.5
```
for file in GEOSChem.AerosolMass.2016*; do cdo -selvar,PM25 $file PM_$file; done && for month in {01,02,03,04,05,06,07,08,09,10,11,12}; do cdo mergetime PM*2016$month*.nc4 ../merged_data/merged_PM_$month.nc ; done
```
NO, NO2, CH2O, SO2:
```
for file in GEOSChem.SpeciesConc.2016*; do cdo -selvar,SpeciesConc_CH2O $file CH2O_$file; done && for month in {01,02,03,04,05,06,07,08,09,10,11,12}; do cdo mergetime CH2O*2016$month*.nc4 ../merged_data/merged_CH2O_$month.nc ; done && for file in GEOSChem.SpeciesConc.2016*; do cdo -selvar,SpeciesConc_NO $file NO_$file; done && for month in {01,02,03,04,05,06,07,08,09,10,11,12}; do cdo mergetime NO*2016$month*.nc4 ../merged_data/merged_NO_$month.nc ; done && for file in GEOSChem.SpeciesConc.2016*; do cdo -selvar,SpeciesConc_NO2 $file NO2_$file; done && for month in {01,02,03,04,05,06,07,08,09,10,11,12}; do cdo mergetime NO2*2016$month*.nc4 ../merged_data/merged_NO2_$month.nc ; done && for file in GEOSChem.SpeciesConc.2016*; do cdo -selvar,SpeciesConc_SO2 $file SO2_$file; done && for month in {01,02,03,04,05,06,07,08,09,10,11,12}; do cdo mergetime SO2*2016$month*.nc4 ../merged_data/merged_SO2_$month.nc ; done
```

Following this, you can run 001_dataset_creation-daily1 (for daily PM and ozone data), 002_dataset_creation-daily2 (for daily PM and ozone data), 003_dataset_creation-hourly (for hourly PM and ozone data), 004_dataset_creation-precursors (for NOx, SO2, CH2O), and then 005_dataset_creation-obs_comp (for comparing to observations from IMPROVE and EPA AQS). Following this, run the 0_dataset_creation_allcode notebook (for health attribution). 


## Data Sources
Sources for all raw data are listed below. The input files are modified, as many have taken tables and turned them into CSV files. 
1. Cost data: https://www.eia.gov/electricity/data/eia923/ EIA-923 with EIA-906/920 previous data 
2. Solar Renewable CF: https://www.epa.gov/airmarkets/power-sector-modeling-platform-v515 Table 4-28 
3. Wind Renewable CF: https://www.epa.gov/airmarkets/power-sector-modeling-platform-v515 Table 4-20
4. Load/Demand data: https://www.eia.gov/todayinenergy/detail.php?id=27212 EIA930_BALANCE_2016 form for both Jan-Jun and Jul-Dec, selecting download data, subregion data for 2016
5. Capacity/Emissions factors: https://www.epa.gov/energy/emissions-generation-resource-integrated-database-egrid (historical data, 2016)
6. Transmission Data and hourly wind and solar profiles are from: https://www.epa.gov/airmarkets/power-sector-modeling-platform-v515 NEEDS v5.16/IPM v5.16 
