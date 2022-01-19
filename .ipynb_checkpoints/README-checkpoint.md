# us-ego
## Optimization_model:

Warning: the input csv files are not included here. They are stored in our current Dropbox folder on Guillaume's server (https://www.dropbox.com/sh/edxgc11umqw4y3b/AABlT1BPtHBfTTVNdsR5dA5da?dl=0)

Run modify_cost_inputs, which loads Alan's input data from ./good_model_inputs/ and adjusts the costs and capacities. This gives you a new generation and transmission file.

Run check_feasibility.ipynb on the new generation and transmission files if you want to check whether or not there is a time where demand exceeds generation capacity+import capacity.

Launch your terminal and create a directory called 'outputs'

Run ./launch.sh, which will run the optimization (sliced into 8 different runs)

To edit the limit for time and CPUs for the run, edit runopt.sh. This is what launches runopt.jl, your optimization script, and is launched by "sbatch runopt.sh t1 t2" where t1 is the first hour of the simulation and t2 is the last one
To edit the time slices for the run, edit launch.sh
Your run will take anywhere from 1-2 days, after which you can look at the output
Edit the input files for runopt.jl (based on the output from 1)
Run combined.ipynb to merge and format the outputs

Validate your output against eGrid generation and emissions with postproc_opt.ipynb, which takes your combined outputs

## ego_nonuclear

Comparing the no nuclear, normal, and egrid cases

## model_validation

Validation notebooks including:


NEI_validation.ipynb for a monthly validation to compare to NEI_2016 (data from ftp://newftp.epa.gov/DMDnLoad/emissions/daily/quarterly/2016/)

To get the data in a format readable by GEOS-Chem, with access to a slurm run sbatch opt_output_to_inventory.py

Additional notes: test_run_opt.ipynb is the draft Julia optmization script

## Data Sources
Sources for all raw data are listed below. The input files are modified, as many have taken tables and turned them into CSV files. 
1. Cost data: https://www.eia.gov/electricity/data/eia860/ EIA form 860
2. Solar Renewable CF: https://www.epa.gov/airmarkets/power-sector-modeling-platform-v515 Table 4-28 
3. Wind Renewable CF: https://www.epa.gov/airmarkets/power-sector-modeling-platform-v515 Table 4-20
4. Load/Demand data: https://www.eia.gov/todayinenergy/detail.php?id=27212 EIA930_BALANCE_2016 form for both Jan-Jun and Jul-Dec
5. Capacity/Emissions factors: https://www.epa.gov/energy/emissions-generation-resource-integrated-database-egrid (historical data, 2016)
