Global AIS-based Apparent Fishing Effort by Flag State and Gear Type, version 3.0, 2012-2024

Data is available in the following formats:
 - CSVs (https://globalfishingwatch.org/data-download/datasets/public-fishing-effort)
 - Public BigQuery tables (global-fishing-watch.fishing_effort_v3.fleet-daily-100-v3; 
 global-fishing-watch.fishing_effort_v3.fleet-monthly-10-v3)

Description:
This dataset provides apparent fishing hours (the number of hours vessels were determined to 
have spent fishing in the grid cell) and vessel presence hours (the number of hours vessels 
were determined to have spent in the grid cell) for fishing vessels based on data from the 
automatic identification system (AIS) for 2012-2024 and grouped by flag state and gear type. 

Data are provided in the following two formats: 

- fleet-daily-100-v3: Data are spatially binned into grid cells that measure 0.01 degrees on a 
side using decimal degrees (WGS84) and temporally aggregated by day

- fleet-monthly-10-v3: Data are spatially binned into grid cells that measure 0.1 degrees on a 
side using decimal degrees (WGS84) and temporally aggregated by month

Data are based on AIS positions of >190,000 broadcast Maritime Mobile Service Identity (MMSI) 
numbers - the vessel identifier in AIS - of which up to ~96,000 are active in a given year. 
MMSI associated with fishing vessels are identified via a neural network classifier, vessel 
registry databases, and manual review by GFW and regional experts. Vessel time is measured in 
hours, calculated by assigning to each AIS position the amount of time elapsed since the previous 
AIS position of the vessel. The time is counted as apparent fishing hours if a neural network 
classifier determines the vessel is engaged in fishing behavior during that AIS detection. 
Apparent fishing hours for squid jiggers are not detected via the neural network, but instead 
through this heuristic (https://github.com/GlobalFishingWatch/global-footprint-of-fisheries/blob/master/data_production/updated-algorithm-for-squid-jiggers.md).

Flag state: 

Information about a vessel’s flag state can come from the MID (the first 3 digits of the vessel’s 
MMSI, which encode a flag state), from registry information, or from both. Vessel operators can 
manually input their MMSI so the MID (as part of the MMSI) may be incorrect. For MMSI without 
registry information and for which the MID is invalid (either the MMSI is the wrong number of 
digits so no MID can be determined, or the MMSI is the correct number of digits but the MID does 
not correspond to a flag state), we list the vessel’s flag state as UNKNOWN; if the vessel spends 
>50% of its time in a single EEZ, we list its flag as UNKNOWN-[ISO], where [ISO] corresponds to the 
ISO3 code of the state to which the EEZ corresponds. The values for the flag field in this dataset 
are based on the flag_gfw field in the fishing-vessels-v3.csv file.

Gear type:

The current version of the GFW vessel classification neural net classifies fishing vessels into sixteen 
categories. Gear types with nested categories are higher order classes that are assigned when the neural 
net and/or vessel registries are not confident enough in one of the lower level atomic classes. 
For example, the model may not score a vessel high enough to label it “pots_and_traps”, “set_longlines”, 
or “set_gillnets”, but collectively these classes score high enough to label the vessel as “fixed_gear”. 
The "fishing" class is assigned to vessels for which the neural net is unsure about the type of fishing 
vessel. In cases in which the neural net and registry information conflict, if the neural net has predicted 
a subclass of the registry class, the neural net class is assigned. (For example, for a vessel classified as 
“purse_seines” based on registry information but as “tuna_purse_seines” by the neural net, a class of 
“tuna_purse_seines” is assigned.) Otherwise, the lowest order class for which the neural net and registry 
both agree is assigned. (For example, for a vessel classified as “set_longlines” based on registry information 
but “set_gillnets” by the neural net, a class of “fixed_gear” will be assigned.) The values for the geartype 
field in this dataset are based on the vessel_class_gfw field in the fishing-vessels-v3.csv file.

Geartypes:
- fishing: fishing vessels that could not be narrowed down to a more specific gear type
 - drifting_longlines: drifting longlines
 - seiners: vessels using seine nets, including potential purse seine vessels
   targeting tuna and other species, as well as danish and other seines
 	- purse_seines: purse seines, both pelagic and demersal
    	- tuna_purse_seines: large purse seines primarily fishing for tuna.
    	- other_purse_seines: purse seiners fishing for mackerel, anchovies, etc, 
        often smaller and operating nearer the coast than tuna purse seines.
	- other_seines: danish seines and other seiners not using purse seines.
 - trawlers: trawlers, all types
 - pole_and_line: vessel from which people fish with pole and line.
 - trollers: vessel that tows multiple fishing lines.
 - fixed_gear: a category that includes potential set longlines, set gillnets,  and pots and traps
 	- pots_and_traps: vessel that deploys pots (small, portable traps) or traps to
   	catch fish
 	- set_longlines: vessel that fishes by setting longlines anchored to the
   	seafloor. These lines have shorter hooked, typically baited, lines hanging
   	from them
 	- set_gillnets: vessel that fishes by setting gillnets anchored to the seafloor.
 - dredge_fishing: vessel that tows a dredge the scrapes up edible bottom
   dwellers such as scallops or oysters.
 - squid_jigger: squid jiggers, mostly large industrial pelagic operating vessels

Vessel information for each MMSI included in this dataset is provided in a separate file (fishing-vessels-v3.csv). 
The flag and geartype assignments are based on the flag_gfw and vessel_class_gfw fields.

Table Schema
- date: Date in YYYY-MM-DD format. For the fleet-monthly-10-v3 data, the date corresponds to the first date of the month
- year: year (fleet-monthly-10-v3 only)
- month: month (fleet-monthly-10-v3 only)
- cell_ll_lat: The latitude of the lower left (ll) corner of the grid cell, in decimal degrees (WGS84)
- cell_ll_lon: The longitude of the lower left (ll) corner of the grid cell, in decimal degrees (WGS84)
- flag: Flag state (ISO3 value), based on flag_gfw in fishing-vessels-v3.csv
- geartype: Gear type, based on vessel_class_gfw in fishing-vessels-v3.csv 
- hours: Hours that MMSI of this geartype and flag were broadcasting on AIS while present in the grid cell on this day
- fishing_hours: Hours that MMSI of this geartype and flag were broadcasting on AIS in this grid cell on this day and detected as fishing by the GFW fishing detection model
- mmsi_present: Number of MMSI of this flag state and geartype that broadcasted on AIS while present in the grid cell on this day

Recommended citation:
Global Fishing Watch. 2025. Global AIS-based Apparent Fishing Effort Dataset, Version 3.0. https://doi.org/10.5281/zenodo.14982712

For additional information, see the journal article associated with the creation of version 1 (2018 release) of this dataset: 
D.A. Kroodsma, J. Mayorga, T. Hochberg, N.A. Miller, K. Boerder, F. Ferretti, A. Wilson, B. Bergman, T.D. White, B.A. Block, P. Woods, B. Sullivan, C. Costello, and B. Worm. "Tracking the global footprint of fisheries." Science 361.6378 (2018). (http://science.sciencemag.org/content/359/6378/904)

Copyright Global Fishing Watch. Non-Commercial Use Only. The Site and the Services are provided for Non-Commercial
use only in accordance with the CC BY-NC 4.0 license. If you would like to use the Site and/or the Services for commercial
purposes, please contact us at support@globalfishingwatch.org . See also our Terms of Use (https://globalfishingwatch.org/terms-of-use/)



