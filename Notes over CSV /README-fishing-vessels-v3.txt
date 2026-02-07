Global AIS-based Fishing Vessels, version 3.0 (2012-2024)

Fishing vessel data is available in the following formats:
 - BigQuery Tables (global-fishing-watch.fishing_effort_v3.fishing_vessels_v3)
 - CSV

Description:

This dataset includes information for all Maritime Mobile Service Identity (MMSI) - the vessel 
identifier in the automatic identification system (AIS) - that are included in version 3.0 
(March 2025 release) of the Global Fishing Watch (GFW) fishing effort data. These MMSI were 
identified as fishing vessels by the GFW vessel classification neural network model, (2) were 
not identified as non-fishing vessels by registries and manual review, and (3) met minimum activity 
and data quality requirements. If an MMSI was matched to a fishing vessel on a registry, but the 
neural net did not classify it as a fishing vessel, it is not included in this list. The dataset 
provides numerous metadata fields related to a vessel’s flag state, gear type, characteristics, 
and activity as obtained from the AIS data and vessel registries.

The dataset generally includes multiple fields for each vessel attribute, with one field per information 
source (usually the GFW machine learning models and vessel registries) and an additional field indicating
 the value used by GFW, which is derived from the available information. For example, for vessel class 
 (gear type), there are three fields: vessel_class_inferred (the vessel class as inferred by GFW machine 
 learning models), vessel_class_registry (the vessel class as listed on vessel registries), and vessel_class_gfw 
 (the vessel class assigned by GFW after taking into account both the machine learning model output and the 
 registry information). See the full schema below for a complete list of characteristics. The vessel_class_gfw 
 and flag_gfw fields are used for grouping MMSI in the fishing effort by fleet dataset.

The fishing-vessels-v3 dataset is provided in long format, with a separate row for each year, in order to provide 
summary information for each MMSI’s activity (hours, fishing_hours) by year. However, the vessel characteristic fields 
do not vary between years in the vast majority of cases, as GFW’s vessel identity algorithms return one classification 
per MMSI. (Exceptions occur for ~0.05% of MMSI in which adjustments have been applied after the initial classification 
occurred, and vessel_class_gfw does have different values in different years for this small subset of vessels.) 

The information in the fishing-vessels-v3 dataset can be used to filter the data in the mmsi-daily-10-v3 dataset 
to MMSI with specific characteristics, allowing users to analyze fishing effort by different categories of vessels. 

Flag state: 

Information about a vessel’s flag state can come from the MMSI MID (the first 3 digits of the vessel’s MMSI, 
which encode a flag state and is listed in the flag_ais field) and/or from registry information (flag_registry). When 
available, these two pieces of information are used to assign the MMSI a flag in the flag_gfw field. . However, vessel 
operators can manually input their MMSI so the MID (as part of the MMSI) may be incorrect. For MMSI without registry 
information and for which the MID is invalid (either the MMSI is the wrong number of digits so no MID can be determined, 
or the MMSI is the correct number of digits but the MID does not correspond to a flag state), we assign the flag_gfw 
field a value of UNKNOWN; if the vessel spends >50% of its time in a single EEZ, we assign the flag_gfw field a value 
of UNKNOWN-[ISO], where [ISO] corresponds to the ISO3 code of the state to which the EEZ corresponds.

Gear type:

The current version of the GFW vessel classification neural net model classifies fishing vessels into sixteen categories. 
Gear types with nested categories are higher order classes that are assigned when the confidence (vessel_class_inferred_score) 
in any one of the lower level atomic classes is not high enough to assign that class. For example, the model may not score 
a vessel high enough (>0.5) to label it “pots_and_traps”, “set_longlines”, or “set_gillnets”, but collectively these classes 
score high enough for the model to label the vessel as “fixed_gear”. The neural net assigns a "fishing" class to vessels when 
it is unsure about the type of fishing vessel. Both information from the neural net model’s classification and from listings 
on registries are used to assign the vessel_class_gfw field. For cases in which the neural net and registry information 
conflict, if the neural net has predicted a subclass of the registry class, the neural net class is assigned. (For example, 
for a vessel classified as “purse_seines” based on registry information but as “tuna_purse_seines” by the neural net, a class 
of “tuna_purse_seines” is assigned to the vessel_class_gfw field.) Otherwise, the lowest order class for which the neural net 
and registry both agree is assigned to vessel_class_gfw. (For example, for a vessel classified as “set_longlines” based on 
registry information but “set_gillnets” by the neural net, a class of “fixed_gear” will be assigned to vessel_class_gfw.)

- fishing: fishing vessels that could not be narrowed down to a more specific gear type
 - drifting_longlines: drifting longlines
 - seiners: vessels using seine nets, including potential purse seine vessels
   targeting tuna and other species, as well as danish and other seines
 	- purse_seines: purse seines, both pelagic and demersal
    		- tuna_purse_seines: large purse seines primarily fishing for tuna.
    		- other_purse_seines: purse seiners fishing for mackerel, anchovies, etc, often smaller and operating nearer 
            the coast than tuna purse seines.
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

Table Schema:

- mmsi: Maritime Mobile Service Identity, the identifier for AIS
- year: Year 
- flag_ais: Flag state (ISO3 value) for the vessel as determined by the first three digits (MID) of the MMSI number
- flag_registry: Flag state (ISO3 value) for the vessel as listed on vessel registries (when applicable)
- flag_gfw: Flag state (ISO3 value) assigned to the vessel by GFW after considering all available information
- vessel_class_inferred: Vessel class (gear type) inferred by the GFW vessel classification neural net model
- vessel_class_inferred_score: Neural net score (0-1) for the top scoring vessel class (gear type) inferred by the GFW vessel 
classification neural net model. Values closer to 1 indicate higher confidence by the neural net
- vessel_class_registry: Vessel class (gear type) for the vessel as listed on vessel registries (if applicable)
- vessel_class_gfw: Vessel class (gear type) assigned to the vessel by GFW after considering all available information
- self_reported_fishing_vessel: Whether the vessel broadcasts the 'Fishing' ship type in > 98% of AIS identity messages
- length_m_inferred: Vessel length (meters) inferred by the GFW vessel classification neural net model
- length_m_registry: Vessel length (meters) for the vessel as listed on vessel registries (if applicable)
- length_m_gfw: Vessel length (meters) assigned to the vessel by GFW after considering all available information
- engine_power_kw_inferred: Engine power (kilowatts) inferred by the GFW vessel classification neural net model
- engine_power_kw_registry: Engine power (kilowatts) for the vessel as listed on vessel registries (if applicable)
- engine_power_kw_gfw: Engine power (kilowatts) assigned to the vessel by GFW after considering all available information
- tonnage_gt_inferred: Tonnage (gross tons) inferred by the GFW vessel classification neural net model
- tonnage_gt_registry: Tonnage (gross tons) for the vessel as listed on vessel registries
- tonnage_gt_gfw: Tonnage (gross tons) assigned to the vessel by GFW after considering all available information
- registries_listed: Registries where the vessel is listed and used to inform the _registry fields (if applicable)
- active_hours: Hours the vessel was broadcasting AIS and moving faster than 0.1 knots
- fishing_hours: Hours the vessel was broadcasting AIS and detected as fishing by the GFW fishing detection neural net model 

Recommended citation:
Global Fishing Watch. 2025. Global AIS-based Apparent Fishing Effort Dataset, Version 3.0. https://doi.org/10.5281/zenodo.14982712

For additional information, see the journal article associated with the creation of version 1 (2018 release) of this dataset: 
[D.A. Kroodsma, J. Mayorga, T. Hochberg, N.A. Miller, K. Boerder, F. Ferretti, A. Wilson, B. Bergman, T.D. White, B.A. Block, P. Woods, B. Sullivan, C. Costello, and B. Worm. "Tracking the global footprint of fisheries." Science 361.6378 (2018)](https://science.sciencemag.org/content/359/6378/904).

Copyright Global Fishing Watch. Non-Commercial Use Only. The Site and the Services are provided for Non-Commercial use only in accordance with the CC BY-NC 4.0 license. If you would like to use the Site and/or the Services for commercial purposes, please contact us at support@globalfishingwatch.org . See also our Terms of Use (https://globalfishingwatch.org/terms-of-use/)




