Global Daily AIS-based Apparent Fishing Effort by MMSI, version 3.0, 2012-2024

Data is available in the following formats:
 - CSVs (https://globalfishingwatch.org/data-download/datasets/public-fishing-effort)
 - BigQuery tables (global-fishing-watch.fishing_effort_v3.mmsi-daily-10-v3)

Description:
This dataset provides apparent fishing hours (the number of hours vessels were determined to have spent
fishing in the grid cell) and vessel presence hours (the number of hours vessels were determined to have 
spent in the grid cell) for fishing vessels based on data from the automatic identification system (AIS) 
for 2012-2024 and grouped by Maritime Mobile Service Identity (MMSI), the vessel identifier in AIS. Data 
are spatially binned into grid cells that measure 0.1 degrees on a side using decimal degrees (WGS84); 
the coordinates defining each cell correspond to the lower-left corner. Data are temporally aggregated by day. 

Data are based on AIS positions of >190,000 broadcast MMSI, of which up to ~96,000 are active in a given year. 
MMSI associated with fishing vessels are identified via a neural network classifier, vessel registry databases, 
and manual review by GFW and regional experts. Vessel time is measured in hours, calculated by assigning each 
AIS position the amount of time elapsed since the previous AIS position of the vessel. The time is counted as 
apparent fishing hours if a neural network classifier determines the vessel is engaged in fishing behavior during 
that AIS position. Apparent fishing hours for squid jiggers are not detected via the neural network, but instead 
through this heuristic (https://github.com/GlobalFishingWatch/global-footprint-of-fisheries/blob/master/data_production/updated-algorithm-for-squid-jiggers.md).

Vessel information for each MMSI, including flag state, gear type, and vessel dimensions is provided in a separate file (fishing-vessels-v3.csv).

Table Schema:
 - date: Date in YYYY-MM-DD format
 - cell_ll_lat: The latitude of the lower left (ll) corner of the grid cell, in decimal degrees
 - cell_ll_lon: The longitude of the lower left (ll) corner of the grid cell, in decimal degrees
 - mmsi: Maritime Mobile Service Identity, the identifier for AIS
 - hours: Hours that the MMSI was broadcasting on AIS while present in the grid cell on this day
 - fishing_hours: Hours that the MMSI was broadcasting on AIS in this grid cell on this day and detected as fishing by the GFW fishing detection model

Recommended citation:
Global Fishing Watch. 2025. Global AIS-based Apparent Fishing Effort Dataset, Version 3.0. https://doi.org/10.5281/zenodo.14982712

For additional information, see the journal article associated with the creation of version 1 (2018 release) of this dataset: D.A. Kroodsma, J. Mayorga, T. Hochberg, N.A. Miller, K. Boerder, F. Ferretti, A. Wilson, B. Bergman, T.D. White, B.A. Block, P. Woods, B. Sullivan, C. Costello, and B. Worm. "Tracking the global footprint of fisheries." Science 361.6378 (2018). (http://science.sciencemag.org/content/359/6378/904)

Copyright Global Fishing Watch. Non-Commercial Use Only. The Site and the Services are provided for Non-Commercial use only in accordance with the CC BY-NC 4.0 license. If you would like to use the Site and/or the Services for commercial purposes, please contact us at support@globalfishingwatch.org . See also our Terms of Use (https://globalfishingwatch.org/terms-of-use/).
