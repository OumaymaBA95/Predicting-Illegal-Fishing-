Known issues with version 3.0 of the Global Fishing Watch AIS-based apparent fishing effort dataset.


The GFW algorithms attempt to classify and map the activity of hundreds of thousands of vessels using AIS data. 
AIS was not designed as a tool to track and monitor the activities of fishing fleets, and as a result there are 
a number of data challenges. Additionally, GFW informs its analyses with known information from vessel registries, 
which may be incorrect at times. As such, there are multiple known issues with the results of our global fishing 
effort dataset that stem from these and other challenges. This outlines and provides context to these issues. 
If you see an issue with our dataset,  we welcome feedback via research@globalfishingwatch.org. For other 
descriptions of the known issues with AIS, we recommend referring to Kroodsma et al 2018 
(https://science.sciencemag.org/content/359/6378/904.abstract; in particular, the supplementary materials) 
and Chapters 1-2 of the The Global Atlas of AIS-based Fishing Activity 
(https://globalfishingwatch.org/fisheries/fao-atlas/).

Recent classifications may change with future data: Data from 2024 are provisional and may appear different in 
future releases as more data from 2025 becomes available. Because our vessel classification algorithms look both forward 
and backward in time, vessel classification for 2024 may change non-trivially as more data for these vessels becomes available. 

Identity spoofing: In theory, each Maritime Mobile Service Identity (MMSI) number is meant to be unique to an individual AIS 
device, and AIS messages from a given MMSI are intended to always correspond to the same vessel. However, MMSI numbers can 
be manually changed and, in practice, it is not uncommon to see the same MMSI number being used by two or more vessels simultaneously. 
GFW algorithms attempt to identify this behavior and exclude MMSI where identity spoofing is an issue. A few hundred fishing MMSI out 
of tens of thousands have this problem regularly, and most of these have been removed from our fishing effort data to avoid confusion. 
See this blog post for more information: Spoofing: One Identity Shared by Multiple Vessels (https://globalfishingwatch.org/data/spoofing-one-identity-shared-by-multiple-vessels/)

Multigear vessels, reflagging, and MMSI recycling: This dataset uses MMSI as the vessel identifier. At present, the GFW 
vessel classification model predicts a single gear type for each MMSI based on the total activity of that MMSI in the AIS 
data. In reality, over the time range of this dataset (2012-2024), some fishing vessels have changed fishing gears, 
while others may use multiple different fishing gears. As a result, some fishing activity by MMSI associated with these 
multigear vessels will not be attributed to the correct fishing gear. There are also several situations where a vessel 
may correspond to multiple MMSI over time, and vice versa. A vessel will receive a new MMSI if it changes its country of 
registration, a practice known as reflagging. Additionally, old MMSI may be recycled to new vessels, or an AIS device may 
be transferred to a new vessel without the MMSI being reset. Therefore, in some cases an MMSI will correspond to one vessel 
for a period of time, and then to another vessel later on. If the vessels are different classes, our algorithm will produce 
an incorrect result given that only one vessel class will be predicted for the MMSI despite the MMSI corresponding to more 
than one vessel class. These challenges can be improved with information from vessel registries, however, public information 
on vessel registration is limited, fragmented and inconsistent, making it challenging to reliably track a vessel’s identity 
over time. This is an active area of work for GFW and we hope to publish data with improved vessel identifiers in the near future.

Invalid MMSI numbers or MMSI’s corresponding to incorrect flag state: MMSI numbers are nine-digit numbers, with the first 
three digits usually a Maritime Identification Digits (MID). MIDs range from 201 to 775 and denote the administration 
(country) or geographical area of the administration responsible for the ship identified by the MMSI number. Some MMSI, 
such as those corresponding to helicopters, other aircraft, or navigational aids such as buoys can start with 0, 1, or 9. 
However, MMSI numbers can be manually changed and it is not uncommon to see MMSI numbers that either have too few digits 
or do not have the first three digits of their MMSI correctly correspond to a flag state. This presents two challenges. 
First, for vessels without registry information, the MID is the only indication of the vessel’s flag state and this 
information is missing from invalid MMSI numbers. In this dataset, MMSI with invalid MIDs are assigned a flag State 
of UNKNOWN. If the MMSI spends >50% of its fishing hours in a single EEZ, then it is assigned a flag State of UNKNOWN-[ISO], 
in which [ISO] corresponds to the EEZ in which it spends the majority of its fishing hours. For example, many vessels 
fishing in Chinese waters broadcast invalid MIDs, so there are a number of MMSI in our dataset assigned a flag of 
UNKNOWN-CHN. Second, if a vessel uses a valid MMSI but with an incorrect MID, the activity of that vessel will be 
attributed to an incorrect flag state. A known example of this latter issue is with the Albanian MID code (201), 
which is commonly known to be incorrectly used by vessels in Asia, likely because it is the first valid MID code 
numerically. Additionally, vessels fishing in Chinese waters frequently broadcast invalid or non-Chinese MIDs, so nearly 
every flag state will appear to have some fishing activity around China; in general, this activity most likely represents 
Chinese vessels broadcasting an incorrect MID rather than vessels of other flag states fishing in these waters.  

AIS reception: For a number of reasons, not all AIS messages that are broadcast by a vessel are received by terrestrial 
and satellite receivers. AIS messages may overlap in time, especially in areas of high vessel density, and their 
signals can interfere (“collide”) with one another, preventing messages from being received by satellites. 
This issue does not affect terrestrial AIS receivers, however, because terrestrial receivers can receive messages 
only a few tens of miles from shore, and thus fewer messages can interfere with one another. Terrestrial receivers, 
though, are not evenly distributed along the world’s coastlines and an AIS device must be within a few tens of nautical 
miles of a terrestrial receiver for its messages to be received. Lastly, AIS devices vary in signal strength and 
broadcast messages at different rates depending on vessel speed and AIS device type (class A or B). Through a combination 
of these factors, the quantity and quality of AIS data is not homogenous around the world and it is not uncommon to observe 
transmission gaps in a vessel’s AIS signal that are many hours long.  A major change to our AIS reception coverage occurred 
in 2022, when we began receiving data from Dynamic AIS from Spire Global, one of our AIS providers. Dynamic AIS receivers 
are AIS receivers carried by several thousand large vessels traveling along major shipping routes; these receivers store 
AIS messages that they pick up from their surroundings, then retransmit the messages to satellites. Dynamic AIS improves 
reception of AIS messages in areas where high vessel density precludes good satellite reception of AIS and where terrestrial 
receivers are out-of-range or do not share their data. This improved AIS reception due to dynamic AIS is particularly 
noticeable in the South China Sea and East China Sea. Users should be aware that apparent large increases in fishing 
hours between 2021 and 2022/2023, especially in this area, are likely attributable to increases in AIS reception due to 
the incorporation of dynamic AIS data starting in 2022. Global Fishing Watch is working to publish a dataset of AIS 
reception to help users understand and account for these underlying changes in the AIS data.

False positives on fishing activity: The Global Fishing Watch fishing detection neural net model predicts whether 
individual AIS positions are fishing or non-fishing positions. When making predictions, this model considers a complex 
set of features related to a vessel’s movements, of which speed is the most important. In fact, the initial Global Fishing Watch 
fishing detection model was a simple speed filtering algorithm. Unfortunately, given the importance of speed to the detection 
of fishing activity, our fishing detection model is prone to false positive predictions during transits when a vessel is moving 
slow. For example, fishing vessels passing through canals or moving slowly near ports (e.g. Chinese vessels awaiting authorization 
to enter Peruvian ports) are problematic. The issue is particularly problematic for squid jigging vessels, which can remain 
relatively stationary when fishing. For this release, we have manually applied a filter that requires any vessel traveling in 
a straight line for more than 24 hours to not be fishing, which affects less than half a percent of our fishing activity. 
However, some false positives still appear, particularly around canals or other areas in which vessels transit slowly following 
paths that are not straight.

Offsetting: Occasionally, the AIS messages transmitted from a ship provide a location that makes no sense, say, in the middle 
of the Antarctic or over a mountain range. In such cases, either the AIS transponder has malfunctioned, the data got scrambled 
in transmission, or the system has been tampered with in a deliberate attempt to disguise the vessel’s location. GFW identifies 
these offsetting vessels using our algorithms and manual review, and we exclude sections of track that contain offsetting. 
We are working on methods to correct rather than exclude these bad portions of tracks. For more information, see these blog 
posts: When Vessels Report False Locations (https://globalfishingwatch.org/data/when-vessels-report-false-locations/); 
AIS ship tracking data shows false vessels tracks circling above Point Reyes, near San Francisco 
(https://globalfishingwatch.org/data-blog/circling-above-point-reyes/)

Missing ~90% of Orbcomm data on October 1st 2015: In the current iteration of the GFW AIS dataset, there was an issue when 
ingesting data for October 1st, 2015 and  ~90% of AIS messages are missing on this date. As a result, data for this date 
includes approximately 50% of the MMSI active at the time and captures ~30% of activity. The loss of activity (hours) is not 
higher because the GFW algorithm assigns time to each AIS position based on the time since the previous position. Thus, only 
activity for vessels entirely absent from the data on this date is lost. For vessels included in the data, the time associated 
with the missing data points is simply attributed to the remaining positions.


