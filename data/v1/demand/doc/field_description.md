This is taken from national grid web site.
Please check:
https://www.nationalgrideso.com/balancing-data/data-finder-and-explorer


Historic Demand Data

Description of contents


This document explains what is included in the historic demand data files
(DemandData_YYYY.csv and DemandData_Update.csv) as published on
the National grid website:

https://www.nationalgrid.com/uk/electricity/market-operations-and-data/data-explorer

# SETTLEMENT_DATE

Settlement Date

# SETTLEMENT_PERIOD

Settlement Period

# ND

National Demand

This is the Great Britain generation requirement and is equivalent to
the Initial National Demand Outturn (INDO) and National Demand Forecast
as published on BM Reports. National Demand is the sum of metered generation,
but excludes generation required to meet station load, pump storage pumping
and interconnector exports.

National Demand is calculated as a sum of generation based on
National Grid operational generation metering.

# FORECAST_ACTUAL_INDICATOR

Indication of whether data is out-turn (A) or forecast (F).

# I014_ND

Equivalent to ND (above) but calculated using settlement metered
generation data from the I014 file where available.

# TSD

Transmission System Demand

This is the Transmission System generation requirement and is equivalent
to the Initial Transmission System Outturn (ITSDO) and Transmission
System Demand Forecast on BM Reports. Transmission System Demand is
equal to the ND plus the additional generation required to meet
station load, pump storage pumping and interconnector exports.

Transmission System Demand is calculated using National Grid operational
metering.

Note that the Transmission System Demand includes an estimate of station
load of 500MW in BST and 600MW in GMT.

# I014_TSD

Equivalent to TSD (above), but calculated using settlement metered
generation data from the I014 file where available.

# ENGLAND_WALES_DEMAND

England and Wales Demand, as ND above but on an England and Wales basis.

# EMBEDDED_WIND_GENERATION

Estimated Embedded Wind Generation

This is an estimate of the GB wind generation from wind farms which
do not have Transmission System metering installed. These wind farms
are embedded in the distribution network and invisible to National Grid.
Their effect is to suppress the electricity demand during periods of
high wind. The true output of these generators is not known so
an estimate is provided based on National Grid’s best model.

Note that embedded wind farms which do have Transmission System metering
are not included in this total.

For future dates a forecast value is shown. This is equivalent to
the data that feeds into the National Demand forecast published on
BM Reports.

# EMBEDDED_WIND_CAPACITY

Estimated Embedded Wind Capacity

This is National Grid’s best view of the installed embedded wind
capacity in GB. This is based on publically available information
compiled from a variety of sources and is not the definitive view.
It is consistent with the generation estimate provided above.

# EMBEDDED_SOLAR_GENERATION

Estimated Embedded Solar Generation

As embedded wind generation above, but for solar generation.

# EMBEDDED_SOLAR_CAPACITY

Embedded Solar Capacity

As embedded wind capacity above, but for solar generation.

# NON_BM_STOR

Non-BM Short-Term Operating Reserve

Operating reserve for units that are not included in the ND generator
definition. This can be in the form of generation or demand reduction.

# PUMP_STORAGE_PUMPING

Pump Storage Pumping

The demand due to pumping at hydro pump storage units; the -ve signifies
pumping load.

# I014_PUMP_STORAGE_PUMPING

As above, but calculated based on settlement data from the I014 file
where available.

# FRENCH_IMPORT, BRITNED_IMPORT, MOYLE_IMPORT, EAST_WEST_IMPORT

Interconnector Flow

The flow on the respective interconnector. -ve signifies export power
out from GB; +ve signifies import power into GB.

# I014_FRENCH_IMPORT, I014_BRITNED_IMPORT, I014_MOYLE_IMPORT, I014_EAST_WEST_IMPORT

Each as above, but calculated based on settlement data from
the I014 file where available.


If you have any questions or feedback please contact:

Email: .box.demand.forecast.queries
















