#!/bin/sh

# Download files from national grid.
# https://demandforecast.nationalgrid.com/efs_demand_forecast/faces/DataExplorer

set -x

mkdir -p raw

if [ ! -f raw/demand2017.csv ]; then
    curl -o raw/demand2017.csv https://demandforecast.nationalgrid.com/efs_demand_forecast/downloadfile?filename=DemandData_2017_1551263464434.csv
fi

if [ ! -f raw/demand2018.csv ]; then
    curl -o raw/demand2018.csv https://demandforecast.nationalgrid.com/efs_demand_forecast/downloadfile?filename=DemandData_2018_1551263484189.csv
fi

if [ ! -f raw/demand2019.csv ]; then
    curl -o raw/demand2019.csv https://demandforecast.nationalgrid.com/efs_demand_forecast/downloadfile?filename=DemandData_2019_1581934964092.csv
fi

if [ ! -f raw/demand2020.csv ]; then
    curl -o raw/demand2020.csv https://demandforecast.nationalgrid.com/efs_demand_forecast/downloadfile?filename=DemandData_2020_1584352637369.csv
fi
