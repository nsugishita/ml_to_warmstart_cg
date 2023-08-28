# Historical demand data in UK

The data is obtained from
[national grid ESO](https://demandforecast.nationalgrid.com/efs_demand_forecast/faces/DataExplorer).

## Download

By running download.sh, relevant data are downloaded in `raw` directry.
Description of fields can be found therein, but in microsoft word format.
The content is converted in markdown (plain text) and stored in `doc` directry.

## Parse

`parse_demand_data.py` is a script to concatenate all demand
data into single csv file
and save it in `out` directry.  Use the following command:
```sh
python parse_demand_data.py
```


# Notes

The demand data contains average demand over 30-minute periods.
Hence, usually there are 48 data points for one day.
However, each year, there is one day in March and in October which have
46 and 50 data points (probably due to the summer time adjustabment).

## Days with only 46 data

datetime.date(2017, 3, 26)
datetime.date(2018, 3, 25)
datetime.date(2019, 3, 31)

## Days with only 46 data

datetime.date(2017, 10, 29)
datetime.date(2018, 10, 28)
datetime.date(2019, 10, 27)
