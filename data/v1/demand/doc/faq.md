This is taken from national grid web site.
Please check:
https://www.nationalgrideso.com/balancing-data/data-finder-and-explorer


Frequently Asked Questions


Note: please take a note of our new website.
https://www.nationalgrid.com/uk/electricity/market-and-operational-data/data-explorer


What are the units measured in?

Answer: MW

Is each settlement time if the value and time represents
an instantaneous value or the average for the next 30 mins?

Answer. All values are an average over the half hour.

Are they in local time UK?

Answer: All times are UK time and for the half hour beginning.

What is included in the interconnector flows?

Answer:  Interconnector imports.

What does it mean by “Station load is 500MW in BST and 600MW in GMT”?

Answer: The generators themselves require power to generate electricity.
In cold conditions in the winter the generators use around 600MW worth
of electricity to generate power and in the summer it’s around 500MW.

Describe how you calculate the latest installed solar figure in your
daily reports?

Answer: It is calculated from Ofgem FIT register, PV from the ROC and
contracts from different incentive scheme (this number comes from external
source) and this equates to over 11GB capacity.

How do interconnector imports work?

Answer: It works by taking the total energy transmitted across the wires
over the half-hour window, and averaging that half-hour.

Is the I014_demand / I014_tgs / indo reports of electricity demand include
allowance for embedded wind, embedded solar and other embedded units?

Answer: They are based upon the Total Generation Output from Transmission
contracted units only.

For any given day, the sum of onshore and offshore wind generation does not
equal the sum of metered and embedded generation. Can you tell me why this is?

Answer: The onshore and offshore generation are a breakdown of wind
generation which are based on metered wind generation and embedded
generation is based on unmetered generation.


What makes the GB demand?

Answer: GB demand is anything that is producing electricity. We publish
demand that is connected to our transmission system. It would be impossible
to obtain GB demand as we do not receive any metering from DNO’s.

What the difference is between TSD and I014_TSD?

Answer: The transmission connected demand includes generation from pump
storage (ve) and interconnectors imports. Transmission System Demand (TSD)
is calculated using National Grid operational metering. Note that the
Transmission System Demand includes an estimate of station load.

The I014_TSD is Equivalent to TSD (above), but calculated using settlement
metered generation data from the I014 file where available.

How are interconnector flows calculated?

Answer: The flows are calculated from settlement data are derived by adding
all the energy trades recorded by Elexon by all interconnector parties per
settlement period.  The data provided by Elexon in their IO14 settlement
file is in MWh per settlement period, but in this spreadsheet they are
converted to average MW per half hour.

The settlement data is higher accuracy than the first dataset, which is
based on operational quality metering of the actual flow on the interconnector.
The IO14 file is not available until five days after the event, and so we
provide the lower quality operational metering as well as the more accurate
but slower settlement metering.

How are embedded Wind and Solar forecast modelled?

Wind and solar are forecasted similarly. Both use weather data (wind speed
and solar radiation respectively). We procure the weather data separately
and so are not able to publish it.

Weather data is applied to a physical model which estimates output. The
model uses installed capacity and location, and power curves put together
using sample data. The capacity and location data behind the model are our
best estimate based on a range of publicly available information – ROCs,
FiTs, DECC, Renewables UK.



If you have any questions or feedback please contact:

Nikhil Madani
Email: nikhil.madani@nationalgrid.com

