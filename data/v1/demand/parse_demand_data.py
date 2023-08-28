# -*- coding: utf-8 -*-

"""Concatenate all data in `raw`."""

import os

import pandas as pd

output_dir = "out"
demand_file = os.path.join(output_dir, "demand.csv")
validation_file = os.path.join(output_dir, "invalid_dates.csv")


def concatenate():
    """Concatenate all csv files.

    This concatenates all csv files in `raw` directry,
    and parse date fields into python friendly format.
    The result is saved as `out/demand.csv`.
    """
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(demand_file):
        print(f"file {demand_file} already exists.")
        return

    # Get a list of file names in `raw`.
    # This does not contain the directry name.
    inputs = list(os.listdir("raw"))
    inputs = [i for i in inputs if i.endswith("csv")]  # Use only csv files.
    inputs.sort()
    print(f"reading: {inputs}")

    # Accumulate the result on `df`.
    df = pd.DataFrame()
    for input in inputs:  # Load each files and append to `df`.
        df_new = pd.read_csv(os.path.join("raw", input))
        df = pd.concat([df, df_new], ignore_index=True, sort=False)

    # Parse datetime and sort the records.
    df["SETTLEMENT_DATE"] = pd.to_datetime(
        df["SETTLEMENT_DATE"], format="%d-%b-%Y"
    )
    df.sort_values(["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"], inplace=True)
    df.to_csv(demand_file, index=False)


def validate():
    """Validate data.

    This validates the demand data.  Namely, this checks
    the number of records per day and find out dates
    which has records more or less than 48.
    This output the results on `out/invalid_dates.csv`.
    """
    df = pd.read_csv(demand_file, parse_dates=["SETTLEMENT_DATE"])
    counts = (
        df["SETTLEMENT_DATE"].groupby(df["SETTLEMENT_DATE"].dt.date).agg(len)
    )
    counts = counts[counts != 48]
    counts.name = "num_data"
    counts.index.name = "date"
    counts.to_csv(validation_file, header=True)


if __name__ == "__main__":
    concatenate()
    validate()
