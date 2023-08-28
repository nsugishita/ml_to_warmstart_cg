# -*- coding: utf-8 -*-

"""Parse downloaded unit commitment data.

This parses thermal generator data files in `raw` directry.

To run this script, type
$ python parse.py

The results are saved in `out` directry.
"""

import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

# Path to a directry where *.mod files are located.
raw_data_dir = os.path.join("raw", "RCUC", "T-Ramp")
# Path to a drectry where parsed data are saved.
output_dir = "out"


def main():
    """Entry point of this script."""
    # List of name of datafiles.  E.g., 100_0_1_w.mod.
    filenames = os.listdir(raw_data_dir)

    # create a directry where processed data are saved.
    os.makedirs(output_dir, exist_ok=True)

    for filename in filenames:
        parse_mod_file(filename)


def parse_mod_file(filename):
    """Parse a given mod file and save in out directry.

    This takes a name of mod file (with extension, e.g. `10_0_1_w.mod`)
    and parse its contents.
    The followings are saved in `out` directry.
    - *_header.json
        Information in a header.
    - *_load.txt
        Load data.
    - *_spining_reserve.txt
        Spinning reserve data.
    - *.csv
        Generators' data.

    Parameters
    ----------
    filename : str
        A filename of a data file to be parsed.  This must contain
        the extension as well, e.g. `10_0_1_w.mod`
    """
    basename = os.path.splitext(filename)[0]  # Remove the extension `.mod`.
    filepath = os.path.join(
        raw_data_dir, filename
    )  # Get full path to the data file.

    # Templates to parse the header.
    templates = [
        ("ProblemNum", int),
        ("HorizonLen", int),
        ("NumThermal", int),
        ("NumHydro", int),
        ("NumCascade", int),
        ("LoadCurve",),
        ("MinSystemCapacity", float),
        ("MaxSystemCapacity", float),
        ("MaxThermalCapacity", float),
        ("Loads", int, int),
    ]

    # Parse lines in the header and store the result on this list.
    # This will look like
    # [('ProblemNum', 0), ('HorizonLen', 24), ('NumThermal', 200), ...]
    parsed_lines = []

    with open(filepath, "r") as f:
        # ---
        # Parse the header.
        # ---
        for index_line, template in enumerate(templates):
            # Read line and match with a template.
            # E.g. template = ('ProblemNum', int).
            line = f.readline()
            cells = line.split()
            parsed_cells = ()  # This will look like ('ProblemNum', 0).
            # Iterate over items in template.
            for index_token, token in enumerate(template):
                if isinstance(
                    token, str
                ):  # str means we have some token expected.
                    assert token == cells[index_token]
                    parsed_cells += (token,)
                elif callable(
                    token
                ):  # callable means we expect a numeric data.
                    try:
                        parsed_cells += (token(cells[index_token]),)
                    except Exception:
                        raise ValueError(
                            f"Failed to parse at line {index_line} at "
                            f"token {index_token}"
                        )
            parsed_lines.append(parsed_cells)  # Successfully parsed one line.

        # header will look like
        # {'ProblemNum': 0, 'HorizonLen': 24, 'NumThermal': 200, ...,
        # 'Loads', (1, 24)}
        # This will be dumped on a json file.
        header = OrderedDict()

        for parsed_line in parsed_lines:
            if len(parsed_line) == 1:
                continue
            elif len(parsed_line) == 2:
                header[parsed_line[0]] = parsed_line[1]
            else:
                header[parsed_line[0]] = parsed_line[1:]

        header_filename = os.path.join(output_dir, f"{basename}_header.json")
        with open(header_filename, "w") as t:
            json.dump(header, t, indent=4)

        # ---
        # Parse load.
        # ---
        # Load data is stored as a matrix, so read line by line and
        # track how many data each line contrains.
        count = 0  # Counter to track how many numbers are read.
        load = []  # Store read numbers on this list.
        while True:
            if count > (header["Loads"][1] - header["Loads"][0]):
                break
            line = f.readline()
            cells = line.split()
            load += [float(x) for x in cells]
            count += len(cells)
        load_filename = os.path.join(output_dir, f"{basename}_load.txt")
        np.savetxt(load_filename, load)

        # ---
        # Parse spinning reserve data.
        # ---
        f.readline()  # -> Spinning Reserve xx
        count = 0  # Counter to track how many numbers are read.
        spinning_reserve = []  # Store read numbers on this list.
        while True:
            if count > (header["Loads"][1] - header["Loads"][0]):
                break
            line = f.readline()
            cells = line.split()
            spinning_reserve += [float(x) for x in cells]
            count += len(cells)
        spinning_reserve_filename = os.path.join(
            output_dir, f"{basename}_spinning_reserve.txt"
        )
        np.savetxt(spinning_reserve_filename, spinning_reserve)

        # ---
        # Parse thermal generator data.
        # ---
        line = f.readline()  # -> Thermal section
        columns = [
            "generator_index",
            "quadratic_cost",
            "linear_cost",
            "constant_cost",
            "min_out",
            "max_out",
            "init_status",
            "min_up",
            "min_down",
            "cool_and_fuel_cost",
            "hot_and_fuel_cost",
            "tau",
            "tau_max",
            "fixed_cost",
            "succ",
            "p0",
            "ramp_up",
            "ramp_down",
        ]
        df_generators = pd.DataFrame(
            columns=columns, index=range(header["NumThermal"])
        )
        for index_generator in range(header["NumThermal"]):
            line = f.readline()  # -> numeric data except ramp_up/down.
            record = line.split()
            line = f.readline()  # -> RampConstraints X X
            record += line.split()[1:]
            df_generators.iloc[index_generator, :] = record

        generators_filename = os.path.join(output_dir, f"{basename}.csv")
        df_generators.to_csv(generators_filename, index=False)


if __name__ == "__main__":
    main()
