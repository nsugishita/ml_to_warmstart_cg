This is the official implementation of "Use of Machine Learning Models to
Warmstart Column Generation for Unit Commitment" by Nagisa Sugishita, Andreas
Grothey and Ken McKinnon.

This is tested with Python 3.9.

# Install

## Virtual Env

First set up a virtual environment:

```
python3 -m venv env
. ./env/bin/activate
```
## Insall Python Package

Except cplex, all depenencies can be installed via pip:

```
pip install -r requirements.txt
pip install -e .
```

One needs to install cplex python interface separately.
Consult to cplex documentation for details.
Typical steps are

```
cd /path/to/cplex/python/3.6/x86-64_linux
pip install -e .
```

Furthermore, PyTorch is required.
See the [documentation](https://pytorch.org/) for the download instructions.

## Download Data

We need to download data to run the experiments.
Check the license of SDPLIB etc. and use the following commands.

```
pushd data/v1/demand
./download.sh
python parse_demand_data.py
popd
pushd data/v1/generators
./download.sh
python parse_generator_data.py
popd
```

## Build C++ Extensions

Finally, we need to build extensions using the following commands.

```
pushd extensions/uniquelist
./build.sh
popd
```

# Run Experiments

Run the experiments as follows:

```
./scripts/run_experiments.sh
```
