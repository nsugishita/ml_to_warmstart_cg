# -*- coding: utf-8 -*-

import os
from typing import Dict

from setuptools import setup

package_name = "sucpy"

_locals: Dict = {}
with open(os.path.join(package_name, "_version.py")) as fp:
    exec(fp.read(), None, _locals)
__version__ = _locals["__version__"]

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name=package_name,
    version=__version__,
    description="ML to warmstart CG.",
    long_description=readme,
    author="Nagisa Sugishita",
    author_email="nsugishi@ed.ac.uk",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas>=0.24",
        "scikit-learn",
    ],
    license=license,
    packages=[
        "sucpy",
    ],
    include_package_data=True,
    test_suite="tests",
)
