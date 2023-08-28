# Generator data

Data are adopted from
[OR-Library](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/unitinfo.html).


## download

download.sh downloads data files from the website.
files are saved in `raw` direcory.

## parse

 To parse data to a python friendly format:

 ```sh
 python parse_generator_data.py
 ```

The results are stored in a directory `out`.



Below are from ReadMe.txt.

This document describes a proposed format for instances of the
ramp-constrained, hydro-thermal Unit Commitment problem in electric power
generation. Randomly generated, “realistic” instances encoded in this format
are publicly available, and have been used to test some algorithmic approaches
to the problem in
- Frangioni, C. Gentile, F. Lacalandra “Solving Unit Commitment Problems with
  General Ramp Contraints” International Journal of Electrical Power and Energy
  Systems, to appear, 2008
- Frangioni, C. Gentile “Solving Nonlinear Single-Unit Commitment Problems with
  Ramping Constraints” Operations Research 54(4), p. 767 - 775, 2006 These
  papers also describe in details the Unit Commitment model.
