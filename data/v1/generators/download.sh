#!/bin/sh

# Download files from OR-Library by J E Beasley.
# http://people.brunel.ac.uk/~mastjjb/jeb/orlib/unitinfo.html

set -x

mkdir -p raw

cd raw

if [ ! -f unitnew.zip ]; then
    curl -O http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/unitnew.zip
fi

if [ ! -d RCUC ]; then
    unzip unitnew.zip
fi

cd ..
