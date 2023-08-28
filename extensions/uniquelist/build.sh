#!/bin/sh

set -e

if [[ -z "${PREFIX}" ]]; then
  MY_PREFIX="/usr"
else
  MY_PREFIX="${PREFIX}"
fi

CMAKE="$MY_PREFIX/bin/cmake"


rm -rf build
mkdir -p build
pushd build
CXX="$MY_PREFIX/bin/g++" $CMAKE ..
make
popd

python3 test.py
echo "ok"
