#!/usr/bin/env bash

set -eux

BASE_NAME=$(basename $(pwd))

# export CXX=/usr/bin/g++-11


if [ $BASE_NAME != "HyperVision_Cpp" ] && [ $BASE_NAME != "hypervision" ]; then
    echo "This script should be executed in the root dir of HyperVision."
    exit -1
fi

echo "Rebuild HyperVision."

if [ -d "./build" ]; then
    echo "Old build dir is removed."
    rm -rf ./build
fi

mkdir build && cd $_ && cmake -G Ninja .. && ninja  && cd ..
if [ $? == 0 ]; then
    echo "Rebuild finished."
else
    echo "Rebuild failed."
fi
