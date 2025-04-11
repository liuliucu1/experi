#!/usr/bin/env bash

set -eux

ninja


ARR=(
    "Friday"
    "Thursday"
    "Wednesday"
    "Tuesday"
)

for item in ${ARR[@]}; do
    ./HyperVision -config /home/sduu2/userspace-20T-1/yyr/HyperVision_Cpp/configuration/CIC2017/${item}.json > ../cache/${item}.log # &
done
