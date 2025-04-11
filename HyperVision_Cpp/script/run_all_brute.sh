#!/usr/bin/env bash

set -eux

ninja


ARR=(
    # "charrdos"
    "cldaprdos"
    "dnsrdos"
    "dnsscan"
    "httpscan"
    "httpsscan"
    "icmpscan"
    "icmpsdos"
    "memcachedrdos"
    "ntprdos"
    "ntpscan"
    "riprdos"
    "rstsdos"
    "sqlscan"
    "ssdprdos"
    "sshscan"
    "synsdos"
    "udpsdos"
)

for item in ${ARR[@]}; do
    ./HyperVision -config /data16/yurun/HyperVision_Cpp/configuration/bruteforce/${item}.json > ../cache/${item}.log # &
done
