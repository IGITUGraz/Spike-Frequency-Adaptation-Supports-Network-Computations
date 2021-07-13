#!/bin/bash

for dirname in $(ls -d results/tutorial_storerecall_with_LSNN/autocorr/*);
do
  if [[ $dirname != *"exclude"* ]]; then
    echo $dirname;
    python3 bin/autocorr.py $dirname
    wait
    sleep 1
  fi
 done
