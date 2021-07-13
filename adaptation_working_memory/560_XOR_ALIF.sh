#!/usr/bin/env bash

NUM_SIMS=10

for ((i=1; i<=NUM_SIMS; i++)); do
    PYTHONPATH=. python3 bin/tutorial_temporalXOR_with_LSNN.py --n_adaptive=80 --n_regular=0 --comment=reg50_ALIF_$i
done
