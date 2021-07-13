#!/usr/bin/env bash

NUM_SIMS=10

for ((i=1; i<=NUM_SIMS; i++)); do
    PYTHONPATH=. python3 bin/tutorial_temporalXOR_with_LSNN.py --n_adaptive=0 --n_regular=80 --beta=0. --comment=reg50_LIF_$i
done
