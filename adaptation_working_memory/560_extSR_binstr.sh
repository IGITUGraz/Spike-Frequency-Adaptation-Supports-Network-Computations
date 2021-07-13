#!/usr/bin/env bash

# run with: CUDA_VISIBLE_DEVICES=? 560_SR_ALIF.sh

COMMON="python3 -u bin/tutorial_extended_storerecall_with_LSNN.py --reproduce=560_extSR --n_adaptive=500 --tau_char=200 --max_in_bit_prob=0.2 --f0=400 --no_recall_distr=False --batch_train=256 --FastALIF=True"
NUM_SIMS=5
HOST=`hostname -s`

for ((i=0; i<NUM_SIMS; i++)); do
    TIME=`date "+%H_%M_%S"`
    # PYTHONPATH=. ${COMMON} --comment=560_ExtSR_FastLONG3_${i} | tee stdout_560_extSR_binstr_FastLONG3_${i}_${HOST}_${TIME}.txt
    PYTHONPATH=. ${COMMON} --eprop=True --tau_a=1600 --comment=560_ExtSR_epropSym_taua1.6_${HOST}_${i} | tee stdout_560_extSR_epropSym_${i}_${HOST}_${TIME}.txt
done


# PYTHONPATH=. python3 -u bin/tutorial_extended_storerecall_with_LSNN.py --reproduce=560_extSR --n_adaptive=500 --tau_char=200 --max_in_bit_prob=0.2 --f0=400 --comment=DEBUG_FastLONG_4SRns_A500_seqDel4 --no_recall_distr=False --batch_train=256