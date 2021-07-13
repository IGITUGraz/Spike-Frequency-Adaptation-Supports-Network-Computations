# Spike frequency adaptation supports network computations on temporally dispersed information

This repository provides a tensorflow library and a tutorial to train a recurrent spiking neural network of adaptive 
neurons (ALIF, neurons with spike frequency adaptation).
The adaptation is implemented as adaptive threshold.
The scripts contained reproduce many results from the paper [1].

[1] [Spike frequency adaptation supports network computations on temporally dispersed information  
Darjan Salaj, Anand Subramoney, Ceca Kraišniković, Guillaume Bellec, Robert Legenstein, Wolfgang Maass](https://www.biorxiv.org/content/10.1101/2020.05.11.081513v1.abstract)

## Reproducing results from [1]

Python scripts are in the `bin` directory, and the notebooks are in the root.

- Figure 1B: constant current response of ALIF neuron `560_fig1_ALIF_step_current_response.py`
- Figure 1E: 2 neuron STORE-RECALL `tutorial_storerecall_2neuron_solution.py` and `plot_storerecall.ipynb`
- Figure 2A: 20-dim STORE-RECALL `tutorial_extended_storerecall_with_LSNN.py` and `plot_ext_storerecall.ipynb`
- Figure 2B: STORE-RECALL comparison of mechanisms `tutorial_storerecall_with_LSNN.py` and `tutorial_storerecall_with_STP.py`
- Figure 2C: sMNIST comparison of mechanisms `tutorial_sequential_mnist_with_LSNN.py` and `tutorial_sequential_mnist_with_STP.py`
- Table 1: `tutorial_storerecall_with_LSNN.py`
- Suppl. Figure S1: intrinsic time scale measurements `autocorr.py`
- Suppl. Figure S2: constant current response of other slow mechanisms `other_mechanisms_dynamics_plot.ipynb` and `nALIF_current_response.py`
- Suppl. Figure S4: Delayed-memory XOR `tutorial_temporalXOR_with_LSNN.py`
- Suppl. Figure S8: Adaptation index `plot_adaptation_index_dist.ipynb`


## Running the scripts

Enter the root directory of the repository and run the training scripts as following:

    PYTHONPATH=. python3 bin/tutorial_storerecall_with_LSNN.py --reproduce=560_ALIF

## Troubleshooting

If the scripts fail with the following error:
`` Illegal instruction (core dumped) ``

It is most probably due to the lack of AVX instructions on the machine you are using.
A known workaround is to reinstall the LSNN package with older tensorflow version (1.5).
Change requirements.txt to contain:

`` tensorflow==1.5 ``
