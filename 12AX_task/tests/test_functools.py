import numpy as np
import matplotlib.pyplot as plt
from randomgen.generator import RandomGenerator

from symltl import Task
from symltl.dataset import spike_encode, generate_string_single_task
from symltl.ftools import fchain, generatorify, fify
from symltl.plot import plot_spikes


def test_fchain():
    def fn0(a):
        return a + '+'

    def fn1(a):
        return a + '-'

    def fn2():
        return str(np.random.rand())

    print(fchain(fn0, fchain(fn1, fn2))())
    print(fchain(fn0, fchain(fn1, fn2))())
    print(fchain(fn0, fchain(fn1, fn2))())


def test_spike_gen():
    rg = RandomGenerator()
    rg.seed(3000)
    n_dt_per_step = 50

    gen_fn = generatorify(fchain(
        fify(spike_encode, random_gen=rg, n_dt_per_step=n_dt_per_step, n_neurons_per_channel=5, dt=1.),
        fify(generate_string_single_task, random_gen=rg, length=10, width=8, task=Task.COPY)))

    str_ = next(gen_fn())

    fig, axs = plt.subplots(2, 1)

    ax = axs[0]
    plot_spikes(str_['spike_input'].T, axs[0])
    ax.set(title='Spikes')

    ax = axs[1]
    im = ax.imshow(str_['input'].T, origin='lower')
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set(title='Input')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_spike_gen()
