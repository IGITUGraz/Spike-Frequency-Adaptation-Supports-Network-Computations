import multiprocessing as mp
import queue
import time

import numpy as np


def multiprocify(fn, *args, **kwargs):
    """
    Assuming that the input function generates data points (one at a time)
    This decorator spins up many instances of the function, and returns a function that returns from the queue.
    """
    py_queue = mp.Queue(maxsize=1)
    stop_generators = mp.Value('b', False)
    generators = []
    n_generators = 2

    def fn_q(seed):
        new_kwargs = dict(kwargs)
        new_kwargs.update(seed=seed)
        while not stop_generators.value:
            while True:
                try:
                    elem = fn(*args, **new_kwargs)
                    py_queue.put(elem, timeout=1.)
                    break
                except queue.Full:
                    # print("Queue is full")
                    if stop_generators.value:
                        break
                except Exception as e:
                    print("Putting new elem failed: ", e)

    seed = kwargs['seed']

    # print("Starting processes")
    for i in range(n_generators):
        p = mp.Process(target=fn_q, args=(seed + i,))
        p.start()
        generators.append(p)
    # print("Done")

    def fn_ret():
        while True:
            try:
                elem = py_queue.get()
                yield elem
            except:
                break

        stop_generators.value = True
        for i in range(n_generators):
            generators[i].terminate()
            generators[i].join()
        print("Breaking")

    return fn_ret


def random_fn(seed):
    return np.random.rand()


if __name__ == '__main__':
    for a in multiprocify(random_fn, seed=1234)():
        # print(a)
        # time.sleep(1)
        pass
