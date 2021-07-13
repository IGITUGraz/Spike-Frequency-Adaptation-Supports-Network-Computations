import multiprocessing as mp
import queue
from randomgen.generator import RandomGenerator


def multiprocify(fn, seed, *args, n_generators=8, **kwargs):
    """
    Assuming that the input function generates data points (one at a time)
    This decorator spins up many instances of the function, and returns a function that returns from the queue.
    An argument of `random_gen` will be passed into `fn` initialized with different seeds for different processes
    This function itself requires a `seed` argument to do said initialization
    """
    py_queue = mp.Queue(maxsize=100)
    stop_generators = mp.Value('b', False)
    generators = []

    def fn_q(seed):
        random_gen = RandomGenerator()
        random_gen.seed(seed)

        new_kwargs = dict(kwargs)
        new_kwargs.update(random_gen=random_gen)

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
                    break

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


def generatorify(fn, seed, *args, **kwargs):
    random_gen = RandomGenerator()
    random_gen.seed(seed)

    new_kwargs = dict(kwargs)
    new_kwargs.update(random_gen=random_gen)

    def fn_ret():
        while True:
            yield fn(*args, **new_kwargs)

    return fn_ret


def fify(fn, *args, **kwargs):
    def fn_ret(*nargs, **nkwargs):
        return fn(*args, *nargs, **kwargs, **nkwargs)

    return fn_ret


def fchain(fn1, fn2):
    def ret_fn(**kwargs):
        return fn1(fn2(**kwargs), **kwargs)

    return ret_fn
