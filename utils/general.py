import pickle
from multiprocessing import Pool
import time
import operator
from functools import reduce


def write_pickle(obj, outfile, protocol=-1):
    """
    A wrapper for pickle.dump().
    """
    with open(outfile, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)


def read_pickle(infile):
    """
    A wrapper for pickle.load().
    """
    with open(infile, 'rb') as f:
        return pickle.load(f)


def pool_map(f, args, init=None, multiple=False, jobs=4):
    """
    A wrapper for multiprocessing.Pool.
    """
    t = time.time()
    pool = Pool(jobs, init)
    if multiple:
        result = pool.starmap(f, args)
    else:
        result = pool.map(f, args)
    print('time {:.2f}'.format(time.time() - t))
    pool.close()
    pool.join()
    return result


def prod(ls):
    """
    Compute the product of a list of numbers.
    """
    return reduce(operator.mul, ls, 1)
