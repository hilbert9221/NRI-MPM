from functools import partial
import os
from utils.general import pool_map, write_pickle
from generate.nri import Spring, Charged
from generate.kuramoto import Kuramoto
import config as cfg
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dyn', type=str, default='kuramoto',
                        help='Type of dynamics: spring, charged or kuramoto.')
    parser.add_argument('--interval', type=int, default=cfg.interval,
                        help='Down-sampling frequency, 10 for kuramoto and 100 for spring and charged.')
    parser.add_argument('--size', type=int, default=cfg.size,
                        help='Number of particles.')
    parser.add_argument('--jobs', type=int, default=64,
                        help='Nmuber of workers for data generation.')
    return parser.parse_args()


def wrapper(func, i):
    """A wrapper for any function func taking an unused parameter i."""
    return func()


def test():
    """
    A multi-processing implementation of data generation to allow parallel generation.
    """
    args = get_args()
    print(args)
    choice = args.dyn
    if choice == 'spring':
        ins = Spring(args.size, cfg.train_steps, args.interval)
    elif choice == 'charged':
        ins = Charged(args.size, cfg.train_steps, args.interval)
    elif choice == 'kuramoto':
        ins = Kuramoto(args.size, cfg.test_steps, args.interval)
    else:
        raise NotImplementedError('spring, charged or kuramoto')
    if choice == 'kuramoto':
        ins.set_trunc(cfg.train_steps)
    f = partial(wrapper, ins.generate)
    train = pool_map(f, range(cfg.train), jobs=args.jobs)
    val = pool_map(f, range(cfg.val), jobs=args.jobs)
    # longer time series for the test stage
    ins.set_epoch(cfg.test_steps)
    if choice == 'kuramoto':
        ins.set_trunc(cfg.test_steps)
    test = pool_map(f, range(cfg.test), jobs=args.jobs)
    data = [train, val, test]

    if not os.path.exists('data'):
        os.mkdir('data')
    path = 'data/{}'.format(choice)
    if not os.path.exists(path):
        os.mkdir(path)
    outfile = '{}/{}.pkl'.format(path, args.size)
    write_pickle(data, outfile)


if __name__ == "__main__":
    test()
