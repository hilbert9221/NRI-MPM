import torch
import config as cfg
from instructors.XNRI import XNRIIns
from instructors.XNRI_enc import XNRIENCIns
from instructors.XNRI_dec import XNRIDECIns
from argparse import ArgumentParser
from utils.general import read_pickle
from models.encoder import AttENC, RNNENC, GNNENC
from models.decoder import GNNDEC, RNNDEC, AttDEC
from models.nri import NRIModel
from torch.nn.parallel import DataParallel
from generate.load import load_kuramoto, load_nri


def init_args():
    parser = ArgumentParser()
    parser.add_argument('--dyn', type=str, default='charged', 
    help='Type of dynamics: spring, charged or kuramoto.')
    parser.add_argument('--size', type=int, default=5, 
    help='Number of particles.')
    parser.add_argument('--dim', type=int, default=4, 
    help='Dimension of the input states.')
    parser.add_argument('--epochs', type=int, default=500, 
    help='Number of training epochs. 0 for testing.')
    parser.add_argument('--reg', type=float, default=0, 
    help='Penalty factor for the symmetric prior.')
    parser.add_argument('--batch', type=int, default=2 ** 7, help='Batch size.')
    parser.add_argument('--skip', action='store_true', default=False,
    help='Skip the last type of edge.')
    parser.add_argument('--no_reg', action='store_true', default=False,
    help='Omit the regularization term when using the loss as an validation metric.')
    parser.add_argument('--sym', action='store_true', default=False,
    help='Hard symmetric constraint.')
    parser.add_argument('--reduce', type=str, default='cnn',
    help='Method for relation embedding, mlp or cnn.')
    parser.add_argument('--enc', type=str, default='RNNENC', help='Encoder.')
    parser.add_argument('--dec', type=str, default='RNNDEC', help='Decoder.')
    parser.add_argument('--scheme', type=str, default='both',
    help='Training schemes: both, enc or dec.')
    parser.add_argument('--load_path', type=str, default='',
    help='Where to load a pre-trained model.')
    return parser.parse_args()


def load_data(args):
    path = 'data/{}/{}.pkl'.format(args.dyn, args.size)
    train, val, test = read_pickle(path)
    data = {'train': train, 'val': val, 'test': test}
    return data


def run():
    args = init_args()
    cfg.init_args(args)
    # load data
    data = load_data(args)
    if args.dyn == 'kuramoto':
        data, es, _ = load_kuramoto(data, args.size)
    else:
        data, es, _ = load_nri(data, args.size)
    dim = args.dim if args.reduce == 'cnn' else args.dim * cfg.train_steps
    encs = {
        'GNNENC': GNNENC,
        'RNNENC': RNNENC,
        'AttENC': AttENC,
    }
    decs = {
        'GNNDEC': GNNDEC,
        'RNNDEC': RNNDEC,
        'AttDEC': AttDEC,
    }
    encoder = encs[args.enc](dim, cfg.n_hid, cfg.edge_type, reducer=args.reduce)
    decoder = decs[args.dec](args.dim, cfg.edge_type, cfg.n_hid, cfg.n_hid, cfg.n_hid, skip_first=args.skip)
    model = NRIModel(encoder, decoder, es, args.size)
    if args.load_path:
        name = 'logs/{}/best.pth'.format(args.load_path)
        model.load_state_dict(torch.load(name)) 
    model = DataParallel(model)
    if cfg.gpu:
        model = model.cuda()
    if args.scheme == 'both':
        # Normal training.
        ins = XNRIIns(model, data, es, args)
    elif args.scheme == 'enc':
        # Only train the encoder.
        ins = XNRIENCIns(model, data, es, args)
    elif args.scheme == 'dec':
        # Only train the decoder.
        ins = XNRIDECIns(model, data, es, args)
    else:
        raise NotImplementedError('training scheme: both, enc or dec')
    ins.train()


if __name__ == "__main__":
    for _ in range(cfg.rounds):
        run()
