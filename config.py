from time import strftime, gmtime


'''Hyperparameters for the input.'''
dyn = 'spring'
size = 10
# 3 for kuramoto and 4 for spring and charged.
dim = 4
edge_type = 2

'''Hyperparameters for the model.'''
enc = 'GNNENC'
dec = 'GNNDEC'
n_hid = 2 ** 8
input_emb_hid = 2 ** 7
# Dimension of hidden layers of attention mechanisms.
emb_hid = 2 ** 8
att_hid = 2 ** 8
skip = False
# Soft constraint for symmetry.
reg = 0.
no_reg = False
# Hard constraint for symmetry.
sym = False

'''Hyperparameters for training.'''
# Scale the loss to avoid gradient explosion.
scale = 1e-5
epochs = 500
lr = 2.5e-4
lr_decay = 200
gamma = 0.5
batch_size = 2 ** 7
M = 10
gpu = True

'''Hyperparameters for data generation.'''
base = 10 ** 4
train = 5 * base
test = base
val = base

timesteps = 50
train_steps = 49
test_steps = 99
temp = 0.5
# NOTE: 10 for kuramoto, 100 for spring and charged.
interval = 10 ** 1
samples = 10 ** 2

'''Others'''
log = strftime('logs/{}_{}_%m-%d_%H:%M:%S/{}_{}.txt'.format(
    dyn, size, enc, dec), gmtime())
# Run n rounds of the code.
rounds = 1


def init_args(args):
    global enc, dec, size, dyn, log, sym, epochs

    enc = args.enc
    dec = args.dec
    dyn = args.dyn
    size = args.size
    sym = args.sym
    epochs = args.epochs
    log = strftime('logs/{}_{}_%m-%d_%H:%M:%S/{}_{}.txt'.format(dyn, size, enc, dec), gmtime())
