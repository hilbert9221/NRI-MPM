import torch
from torch import Tensor
import config as cfg
from torch.nn.functional import cross_entropy
from utils.torch_extension import edge_accuracy, asym_rate
from instructors.base import Instructor
from torch.utils.data.dataset import TensorDataset
from torch import optim
from torch.optim.lr_scheduler import StepLR
import numpy as np


class XNRIENCIns(Instructor):
    """
    Train the encoder in an supervised manner given the ground truth relations.
    """
    def __init__(self, model: torch.nn.DataParallel, data: dict, es: np.ndarray, cmd):
        """
        Args:
            model: an auto-encoder
            data: train / val /test set
            es: edge list
            cmd: command line parameters
        """
        super(XNRIENCIns, self).__init__(cmd)
        self.model = model
        self.data = {key: TensorDataset(value[0], value[1])
                     for key, value in data.items()}
        self.es = torch.LongTensor(es)
        # number of nodes
        self.size = cmd.size
        self.batch_size = cmd.batch
        # optimizer
        self.opt = optim.Adam(self.model.parameters(), lr=cfg.lr)
        # learning rate scheduler, same as in NRI
        self.scheduler = StepLR(self.opt, step_size=cfg.lr_decay, gamma=cfg.gamma)

    def train(self):
        # use the accuracy as the metric for model selection, default: 0
        val_best = 0
        # path to save the current best model
        prefix = '/'.join(cfg.log.split('/')[:-1])
        name = '{}/best.pth'.format(prefix)
        for epoch in range(1, 1 + self.cmd.epochs):
            self.model.train()
            # shuffle the data at each epoch
            data = self.load_data(self.data['train'], self.batch_size)
            loss_a = 0.
            N = 0.
            for adj, states in data:
                if cfg.gpu:
                    adj = adj.cuda()
                    states = states.cuda()
                scale = len(states) / self.batch_size
                # N: number of samples, equal to the batch size with possible exception for the last batch
                N += scale
                loss_a += scale * self.train_nri(states, adj)
            loss_a /= N 
            self.log.info('epoch {:03d} loss {:.3e}'.format(epoch, loss_a))
            acc = self.report('val')

            val_cur = max(acc, 1 - acc)
            if val_cur > val_best:
                # update the current best model when approaching a higher accuray
                val_best = val_cur
                torch.save(self.model.module.state_dict(), name)

            self.scheduler.step()
        # learning rate scheduling
        if self.cmd.epochs > 0:
            self.model.module.load_state_dict(torch.load(name))
        _ = self.report('test')

    def report(self, name: str) -> float:
        """
        Evaluate the accuracy.

        Args:
            name: 'train' / 'val' / 'test'
        
        Return:
            acc: accuracy of relation reconstruction
        """
        loss, acc, rate, sparse = self.evaluate(self.data[name])
        self.log.info('{} acc {:.4f} _acc {:.4f} rate {:.4f} sparse {:.4f}'.format(
            name, acc, 1 - acc, rate, sparse))
        return acc

    def train_nri(self, states: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            states: [batch, step, node, dim], observed node states
            adj: [batch, E, K], ground truth interacting relations

        Return:
            loss: cross-entropy of edge classification
        """
        prob = self.model.module.predict_relations(states)
        loss = cross_entropy(prob.view(-1, prob.shape[-1]), adj.transpose(0, 1).flatten())
        self.optimize(self.opt, loss)
        return loss

    def evaluate(self, test):
        """
        Evaluate related metrics to monitor the training process.

        Args:
            test: data set to be evaluted

        Return:
            loss: loss_nll + loss_kl (+ loss_reg) 
            acc: accuracy of relation reconstruction
            rate: rate of assymmetry
            sparse: rate of sparsity in terms of the first type of edge
        """
        acc, rate, sparse, losses = [], [], [], []
        data = self.load_data(test, self.batch_size)
        N = 0.
        with torch.no_grad():
            for adj, states in data:
                if cfg.gpu:
                    adj = adj.cuda()
                    states = states.cuda()
                states_enc = states[:, :cfg.train_steps, :, :]
                prob = self.model.module.predict_relations(states_enc)

                scale = len(states) / self.batch_size
                N += scale

                # use loss as the validation metric
                loss = cross_entropy(prob.view(-1, prob.shape[-1]), adj.transpose(0, 1).flatten())
                # scale all metrics to match the batch size
                loss = loss * scale
                losses.append(loss)

                acc.append(scale * edge_accuracy(prob, adj))
                _, p = prob.max(-1)
                rate.append(scale * asym_rate(p.t(), self.size))
                sparse.append(prob.max(-1)[1].float().mean() * scale)
        loss = sum(losses) / N
        acc = sum(acc) / N
        rate = sum(rate) / N
        sparse = sum(sparse) / N
        return loss, acc, rate, sparse
