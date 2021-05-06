import torch
from torch import Tensor
import config as cfg
from torch.nn.functional import mse_loss, one_hot
from instructors.base import Instructor
import numpy as np
from torch.utils.data.dataset import TensorDataset
from torch import optim
from utils.metric import nll_gaussian
from torch.optim.lr_scheduler import StepLR


class XNRIDECIns(Instructor):
    """
    Train the decoder given the ground truth relations.
    """
    def __init__(self, model: torch.nn.DataParallel, data: dict, es: np.ndarray, cmd):
        """
        Args:
            model: an auto-encoder
            data: train / val /test set
            es: edge list
            cmd: command line parameters
        """
        super(XNRIDECIns, self).__init__(cmd)
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
        # use the loss as the metric for model selection, default: +\infty
        val_best = np.inf
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
            losses = self.report('val', [cfg.M])

            val_cur = losses[0]
            if val_cur < val_best:
                # update the current best model when approaching a lower loss
                val_best = val_cur
                torch.save(self.model.module.state_dict(), name)

            # learning rate scheduling
            self.scheduler.step()
        if self.cmd.epochs > 0:
            self.model.module.load_state_dict(torch.load(name))
        _ = self.test('test', 20)

    def report(self, name: str, Ms: list) -> list:
        """
        Evaluate the mean squared errors.

        Args:
            name: 'train' / 'val' / 'test'
            Ms: [...], each element is a number of steps to predict
        
        Return:
            mses: [...], mean squared errors over all steps
        """
        mses = []
        for M in Ms:
            mse, ratio = self.evaluate(self.data[name], M)
            mses.append(mse)
            self.log.info('{} M {:02d} mse {:.3e} ratio {:.4f}'.format(
                name, M, mse, ratio))
        return mses

    def train_nri(self, states: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            states: [batch, step, node, dim], observed node states
            adj: [batch, E, K], ground truth interacting relations

        Return:
            loss: reconstruction loss
        """
        output = self.model.module.predict_states(states, one_hot(adj.transpose(0, 1)).float(), cfg.M)
        loss = nll_gaussian(output, states[:, 1:], 5e-5)
        self.optimize(self.opt, loss * cfg.scale)
        return loss

    def evaluate(self, test, M: int):
        """
        Evaluate related metrics to monitor the training process.

        Args:
            test: data set to be evaluted
            M: number of steps to predict

        Return: 
            mse: mean square error over all steps
            ratio: relative root mean squared error
        """
        mse, ratio = [], []
        data = self.load_data(test, self.batch_size)
        N = 0.
        with torch.no_grad():
            for adj, states in data:
                if cfg.gpu:
                    adj = adj.cuda()
                    states = states.cuda()
                states_dec = states[:, -cfg.train_steps:, :, :]
                target = states_dec[:, 1:]
                
                output = self.model.module.predict_states(states_dec, one_hot(adj.transpose(0, 1)).float(), M)
                # scale all metrics to match the batch size
                scale = len(states) / self.batch_size
                N += scale

                mse.append(scale * mse_loss(output, target).data)
                ratio.append(scale * (((output - target) ** 2).sum(-1).sqrt() / (target ** 2).sum(-1).sqrt()).mean())
        mse = sum(mse) / N
        ratio = sum(ratio) / N
        return mse, ratio

    def test(self, name: str, M: int):
        """
        Evaluate related metrics to measure the model performance.
        The biggest difference between this function and evalute() is that, the mses are evaluated at each step.

        Args:
            name: 'train' / 'val' / 'test'
            M: number of steps to predict

        Return:
            mse_multi: mse at each step
        """
        """
        mses: mean square error over all steps
        ratio: relative root mean squared error
        mse_multi: mse at each step
        """
        mse_multi, mses, ratio = [], [], []
        data = self.load_data(self.data[name], self.batch_size)
        N = 0.
        with torch.no_grad():
            for adj, states in data:
                if cfg.gpu:
                    adj = adj.cuda()
                    states = states.cuda()
                states_dec = states[:, -cfg.train_steps:, :, :]
                target = states_dec[:, 1:]
                
                output = self.model.module.predict_states(states_dec, one_hot(adj.transpose(0, 1)).float(), cfg.M)
                # scale all metrics to match the batch size
                scale = len(states) / self.batch_size
                N += scale

                mses.append(scale * mse_loss(output, target).data)
                ratio.append(scale * (((output - target) ** 2).sum(-1).sqrt() / (target ** 2).sum(-1).sqrt()).mean())

                states_dec = states[:, cfg.train_steps:cfg.train_steps+M+1, :, :]
                target = states_dec[:, 1:]
                
                output = self.model.module.predict_states(states_dec, one_hot(adj.transpose(0, 1)).float(), M)
                mse = ((output - target) ** 2).mean(dim=(0, 2, -1))
                mse *= scale
                mse_multi.append(mse)
        mses = sum(mses) / N
        mse_multi = sum(mse_multi) / N
        ratio = sum(ratio) / N
        self.log.info('{} M {:02d} mse {:.3e} ratio {:.4f}'.format(
                name, M, mses, ratio,))
        msteps = ','.join(['{:.3e}'.format(i) for i in mse_multi])
        self.log.info(msteps)
        return mse_multi
