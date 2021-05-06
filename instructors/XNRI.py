import torch
import config as cfg
from torch.nn.functional import mse_loss
from utils.torch_extension import edge_accuracy, asym_rate, transpose
from instructors.base import Instructor
import numpy as np
from torch.utils.data.dataset import TensorDataset
from torch import Tensor, optim
from utils.metric import cross_entropy, kl_divergence, nll_gaussian
from torch.optim.lr_scheduler import StepLR


class XNRIIns(Instructor):
    """
    Training and testing for the neural relational inference task.
    """
    def __init__(self, model: torch.nn.DataParallel, data: dict, es: np.ndarray, cmd):
        """
        Args:
            model: an auto-encoder
            data: train / val /test set
            es: edge list
            cmd: command line parameters
        """
        super(XNRIIns, self).__init__(cmd)
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
            for _, states in data:
                if cfg.gpu:
                    states = states.cuda()
                scale = len(states) / self.batch_size
                # N: number of samples, equal to the batch size with possible exception for the last batch
                N += scale
                loss_a += scale * self.train_nri(states)
            loss_a /= N 
            self.log.info('epoch {:03d} loss {:.3e}'.format(epoch, loss_a))
            losses = self.report('val', [cfg.M])

            val_cur = losses[0]
            if val_cur < val_best:
                # update the current best model when approaching a lower loss
                self.log.info('epoch {:03d} metric {:.3e}'.format(epoch, val_cur))
                val_best = val_cur
                torch.save(self.model.module.state_dict(), name)

            # learning rate scheduling
            self.scheduler.step()
        if self.cmd.epochs > 0:
            self.model.module.load_state_dict(torch.load(name))
        self.test('test', 20)

    def report(self, name: str, Ms: list) -> list:
        """
        Evaluate the loss.

        Args:
            name: 'train' / 'val' / 'test'
            Ms: [...], each element is a number of steps to predict
        
        Return:
            losses: [...], each element is an average loss
        """
        losses = []
        for M in Ms:
            loss, mse, acc, rate, ratio, sparse = self.evalate(self.data[name], M)
            losses.append(loss)
            self.log.info('{} M {:02d} mse {:.3e} acc {:.4f} _acc {:.4f} rate {:.4f} ratio {:.4f} sparse {:.4f}'.format(
                name, M, mse, acc, 1 - acc, rate, ratio, sparse))
        return losses

    def train_nri(self, states: Tensor) -> Tensor:
        """
        Args:
            states: [batch, step, node, dim], all node states, including historical states and the states to predict
        """
        # compute the relation distribution (prob) and predict future node states (output)
        output, prob = self.model(states, states, p=True, M=cfg.M, tosym=cfg.sym)
        prob = prob.transpose(0, 1).contiguous()
        # reconstruction loss and the KL-divergence
        loss_nll = nll_gaussian(output, states[:, 1:], 5e-5)
        loss_kl = cross_entropy(prob, prob) / (prob.shape[1] * self.size)
        loss = loss_nll + loss_kl
        # impose the soft symmetric contraint by adding a regularization term
        if self.cmd.reg > 0:
            # transpose the relation distribution
            prob_hat = transpose(prob, self.size)
            loss_sym = kl_divergence(prob_hat, prob) / (prob.shape[1] * self.size)
            loss = loss + loss_sym * self.cmd.reg
        self.optimize(self.opt, loss * cfg.scale)

        # choice for the evaluation metric, adding the regularization term or not
        # when the penalty factor is large, it may be misleading to add the regularization term
        if self.cmd.no_reg:
            loss = loss_nll + loss_kl
        return loss

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
        acc: accuracy of relation reconstruction
        mses: mean square error over all steps
        rate: rate of assymmetry
        ratio: relative root mean squared error
        sparse: rate of sparsity in terms of the first type of edge
        losses: loss_nll + loss_kl (+ loss_reg) 
        mse_multi: mse at each step
        """
        acc, mses, rate, ratio, sparse, losses, mse_multi = [], [], [], [], [], [], []
        data = self.load_data(self.data[name], self.batch_size)
        N = 0.
        with torch.no_grad():
            for adj, states in data:
                if cfg.gpu:
                    adj = adj.cuda()
                    states = states.cuda()
                states_enc = states[:, :cfg.train_steps, :, :]
                states_dec = states[:, -cfg.train_steps:, :, :]
                target = states_dec[:, 1:]
                
                output, prob = self.model(states_enc, states_dec, hard=True, p=True, M=M, tosym=cfg.sym)
                prob = prob.transpose(0, 1).contiguous()

                scale = len(states) / self.batch_size
                N += scale

                # use loss as the validation metric
                loss_nll = nll_gaussian(target, output, 5e-5)
                loss_kl = cross_entropy(prob, prob) / (prob.shape[1] * self.size)
                loss = loss_nll + loss_kl
                if self.cmd.reg > 0 and not self.cmd.no_reg:
                    prob_hat = transpose(prob, self.size)
                    loss_sym = kl_divergence(prob_hat, prob) / (prob.shape[1] * self.size)
                    loss = loss + loss_sym * self.cmd.reg

                # scale all metrics to match the batch size
                loss = loss * scale
                losses.append(loss)

                mses.append(scale * mse_loss(output, target).data)
                ratio.append(scale * (((output - target) ** 2).sum(-1).sqrt() / (target ** 2).sum(-1).sqrt()).mean())
                acc.append(scale * edge_accuracy(prob, adj))
                _, p = prob.max(-1)
                rate.append(scale * asym_rate(p.t(), self.size))
                sparse.append(prob.max(-1)[1].float().mean() * scale)

                states_dec = states[:, cfg.train_steps:cfg.train_steps+M+1, :, :]
                target = states_dec[:, 1:]
                
                output, prob = self.model(states_enc, states_dec, hard=True, p=True, M=M, tosym=cfg.sym)
                prob = prob.transpose(0, 1).contiguous()
                mse = ((output - target) ** 2).mean(dim=(0, 2, -1))
                mse *= scale
                mse_multi.append(mse)
        loss = sum(losses) / N
        mses = sum(mses) / N
        mse_multi = sum(mse_multi) / N
        acc = sum(acc) / N
        rate = sum(rate) / N
        ratio = sum(ratio) / N
        sparse = sum(sparse) / N
        self.log.info('{} M {:02d} mse {:.3e} acc {:.4f} _acc {:.4f} rate {:.4f} ratio {:.4f} sparse {:.4f}'.format(
                name, M, mses, acc, 1 - acc, rate, ratio, sparse))
        msteps = ','.join(['{:.3e}'.format(i) for i in mse_multi])
        self.log.info(msteps)
        return mse_multi

    def evalate(self, test, M: int):
        """
        Evaluate related metrics to monitor the training process.

        Args:
            test: data set to be evaluted
            M: number of steps to predict

        Return:
            loss: loss_nll + loss_kl (+ loss_reg) 
            mse: mean square error over all steps
            acc: accuracy of relation reconstruction
            rate: rate of assymmetry
            ratio: relative root mean squared error
            sparse: rate of sparsity in terms of the first type of edge
        """
        acc, mse, rate, ratio, sparse, losses = [], [], [], [], [], []
        data = self.load_data(test, self.batch_size)
        N = 0.
        with torch.no_grad():
            for adj, states in data:
                if cfg.gpu:
                    adj = adj.cuda()
                    states = states.cuda()
                states_enc = states[:, :cfg.train_steps, :, :]
                states_dec = states[:, -cfg.train_steps:, :, :]
                target = states_dec[:, 1:]
                
                output, prob = self.model(states_enc, states_dec, hard=True, p=True, M=M, tosym=cfg.sym)
                prob = prob.transpose(0, 1).contiguous()

                scale = len(states) / self.batch_size
                N += scale

                # use loss as the validation metric
                loss_nll = nll_gaussian(target, output, 5e-5)
                loss_kl = cross_entropy(prob, prob) / (prob.shape[1] * self.size)
                loss = loss_nll + loss_kl
                if self.cmd.reg > 0 and not self.cmd.no_reg:
                    prob_hat = transpose(prob, self.size)
                    loss_sym = kl_divergence(prob_hat, prob) / (prob.shape[1] * self.size)
                    loss = loss + loss_sym * self.cmd.reg
                # scale all metrics to match the batch size
                loss = loss * scale
                losses.append(loss)

                mse.append(scale * mse_loss(output, target).data)
                ratio.append(scale * (((output - target) ** 2).sum(-1).sqrt() / (target ** 2).sum(-1).sqrt()).mean())
                acc.append(scale * edge_accuracy(prob, adj))
                _, p = prob.max(-1)
                rate.append(scale * asym_rate(p.t(), self.size))
                sparse.append(prob.max(-1)[1].float().mean() * scale)
        loss = sum(losses) / N
        mse = sum(mse) / N
        acc = sum(acc) / N
        rate = sum(rate) / N
        ratio = sum(ratio) / N
        sparse = sum(sparse) / N
        return loss, mse, acc, rate, ratio, sparse
