from torch import Tensor, nn
from utils.torch_extension import gumbel_softmax, sym_hard
import config as cfg
import torch


class NRIModel(nn.Module):
    """Auto-encoder."""
    def __init__(self, encoder: nn.Module, decoder: nn.Module, es: Tensor, size: int):
        """
        Args:
            encoder: an encoder inferring relations
            decoder: an decoder predicting future states
            es: edge list
            size: number of nodes
        """
        super(NRIModel, self).__init__()
        self.enc = encoder
        self.dec = decoder
        self.es = torch.LongTensor(es)
        self.size = size

    def predict_relations(self, states: Tensor) -> Tensor:
        """
        Given historical node states, infer interacting relations.

        Args:
            states: [batch, step, node, dim]

        Return:
            prob: [E, batch, K]
        """
        if not self.es.is_cuda:
            self.es = self.es.cuda(states.device)
        logits = self.enc(states, self.es)
        prob = logits.softmax(-1)
        return prob

    def predict_states(self, states: Tensor, edges: Tensor, M: int=1) -> Tensor:
        """
        Given historical node states and inferred relations, predict future node states.

        Args:
            states: [batch, step, node, dim]
            edges: [E, batch, K]
            M: number of steps to predict

        Return:
            states: [batch, step_out, node, dim]
        """
        if not self.es.is_cuda:
            self.es = self.es.cuda(states.device)
        return self.dec(states, edges, self.es, M)

    def forward(self, states_enc: Tensor, states_dec: Tensor, hard: bool=False, p: bool=False, M: int=1, tosym: bool=False):
        """
        Args:
            states_enc: [batch, step_enc, node, dim], input node states for the encoder
            states_dec: [batch, step_dec, node, dim], input node states for the decoder
            hard: predict one-hot representation of relations or its continuous relaxation, default: False
            p: return the distribution of relations or not, default: True
            M: number of steps to predict
            tosym: impose hard constraint to inferred relations or not, default: False

        Return:
            output: [batch, step, node, dim]
            prob (optioanl): [E, batch, K]
        """
        if not self.es.is_cuda:
            self.es = self.es.cuda(states_enc.device)
        logits = self.enc(states_enc, self.es)
        if tosym:
            logits = sym_hard(logits, self.size)
        edges = gumbel_softmax(logits, tau=cfg.temp, hard=hard)
        output = self.dec(states_dec, edges, self.es, M)
        if p:
            prob = logits.softmax(-1)
            prob = prob.transpose(0, 1).contiguous()
            return output, prob
        else:
            return output
