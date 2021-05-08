from torch import Tensor, nn
import torch
from torch.autograd import Variable
from models.gnn import GNN
from models.base import LinAct
import config as cfg


class GRUX(nn.Module):
    """
    GRU from https://github.com/ethanfetaya/NRI/modules.py.
    """
    def __init__(self, dim_in: int, dim_hid: int, bias: bool=True):
        """
        Args:
            dim_in: input dimension
            dim_hid: dimension of hidden layers
            bias: adding a bias term or not, default: True
        """
        super(GRUX, self).__init__()
        self.hidden = nn.ModuleList([
            nn.Linear(dim_hid, dim_hid, bias)
            for _ in range(3)
        ])
        self.input = nn.ModuleList([
            nn.Linear(dim_in, dim_hid, bias)
            for _ in range(3)
        ])

    def forward(self, inputs: Tensor, hidden: Tensor, state: Tensor=None) -> Tensor:
        """
        Args:
            inputs: [..., dim]
            hidden: [..., dim]
            state: [..., dim], default: None
        """
        r = torch.sigmoid(self.input[0](inputs) + self.hidden[0](hidden))
        i = torch.sigmoid(self.input[1](inputs) + self.hidden[1](hidden))
        n = torch.tanh(self.input[2](inputs) + r * self.hidden[2](hidden))
        if state is None:
            state = hidden
        output = (1 - i) * n + i * state
        return output


class GNNDEC(GNN):
    """
    MLPDecoder of NRI from https://github.com/ethanfetaya/NRI/modules.py.
    """
    def __init__(self, n_in_node: int, edge_types: int,
                 msg_hid: int, msg_out: int, n_hid: int,
                 do_prob: float=0., skip_first: bool=False):
        """
        Args:
            n_in_node: input dimension
            edge_types: number of edge types
            msg_hid, msg_out, n_hid: dimension of different hidden layers
            do_prob: rate of dropout, default: 0
            skip_first: setting the first type of edge as non-edge or not, if yes, the first type of edge will have no effect, default: False
        """
        super(GNNDEC, self).__init__()
        self.msgs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * n_in_node, msg_hid),
                nn.ReLU(),
                nn.Dropout(do_prob),
                nn.Linear(msg_hid, msg_out),
                nn.ReLU(),
            )
            for _ in range(edge_types)
        ])

        self.out = nn.Sequential(
            nn.Linear(n_in_node + msg_out, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_in_node)
        )

        self.msg_out = msg_out
        self.skip_first = skip_first

    def move(self, x: Tensor, es: Tensor, z: Tensor) -> Tensor:
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
            z: [E, batch, K]

        Return:
            x: [node, batch, step, dim], future node states
        """
        # z: [E, batch, K] -> [E, batch, step, K]
        z = z.repeat(x.size(2), 1, 1, 1).permute(1, 2, 0, 3).contiguous()
        msg, col, size = self.message(x, es)
        idx = 1 if self.skip_first else 0
        norm = len(self.msgs)
        if self.skip_first:
            norm -= 1
        msgs = sum(self.msgs[i](msg) * torch.select(z, -1, i).unsqueeze(-1) / norm
                   for i in range(idx, len(self.msgs)))
        # aggregate all msgs from the incoming edges
        msgs = self.aggregate(msgs, col, size, 'add')
        # skip connection
        h = torch.cat([x, msgs], dim=-1)
        # predict the change in states
        delta = self.out(h)
        return x + delta

    def forward(self, x: Tensor, z: Tensor, es: Tensor, M: int=1) -> Tensor:
        """
        Args:
            x: [batch, step, node, dim], historical node states
            z: [E, batch, K], distribution of edge types
            es: [2, E], edge list
            M: number of steps to predict

        Return:
            future node states
        """
        # x: [batch, step, node, dim] -> [node, batch, step, dim]
        x = x.permute(2, 0, 1, -1).contiguous()
        assert (M <= x.shape[2])
        # only take m-th timesteps as starting points (m: pred_steps)
        x_m = x[:, :, 0::M, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
        # predict M steps.
        xs = []
        for _ in range(0, M):
            x_m = self.move(x_m, es, z)
            xs.append(x_m)

        node, batch, skip, dim = xs[0].shape
        sizes = [node, batch, skip * M, dim]
        x_hat = Variable(torch.zeros(sizes))
        if x.is_cuda:
            x_hat = x_hat.cuda()
        # re-assemble correct timeline
        for i in range(M):
            x_hat[:, :, i::M, :] = xs[i]
        x_hat = x_hat.permute(1, 2, 0, -1).contiguous()
        return x_hat[:, :(x.size(2) - 1)]


class RNNDEC(GNN):
    """
    RNN decoder with spatio-temporal message passing mechanisms.
    """
    def __init__(self, n_in_node: int, edge_types: int,
                 msg_hid: int, msg_out: int, n_hid: int,
                 do_prob: float=0., skip_first: bool=False, option='both'):
        """
        Args:
            n_in_node: input dimension
            edge_types: number of edge types
            msg_hid, msg_out, n_hid: dimension of different hidden layers
            do_prob: rate of dropout, default: 0
            skip_first: setting the first type of edge as non-edge or not, if yes, the first type of edge will have no effect, default: False
            option: default: 'both'
                'both': using both node-level and edge-level spatio-temporal message passing operations
                'node': using node-level the spatio-temporal message passing operation
                'edge': using edge-level the spatio-temporal message passing operation
        """
        super(RNNDEC, self).__init__()
        self.option = option
        self.msgs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * n_in_node, msg_hid),
                nn.ReLU(),
                nn.Dropout(do_prob),
                nn.Linear(msg_hid, msg_out),
                nn.ReLU(),
            )
            for _ in range(edge_types)
        ])

        self.out = nn.Sequential(
            nn.Linear(n_in_node + msg_out, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_in_node)
        )
        self.gru_edge = GRUX(n_hid, n_hid)
        self.gru_node = GRUX(n_hid + n_in_node, n_hid + n_in_node)
        self.msg_out = msg_out
        self.skip_first = skip_first
        print('Using learned interaction net decoder.')

    def move(self, x: Tensor, es: Tensor, z: Tensor, h_node: Tensor=None, h_edge: Tensor=None):
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
            z: [E, batch, K]
            h_node: [node, batch, step, dim], hidden states of nodes, default: None
            h_edge: [E, batch, step, dim], hidden states of edges, default: None

        Return:
            x: [node, batch, step, dim], future node states
            msgs: [E, batch, step, dim], hidden states of edges
            cat: [node, batch, step, dim], hidden states of nodes
        """
        # z: [E, batch, K] -> [E, batch, step, K]
        z = z.repeat(x.size(2), 1, 1, 1).permute(1, 2, 0, 3).contiguous()
        msg, col, size = self.message(x, es)
        idx = 1 if self.skip_first else 0
        norm = len(self.msgs)
        if self.skip_first:
            norm -= 1
        msgs = sum(self.msgs[i](msg) * torch.select(z, -1, i).unsqueeze(-1) / norm
                   for i in range(idx, len(self.msgs)))
        if h_edge is not None and self.option in {'edge', 'both'}:
            msgs = self.gru_edge(msgs, h_edge)
        # aggregate all msgs from the incoming edges
        msg = self.aggregate(msgs, col, size)
        # skip connection
        cat = torch.cat([x, msg], dim=-1)
        if h_node is not None and self.option in {'node', 'both'}:
            cat = self.gru_node(cat, h_node)
        delta = self.out(cat)
        if self.option == 'node':
            msgs = None
        if self.option == 'edge':
            cat = None
        return x + delta, cat, msgs

    def forward(self, x: Tensor, z: Tensor, es: Tensor, M: int=1) -> Tensor:
        """
        Args:
            x: [batch, step, node, dim], historical node states
            z: [E, batch, K], distribution of edge types
            es: [2, E], edge list
            M: number of steps to predict

        Return:
            future node states
        """
        # x: [batch, step, node, dim] -> [node, batch, step, dim]
        x = x.permute(2, 0, 1, -1).contiguous()
        assert (M <= x.shape[2])
        # only take m-th timesteps as starting points (m: pred_steps)
        x_m = x[:, :, 0::M, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
        # predict m steps.
        xs = []
        h_node, h_edge = None, None
        for _ in range(0, M):
            x_m, h_node, h_edge = self.move(x_m, es, z, h_node, h_edge)
            xs.append(x_m)

        node, batch, skip, dim = xs[0].shape
        sizes = [node, batch, skip * M, dim]
        x_hat = Variable(torch.zeros(sizes))
        if x.is_cuda:
            x_hat = x_hat.cuda()
        # Re-assemble correct timeline
        for i in range(M):
            x_hat[:, :, i::M, :] = xs[i]
        x_hat = x_hat.permute(1, 2, 0, -1).contiguous()
        return x_hat[:, :(x.size(2) - 1)]


class AttDEC(GNN):
    """
    Spatio-temporal message passing mechanisms implemented by combining RNNs and attention mechanims.
    """
    def __init__(self, n_in_node: int, edge_types: int,
                 msg_hid: int, msg_out: int, n_hid: int,
                 do_prob: float=0., skip_first: bool=False):
        """
        Args:
            n_in_node: input dimension
            edge_types: number of edge types
            msg_hid, msg_out, n_hid: dimension of different hidden layers
            do_prob: rate of dropout, default: 0
            skip_first: setting the first type of edge as non-edge or not, if yes, the first type of edge will have no effect, default: False
        """
        super(AttDEC, self).__init__()
        self.input_emb = nn.Linear(n_in_node, cfg.input_emb_hid)
        self.msgs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * (n_in_node + cfg.input_emb_hid), msg_hid),
                nn.ReLU(),
                nn.Dropout(do_prob),
                nn.Linear(msg_hid, msg_out),
                nn.ReLU(),
            )
            for _ in range(edge_types)
        ])

        self.out = nn.Sequential(
            nn.Linear(n_in_node + msg_out, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_in_node)
        )
        self.gru_edge = GRUX(msg_out, msg_out)
        self.gru_node = GRUX(n_in_node + msg_out, n_in_node + msg_out)
        self.msg_out = msg_out
        self.skip_first = skip_first

        # attention mechanism
        self.attn = nn.Linear(msg_out + n_in_node, cfg.att_hid)
        self.query = LinAct(n_in_node + msg_out, cfg.att_hid)
        self.key = LinAct(n_in_node + msg_out, cfg.att_hid)
        self.value = LinAct(n_in_node + msg_out, cfg.att_hid)
        self.att_out = LinAct(cfg.att_hid, n_in_node + msg_out)

    def temporalAttention(self, x: Tensor, h: Tensor):
        """
        Update hidden states of nodes by the temporal attention mechanism.

        Args:
            x: [step_att, node, batch, step, dim], historical hidden states of nodes used for temporal attention
            h: [node, batch, step, dim], hidden states of nodes from RNNs
        
        Return:
            output: [node, batch, step, dim], hidden states of nodes updated by the attention mechanism
            out_att: [batch, node, step, step_att], attentions of historical steps w.r.t. current step
        """
        # concatenate current hidden states of nodes with historical hidden states
        h_current = h.unsqueeze(0).contiguous()
        x = torch.cat([x, h_current], dim=0)

        # x: [step_att, node, batch, step, dim] -> [node, batch, step, step_att, dim]
        x = x.permute(1, 2, 3, 0, 4).contiguous()
        # [node, batch, step, 1, att_hid]
        query = self.query(h.unsqueeze(3))
        # [node, batch, step, step_att, att_hid]
        key = self.key(x)
        value = self.value(x)
        # key: [node, batch, step, step_att, att_hid] -> [node, batch, step, att_hid, step_att]
        key = key.transpose(-1, -2).contiguous()

        # [node, batch, step, 1, step_att]
        attention = torch.matmul(query, key) / (cfg.att_hid ** 0.5)
        attention = attention.softmax(-1)
        # [node, batch, step, att_hid]
        att_value = torch.matmul(attention, value).squeeze(3)
        output = self.att_out(att_value)

        # [batch, node, step, step_att]
        out_att = attention.squeeze(3).transpose(0, 1).contiguous()  
        return output, out_att

    def move(self, x, es, z, h_att, h_node=None, h_edge=None):
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
            z: [E, batch, K]
            h_att: [step_att, node, batch, step, dim], historical hidden states of nodes used for temporal attention
            h_node: [node, batch, step, dim], hidden states of nodes, default: None
            h_edge: [E, batch, step, dim], hidden states of edges, default: None

        Return:
            x: [node, batch, step, dim], future node states
            h_att: [step_att + 1, node, batch, step, dim], accumulated historical hidden states of nodes used for temporal attention
            msgs: [E, batch, step, dim], hidden states of edges
            cat: [node, batch, step, dim], hidden states of nodes
        """
        x_emb = self.input_emb(x)
        x_emb = torch.cat([x_emb, x], dim=-1)
        # z: [E, batch, K] -> [E, batch, step, K]
        z = z.repeat(x.size(2), 1, 1, 1).permute(1, 2, 0, 3).contiguous()
        msg, col, size = self.message(x_emb, es)
        idx = 1 if self.skip_first else 0
        norm = len(self.msgs)
        if self.skip_first:
            norm -= 1
        msgs = sum(self.msgs[i](msg) * torch.select(z, -1, i).unsqueeze(-1) / norm
                   for i in range(idx, len(self.msgs)))

        if h_edge is not None:
            msgs = self.gru_edge(msgs, h_edge)
        # aggregate all msgs to receiver
        msg = self.aggregate(msgs, col, size)

        cat = torch.cat([x, msg], dim=-1)
        if h_node is None:
            delta = self.out(cat)
            h_att = cat.unsqueeze(0)
        else:
            cat = self.gru_node(cat, h_node)
            cur_hidden, _ = self.temporalAttention(h_att, cat)
            h_att = torch.cat([h_att, cur_hidden.unsqueeze(0)], dim=0)
            delta = self.out(cur_hidden)

        return x + delta, h_att, cat, msgs

    def forward(self, x: Tensor, z: Tensor, es: Tensor, M: int=1) -> Tensor:
        """
        Args:
            x: [batch, step, node, dim], historical node states
            z: [E, batch, K], distribution of edge types
            es: [2, E], edge list
            M: number of steps to predict

        Return:
            future node states
        """
        # x: [batch, step, node, dim] -> [node, batch, step, dim]
        x = x.permute(2, 0, 1, -1).contiguous()
        assert (M <= x.shape[2])
        # only take m-th timesteps as starting points (m: pred_steps)
        x_m = x[:, :, 0::M, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
        # predict M steps.
        xs = []
        att_hidden, edge_hidden, node_hidden = None, None, None

        for _ in range(0, M):
            x_m, att_hidden, node_hidden, edge_hidden = self.move(x_m, es, z, att_hidden, node_hidden, edge_hidden)
            xs.append(x_m)

        node, batch, skip, dim = xs[0].shape
        sizes = [node, batch, skip * M, dim]
        x_hat = Variable(torch.zeros(sizes))

        if x.is_cuda:
            x_hat = x_hat.cuda()
        # re-assemble correct timeline
        for i in range(M):
            x_hat[:, :, i::M, :] = xs[i]
        x_hat = x_hat.permute(1, 2, 0, -1).contiguous()
        return x_hat[:, :(x.size(2) - 1)]
