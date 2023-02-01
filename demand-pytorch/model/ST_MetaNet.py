import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import dgl
from dgl import DGLGraph
import dgl.function as fn
import numpy as np

MODEL = {'meta_hiddens': [16,2]}

class MLP(nn.Module):
    """ Multilayer perceptron. """
    def __init__(self, hiddens, act_type, out_act, input_dim):
        """
        The initiializer.

        Parameters
        ----------
        hiddens: list
            The list of hidden units of each dense layer.
        act_type: str
            The activation function after each dense layer.
        """
        super(MLP, self).__init__()
        modules = []
        for i, h in enumerate(hiddens):
            modules.append(nn.Linear(input_dim, h))
            input_dim = h
            if act_type == 'sigmoid':
                modules.append(nn.Sigmoid())
            elif act_type == 'relu':
                modules.append(nn.ReLU())          
        self.layer = nn.Sequential(*modules)

    def forward(self, x):
        x = self.layer(x)
        return x
    
    
class MetaDense(nn.Module):
    """ The meta-dense layer. """
    def __init__(self, input_hidden_size, output_hidden_size):
        """
        The initializer.

        Parameters
        ----------
        input_hidden_size: int
            The hidden size of the input.
        output_hidden_size: int
            The hidden size of the output.
        meta_hiddens: list of int
            The list of hidden units of meta learner (a MLP).
        """
        super(MetaDense, self).__init__()
        self.input_hidden_size = input_hidden_size
        self.output_hidden_size = output_hidden_size
        self.act_type = 'sigmoid'

        self.w_mlp = MLP([16, 2, self.input_hidden_size * self.output_hidden_size], act_type=self.act_type, out_act=False, input_dim=32)
        self.b_mlp = MLP([16, 2, 1], act_type=self.act_type, out_act=False, input_dim=32)

    def forward(self, feature, data):
        """ Forward process of a MetaDense layer

        Parameters
        ----------
        feature: NDArray with shape [n, d]
        data: NDArray with shape [n, b, input_hidden_size]

        Returns
        -------
        output: NDArray with shape [n, b, output_hidden_size]
        """

        weight = self.w_mlp(feature) # [n, input_hidden_size * output_hidden_size]
        weight = torch.reshape(weight, (-1, self.input_hidden_size, self.output_hidden_size))
        bias = torch.reshape(self.b_mlp(feature), (-1, 1, 1)) # [n, 1, 1]

        return torch.bmm(data, weight) + bias


class MetaGAT(nn.Module):
    """ Meta Graph Attention. """

    def __init__(self, dist, src, dst, hidden_size):
        super(MetaGAT, self).__init__()
        self.dist = torch.tensor(dist[src, dst])
        self.src = src
        self.dst = dst
        self.hidden_size = hidden_size
        self.num_nodes = dist.shape[0]

        self.w_mlp = MLP( [16, 2, self.hidden_size * self.hidden_size * 2], 'sigmoid', False, 96)

        self.weight = nn.Parameter(torch.tensor([1.0]))

        self.g = None

    def build_graph_on_ctx(self):
        self.g = dgl.DGLGraph()
        self.g.add_nodes(self.num_nodes)
        self.g.add_edges(self.src, self.dst)
        self.g.edata['dist'] = self.dist

    def forward(self, state, feature):
        if self.g is None:
            self.build_graph_on_ctx()

        self.g.ndata['state'] = state.cpu()
        self.g.ndata['feature'] = feature.cpu()
        
        self.g.apply_edges(self.message_function)
        self.g.update_all(self.message_function, self.reduce_function)
        # self.g.update_all(fn.copy_src(src='alpha', out='alpha'), self.reduce_function)
        
        return self.g.ndata.pop('new_state')

    def message_function(self, edges):
        state = torch.cat([edges.src['state'], edges.dst['state']], dim=-1)
        feature = torch.cat([edges.src['feature'], edges.dst['feature'], edges.data['dist']], dim=-1)

        weight = self.w_mlp(feature.cuda())
        weight = weight.reshape(-1, self.hidden_size * 2, self.hidden_size)

        state = torch.reshape(state, (state.shape[0], -1, state.shape[-1]))
        # print(state.shape, weight.shape)
        r = torch.bmm(state.cuda(), weight.cuda())
        alpha = F.leaky_relu(r).cpu()

        return { 'alpha': alpha, 'state': edges.src['state'] }

    def reduce_function(self, nodes):
        state = nodes.mailbox['state']
        alpha = nodes.mailbox['alpha']
        alpha = F.softmax(alpha, dim=1)
        a = torch.sum(alpha * state, dim=1) 
        b = torch.sigmoid(self.weight)
        # print(a.shape, b.shape)

        new_state = (a.cpu() * b.cpu()).relu()
        return { 'new_state': new_state }
    

class MyGRUCell(nn.Module):
    """ A common GRU Cell. """
    def __init__(self,input_size ,hidden_size):
        super(MyGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.cell = nn.GRU(input_size, self.hidden_size, batch_first=True)

    def forward_single(self, feature, data, begin_state):
        # add a temporal axis
        # print(data.shape)
        data = data.unsqueeze(2)

        # unroll
        data, state = self(feature, data, begin_state)


        return data, state

    def forward(self, feature, data, begin_state):
        n, b, length, d = data.shape

        # reshape the data and states for rnn unroll
        data = torch.reshape(data, (n * b, length, -1)) # [n * b, t, d]
        if begin_state is not None:
            begin_state = [
                torch.reshape(state, (n * b, -1)) for state in begin_state
            ] # [n * b, d]
            data, state = self.cell(data.cuda(), begin_state[0].unsqueeze(0)) 
            # 조심
        else:
            data, state = self.cell(data)
        # reshape the data & states back
        data = torch.reshape(data, (n, b* length, -1))
        state = [torch.reshape(s, (n, b, -1)) for s in state]

        return data, state


class MetaGRUCell(nn.Module):
    """ Meta GRU Cell. """

    def __init__(self, pre_hidden_size, hidden_size):
        super(MetaGRUCell, self).__init__()
        self.pre_hidden_size = pre_hidden_size
        self.hidden_size = hidden_size
        self.dense_z = MetaDense(pre_hidden_size + hidden_size, hidden_size)
        self.dense_r = MetaDense(pre_hidden_size + hidden_size, hidden_size)

        self.dense_i2h = MetaDense(pre_hidden_size, hidden_size)
        self.dense_h2h = MetaDense(hidden_size, hidden_size)

    def forward_single(self, feature, data, begin_state):
        """ unroll one step

        Parameters
        ----------
        feature: a NDArray with shape [n, d].
        data: a NDArray with shape [n, b, d].        
        begin_state: a NDArray with shape [n, b, d].
        
        Returns
        -------
        output: ouptut of the cell, which is a NDArray with shape [n, b, d]
        states: a list of hidden states (list of hidden units with shape [n, b, d]) of RNNs.
        
        """
        if begin_state is None:
            num_nodes, batch_size, _= data.shape
            begin_state = [torch.zeros((num_nodes, batch_size, self.hidden_size))]

        prev_state = begin_state[0].cuda()
        # print(data.shape, prev_state.shape)
        data_and_state = torch.concat((data, prev_state), dim=-1)
        z = nn.Sigmoid()(self.dense_z(feature, data_and_state))
        r = nn.Sigmoid()(self.dense_r(feature, data_and_state))

        state = z * prev_state + (1 - z) * torch.tanh(self.dense_i2h(feature, data) + self.dense_h2h(feature, r * prev_state))
        return state, [state]

    def forward(self, feature, data, begin_state):
        num_nodes, batch_size, length = data.shape

        data = torch.split(data, split_size_or_sections=length, dim=2)
        data = [torch.squeeze(d, dim=1) for d in data]

        outputs, state = [], begin_state
        for input in data:
            output, state = self.forward_single(feature, input, state)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=2)
        outputs = torch.reshape(outputs, (1024,-1, 64))
        return outputs, state    

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, graph):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = MyGRUCell(2, 64)
        dist, src, dst = graph
        self.metagat = MetaGAT(dist, src, dst, hidden_size)
        self.metagru = MetaGRUCell(64, 64)
        
    def forward(self, input, feature):
        states = []
        input = torch.reshape(input, (1024, -1, 12, 2))
        output, state = self.gru(feature, input, None)
        states.append(state)
        tmp = torch.reshape(self.metagat(output, feature), output.shape).cuda()
        output = output + tmp
        output, state = self.metagru(feature, output, None)
        states.append(state)
        return states

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, graph, cl_decay_steps):
        super(Decoder, self).__init__()
        self.cl_decay_steps = cl_decay_steps
        self.global_steps = 0.0
        self.hidden_size = hidden_size
        self.proj = nn.Linear(96, output_size)
        dist, src, dst = graph
        self.gru = MyGRUCell(2, 64)
        self.metagat = MetaGAT(dist, src, dst, hidden_size)
        self.metagru = MetaGRUCell(64, 64)
        
    def sampling(self):
        """ Schedule sampling: sampling the ground truth. """
        threshold = self.cl_decay_steps / (self.cl_decay_steps + math.exp(self.global_steps / self.cl_decay_steps))
        return float(random.random() < threshold)        
        
    def forward(self, label, feature, begin_states, is_training):
        label = torch.reshape(label, (1024, -1, 3, 2))
        num_nodes, batch_size, seq_len, _ = label.shape 
        aux = label[:,:,:, 2:] # [n,b,t,d]
        label = label[:,:,:, :2] # [n,b,t,d]
        
        go = torch.zeros((num_nodes, batch_size, 2))
        output, states = [], begin_states
        
        for i in range(seq_len):
            # get next input
            if i == 0: data = go
            else:
                prev = torch.cat((output[i - 1], aux[:,:,i - 1]), dim=-1)
                truth = torch.cat((label[:,:,i - 1], aux[:,:,i - 1]), dim=-1)
                if is_training:
                    value = self.sampling()
                else:
                    value = 0
                data = value * truth + (1 - value) * prev
            # print("data 0",data.shape)
            data, states[0] = self.gru.forward_single(feature, data, states[0])
            tmp = torch.reshape(self.metagat(data, feature), data.shape).cuda()
            data = data + tmp
            if i==0:
                st = torch.reshape(states[1][0], (1024, -1, 12, 64))
                states[1][0] =  torch.mean(st,2,True).squeeze(2)
            
            data, states[1] = self.metagru.forward_single(feature, data, states[1])
        
            # append feature to output
            _feature = feature.unsqueeze(1) # [n, 1, d]
            _feature = _feature.repeat(1, batch_size, 1) # [n, b, d]
            data = torch.cat((data, _feature), dim=-1) # [n, b, t, d]

            # proj output to prediction
            data = data.reshape((num_nodes * batch_size, -1))
            data = self.proj(data)
            data = data.reshape((num_nodes, batch_size, -1))
            
            output.append(data)
        output = torch.stack(output, dim=2)
        return output

class ST_MetaNetModel(nn.Module):
    def __init__(self, graph):
        super(ST_MetaNetModel, self).__init__()
        input_size = 2
        hidden_size = 64
        output_size = 2
        self.fc = nn.Linear(2,64)
        self.encoder = Encoder(input_size, hidden_size, graph)
        self.decoder = Decoder(hidden_size, output_size, graph, 2000)
        self.geo_encoder = MLP([32, 32], act_type='relu',out_act=True,input_dim=989)
        self.graph = graph

    def forward(self, x, feature, label, is_training):
        x = x.transpose(0,1)
        label = label.transpose(0,1)
        feature = self.geo_encoder(feature)
        states = self.encoder(x, feature)
        out = self.decoder(label, feature, states, is_training)
        out = out.transpose(0,1).transpose(1,2)
        return out