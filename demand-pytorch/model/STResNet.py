import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import torch.optim as optim
import time

class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, lng, lat):
        # It takes in a four dimensional input (B, C, lng, lat)
        super(ResUnit, self).__init__()
        # self.ln1 = nn.LayerNorm(normalized_shape = (lng, lat))
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        # self.ln2 = nn.LayerNorm(normalized_shape = (lng, lat))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
    
    def forward(self, x):
        # z = self.ln1(x)
        z = self.bn1(x)
        z = F.relu(z)
        z = self.conv1(z)
        # z = self.ln2(z)
        z = self.bn2(z)
        z = F.relu(z)
        z = self.conv2(z)
        return z + x


class STResNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.len_closeness = config.len_closeness
        self.len_period = config.len_period
        self.len_trend = config.len_trend
        self.external_dim = config.external_dim
        self.map_heigh = config.map_heigh
        self.map_width = config.map_width
        self.nb_flow = config.nb_flow
        self.nb_residual_unit = config.nb_residual_unit
        self._build_stresnet()


    def _build_stresnet(self, ):
        branches = ['c', 'p', 't']
        self.c_net = nn.ModuleList([
            nn.Conv2d(self.len_closeness * self.nb_flow, 64, kernel_size=3, stride = 1, padding = 1), 
        ])
        for i in range(self.nb_residual_unit):
            self.c_net.append(ResUnit(64, 64, self.map_heigh, self.map_width))
        self.c_net.append(nn.Conv2d(64, self.nb_flow, kernel_size=3, stride = 1, padding = 1))

        self.p_net = nn.ModuleList([
            nn.Conv2d(self.len_period * self.nb_flow, 64, kernel_size=3, stride = 1, padding = 1), 
        ])
        for i in range(self.nb_residual_unit):
            self.p_net.append(ResUnit(64, 64, self.map_heigh, self.map_width))
        self.p_net.append(nn.Conv2d(64, self.nb_flow, kernel_size=3, stride = 1, padding = 1))

        self.t_net = nn.ModuleList([
            nn.Conv2d(self.len_trend * self.nb_flow, 64, kernel_size=3, stride = 1, padding = 1), 
        ])
        for i in range(self.nb_residual_unit):
            self.t_net.append(ResUnit(64, 64, self.map_heigh, self.map_width))
        self.t_net.append(nn.Conv2d(64, self.nb_flow, kernel_size=3, stride = 1, padding = 1))

        self.ext_net = nn.Sequential(
            nn.Linear(self.external_dim, 10), 
            nn.ReLU(inplace = True),
            nn.Linear(10, self.nb_flow * self.map_heigh * self.map_width)
        )
        self.w_c = nn.Parameter(torch.rand((self.nb_flow, self.map_heigh, self.map_width)), requires_grad=True)
        self.w_p = nn.Parameter(torch.rand((self.nb_flow, self.map_heigh, self.map_width)), requires_grad=True)
        self.w_t = nn.Parameter(torch.rand((self.nb_flow, self.map_heigh, self.map_width)), requires_grad=True)



    def forward_branch(self, branch, x_in):
        for layer in branch:
            x_in = layer(x_in)
        return x_in

    def forward(self, xc, xp, xt, ext):
        c_out = self.forward_branch(self.c_net, xc)
        p_out = self.forward_branch(self.p_net, xp)
        t_out = self.forward_branch(self.t_net, xt)
        ext_out = self.ext_net(ext).view([-1, self.nb_flow, self.map_heigh, self.map_width])
        # FUSION
        res = self.w_c.unsqueeze(0) * c_out + \
                self.w_p.unsqueeze(0) * p_out + \
                self.w_t.unsqueeze(0) * t_out
        res += ext_out
        return torch.tanh(res)