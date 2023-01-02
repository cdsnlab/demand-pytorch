from torch import nn
import torch
import torch.nn.functional as F
import logging
import numpy as np

class ResPlus(nn.Module):
    def __init__(self, in_channels, out_channels, H, W):
        # It takes in a four dimensional input (B, C, lng, lat)
        super(ResPlus, self).__init__()
        self.out_channels = out_channels
        self.H = H
        self.W = W
        # self.ln1 = nn.LayerNorm(normalized_shape = (lng, lat))
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv1 = nn.Conv2d(in_channels, in_channels - out_channels, 3, 1, 1)
        # self.ln2 = nn.LayerNorm(normalized_shape = (lng, lat))
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.plus_conv = nn.Conv2d(in_channels, out_channels*H*W, (H,W) ,1, 0)
        
    
    def forward(self, x):
        
        z1 = self.bn1(x)
        z1 = F.relu(z1)
        z1 = self.drop(z1)
        z1 = self.conv1(z1)
        print("z1", z1.shape)
        
        z2 = F.relu(x)
        z2 = self.bn1(z2)
        print("z2", z2.shape)
        z2 = self.plus_conv(z2)
        print("z2", z2.shape)
        z2 = z2.reshape((z2.shape[0], self.out_channels, self.H, self.W))
        print("z2", z2.shape)
        
        z3 = np.concatenate((z1.cpu().detach().numpy(),z2.cpu().detach().numpy()), axis=1)
        z3 =torch.Tensor(z3).cuda()
        print("z3", z3.shape)
        z3 = self.bn2(z3)
        z3 = F.relu(z3)
        z3 = self.drop(z3)
        z3 = self.conv2(z3)
        print("z3", z3.shape)
        return z3 + x

class DeepSTNModel(nn.Module):
    def __init__(self, config):
        super(DeepSTNModel, self).__init__()
        self.c = config.c
        self.p = config.p
        self.t = config.t
        self.channel = config.channel
        self.map_heigh = config.heigh
        self.map_width = config.width
        
        self.all_channel = self.channel * (self.c+self.p+self.t)
            
        self.cut0 = int( 0 )
        self.cut1 = int( self.cut0 + self.channel*self.c )
        self.cut2 = int( self.cut1 + self.channel*self.p )
        self.cut3 = int( self.cut2 + self.channel*self.t )
        
        self.convc = nn.Conv2d(self.channel*self.c, 64, kernel_size=1, stride = 1, padding =0).cuda()
        self.convp = nn.Conv2d(self.channel*self.p, 64, kernel_size=1, stride = 1, padding =0).cuda()
        self.convt = nn.Conv2d(self.channel*self.t, 64, kernel_size=1, stride = 1, padding =0).cuda()
        
        self.bn1 = nn.BatchNorm2d(64*3)
        self.drop = nn.Dropout(0.1)
        self.conv1 = nn.Conv2d(64*3, 64, kernel_size=1, stride = 1, padding =0)
        
        self.net = ResPlus(64, 8, self.map_heigh, self.map_width)
        
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 2, kernel_size=1, stride = 1, padding =0)
        

    def forward(self, x):
        print(x.shape)
        c_input = x[:,self.cut0:self.cut1,:,:].float().cuda()
        p_input = x[:,self.cut1:self.cut2,:,:].float().cuda()
        t_input = x[:,self.cut2:self.cut3,:,:].float().cuda()
        print(c_input.shape)
        c_input = self.convc(c_input)
        p_input = self.convp(p_input)
        t_input = self.convt(t_input)
        print(c_input.shape)
        
        cpt_con1 = np.concatenate((c_input.cpu().detach().numpy(), p_input.cpu().detach().numpy(), t_input.cpu().detach().numpy()), axis=1)
        cpt_con1 =torch.Tensor(cpt_con1).cuda()
        cpt = F.relu(cpt_con1)
        print(cpt.shape)
        cpt = self.bn1(cpt)
        cpt = self.drop(cpt)
        cpt = self.conv1(cpt)
        print(cpt.shape)
        
        cpt = self.net(cpt)
        cpt = self.net(cpt)
        
        cpt2 = F.relu(cpt)
        cpt2 = self.bn2(cpt2)
        cpt1 = self.drop(cpt2)
        cpt1  = self.conv2(cpt1)
        cpt_out = torch.tanh(cpt1)
        
        return cpt_out
        
    
