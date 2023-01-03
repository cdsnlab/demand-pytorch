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
        
        z2 = F.relu(x)
        z2 = self.bn1(z2)
        z2 = self.plus_conv(z2)
        z2 = z2.reshape((z2.shape[0], self.out_channels, self.H, self.W))
        
        z3 = np.concatenate((z1.cpu().detach().numpy(),z2.cpu().detach().numpy()), axis=1)
        z3 =torch.Tensor(z3).cuda()

        z3 = self.bn2(z3)
        z3 = F.relu(z3)
        z3 = self.drop(z3)
        z3 = self.conv2(z3)

        return z3 + x

class Time_trans(nn.Module):
    def __init__(self, T_F):
        # It takes in a four dimensional input (B, C, lng, lat)
        super(Time_trans, self).__init__()
        self.T_feat = T_F
        self.convm = nn.Conv2d(in_channels=31, out_channels=T_F, kernel_size=1)
        self.convf = nn.Conv2d(in_channels=T_F, out_channels=1, kernel_size=1)

    def forward(self, x):
        
        z = self.convm(x)
        z = F.relu(z)
        z = self.convf(z)
        z = F.relu(z)

        return z
    

class PoI_trans(nn.Module):
    def __init__(self, PoI_N, PT_feat, T_feat):
        # It takes in a four dimensional input (B, C, lng, lat)
        super(PoI_trans, self).__init__()
        self.PoI_N = PoI_N
        self.PT_feat = PT_feat
        self.T_feat = T_feat
        self.time_trans =  Time_trans(T_F= T_feat)
        self.conv = nn.Conv2d(in_channels=PoI_N, out_channels=PT_feat, kernel_size=1)
    
    def forward(self, poi, time):
        T_x = self.time_trans(time)
        T_x = np.tile(T_x.cpu().detach().numpy(), (1, self.PoI_N, 1,1))
        poi_time = T_x * poi.cpu().detach().numpy()
        poi_time =torch.Tensor(poi_time).cuda()
        poi_time = self.conv(poi_time)
        return poi_time
    

class DeepSTNModel(nn.Module):
    def __init__(self, config):
        super(DeepSTNModel, self).__init__()
        self.c = config.c
        self.p = config.p
        self.t = config.t
        self.channel = config.channel
        self.map_heigh = config.heigh
        self.map_width = config.width
        self.RP_N = config.RP_N
        self.PoI_N = config.PoI_N
        self.PT_F = config.PT_F
        self.T_feat = config.T_feat
        
        self.all_channel = self.channel * (self.c+self.p+self.t)
            
        self.cut0 = int( 0 )
        self.cut1 = int( self.cut0 + self.channel*self.c )
        self.cut2 = int( self.cut1 + self.channel*self.p )
        self.cut3 = int( self.cut2 + self.channel*self.t )
        
        self.convc = nn.Conv2d(self.channel*self.c, 64, kernel_size=1, stride = 1, padding =0).cuda()
        self.convp = nn.Conv2d(self.channel*self.p, 64, kernel_size=1, stride = 1, padding =0).cuda()
        self.convt = nn.Conv2d(self.channel*self.t, 64, kernel_size=1, stride = 1, padding =0).cuda()
        
        self.bn1 = nn.BatchNorm2d(64*3 + self.PT_F)
        self.drop = nn.Dropout(0.1)
        self.conv1 = nn.Conv2d(64*3 + self.PT_F, 64, kernel_size=1, stride = 1, padding =0)
        
        self.net = ResPlus(64, 8, self.map_heigh, self.map_width)
        
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 2, kernel_size=1, stride = 1, padding =0)
        
        self.poi_trans = PoI_trans(PoI_N=self.PoI_N, PT_feat=self.PT_F, T_feat=self.T_feat)
        

    def forward(self, x):
        c_input = x[:,self.cut0:self.cut1,:,:].float().cuda()
        p_input = x[:,self.cut1:self.cut2,:,:].float().cuda()
        t_input = x[:,self.cut2:self.cut3,:,:].float().cuda()

        c_input = self.convc(c_input)
        p_input = self.convp(p_input)
        t_input = self.convt(t_input)
        
        poi_in = x[:, -(self.PoI_N + 31):-31,:,:]
        time_in = x[:,-31:,:,:]
        
        poi_time = self.poi_trans(poi_in, time_in)
        
        
        cpt_con1 = np.concatenate((c_input.cpu().detach().numpy(), p_input.cpu().detach().numpy(), t_input.cpu().detach().numpy(), poi_time.cpu().detach().numpy()), axis=1)
        cpt_con1 =torch.Tensor(cpt_con1).cuda()
        cpt = F.relu(cpt_con1)

        cpt = self.bn1(cpt)
        cpt = self.drop(cpt)
        cpt = self.conv1(cpt)        
        
        for _ in range(self.RP_N): cpt = self.net(cpt)
        
        cpt2 = F.relu(cpt)
        cpt2 = self.bn2(cpt2)
        cpt1 = self.drop(cpt2)
        cpt1  = self.conv2(cpt1)
        cpt_out = torch.tanh(cpt1)
        
        return cpt_out
        
    
