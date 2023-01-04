import torch 
import torch.nn as nn
import torch.nn.functional as F

class LocalSeqConv(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.convs = nn.ModuleList([
            nn.Conv2d(input_dim, output_dim, 3, 1, 1) for _ in range(seq_len)
        ])
    
    def forward(self, x):
        # (B, seq_len, C, W, H)
        output = None 
        for i in range(self.seq_len):
            out = F.relu(self.convs[i](x[:, i]))
            if output == None: 
                output = out.unsqueeze(1)
            else:
                output = torch.cat([output, out.unsqueeze(1)], dim=1)
        return output 

class DMVSTNetModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = LocalSeqConv(config.input_dim, config.cnn_hidden, config.num_his)
        self.bn1 = nn.BatchNorm3d(config.cnn_hidden)

        self.conv2 = LocalSeqConv(config.cnn_hidden, config.cnn_hidden, config.num_his)
        self.bn2 = nn.BatchNorm3d(config.cnn_hidden)
        
        self.conv3 = LocalSeqConv(config.cnn_hidden, config.cnn_hidden, config.num_his)

        self.unfold = nn.Unfold(kernel_size=7, padding=3)
        self.spatial_conv = nn.Conv2d(config.cnn_hidden*7*7, 64, 3, 1, 1)
        
        self.lstm = nn.LSTM(input_size=64+config.cnn_hidden, hidden_size=config.lstm_hidden, num_layers=config.lstm_layers, batch_first=True)
        self.topo_dense = nn.Linear(32, 6)

        self.final_dense = nn.Linear(config.lstm_hidden+6, config.output_dim)

    def forward(self, x: torch.Tensor, topo_input):
        x = self.conv1(x)
        x = self.bn1(x.transpose(2, 1)).transpose(2, 1)
        x = self.conv2(x)
        x = self.bn2(x.transpose(2, 1)).transpose(2, 1)
        x = self.conv3(x)

        b, t, c, w, h = x.size()
        x = x.reshape(b*t, c, w, h)
        spatial = self.unfold(x)
        spatial = spatial.reshape(b*t, -1, w, h)
        spatial = self.spatial_conv(spatial)
        
        spatial: torch.Tensor = torch.cat([x, spatial], axis=1)
        spatial = spatial.reshape(b, t, spatial.size(1), w, h)
        spatial = spatial.reshape(b, t, spatial.size(2), w*h)
        spatial = spatial.permute(dims=(0, 3, 1, 2))
        spatial = spatial.reshape(b*spatial.size(1), t, -1)
        
        out, (hn, cn) = self.lstm(spatial)
        hn = hn[-1]

        _, tc, _, _ = topo_input.size()
        topo_input = topo_input.reshape(b, tc, w*h)
        topo_input = topo_input.transpose(2, 1).reshape(b*w*h, tc)
        topo_embed = self.topo_dense(topo_input)
        topo_embed = F.tanh(topo_embed)
        
        out = self.final_dense(torch.cat([hn, topo_embed], dim=1))
        out = out.view(b, w*h, -1)
        out = out.transpose(2, 1)
        out = out.view(b, -1, w, h)
        return out

# x = torch.zeros((4, 12, 2, 32, 32))
# topo_input = torch.zeros((4, 32, 32, 32))
# net = DMVSTNetModel(cfg)
# net.forward(x, topo_input)