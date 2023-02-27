# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SpatialTemporalGGCM(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(SpatialTemporalGGCM, self).__init__()
        
#         self.gcn_layer = nn.ModuleList([
#             nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)).to('cuda'),
#             nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)).to('cuda')
#         ])
        
#         self.attention_layer = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, kernel_size=1).to('cuda'),
#             nn.Sigmoid()
#         )
        
#         self.gate_layer = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, kernel_size=1).to('cuda'),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x):
#         # x shape: (batch_size, in_channels, num_nodes, num_frames)
        
#         # Spatial-temporal GCN
#         x = self.gcn_layer[0](x)
#         x = F.relu(x)
#         x = self.gcn_layer[1](x)
        
#         # Spatial-temporal attention
#         a = self.attention_layer(x)
#         x = x * a
        
#         # Spatial-temporal gating
#         g = self.gate_layer(x)
#         x = x * g
        
#         return x


# class STG2SeqModel(nn.Module):
#     def __init__(self, in_channels, num_nodes, num_frames, hidden_size, out_channels):
#         super(STG2SeqModel, self).__init__()
        
#         self.long_term_encoder = nn.ModuleList([
#             SpatialTemporalGGCM(in_channels, hidden_size),
#             SpatialTemporalGGCM(hidden_size, hidden_size),
#             SpatialTemporalGGCM(hidden_size, hidden_size)
#         ])
        
#         self.short_term_encoder = nn.ModuleList([
#             SpatialTemporalGGCM(in_channels, hidden_size),
#             SpatialTemporalGGCM(hidden_size, hidden_size),
#             SpatialTemporalGGCM(hidden_size, hidden_size)
#         ])
        
#         self.output_module = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size).to('cuda'),
#             nn.ReLU(),
#             nn.Linear(hidden_size, out_channels).to('cuda')
#         )
        
#         self.num_nodes = num_nodes
#         self.num_frames = num_frames
        
#     def forward(self, x):
#         # x shape: (batch_size, in_channels, num_nodes, num_frames)
        
#         # Long-term encoding
#         for module in self.long_term_encoder:
#             x = module(x)
#         long_term_output = x
        
#         # Short-term encoding
#         for module in self.short_term_encoder:
#             x = module(x)
#         short_term_output = x
#         # print(long_term_output.shape)
#         # Attention-based output module
#         output = self.output_module(short_term_output.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
#         # attention_weights = F.softmax(torch.bmm(long_term_output.permute(0, 3, 2, 1).reshape(-1, long_term_output.shape[2], long_term_output.shape[3]), short_term_output.permute(0, 3, 1, 2).reshape(-1, long_term_output.shape[2], long_term_output.shape[3])), dim=2)
#         # output = torch.bmm(attention_weights.permute(0, 2, 1), output.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        
#         return output

import torch
import torch.nn as nn
import torch.nn.functional as F

class STG2SeqModel(nn.Module):
    def __init__(self, num_node, in_channel, out_channel):
        super(STG2SeqModel, self).__init__()
        
        self.num_node = num_node
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        # Long-term encoder
        self.encoder_l = nn.Sequential(
            nn.Conv2d(self.in_channel, 64, kernel_size=1).to('cuda'),
            nn.BatchNorm2d(64).to('cuda'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1).to('cuda'),
            nn.BatchNorm2d(64).to('cuda'),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1).to('cuda'),
            nn.BatchNorm2d(128).to('cuda'),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1).to('cuda'),
            nn.BatchNorm2d(256).to('cuda'),
            nn.ReLU()
        )
        
        # Short-term encoder
        self.encoder_s = nn.Sequential(
            nn.Conv2d(self.in_channel, 64, kernel_size=1).to('cuda'),
            nn.BatchNorm2d(64).to('cuda'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1).to('cuda'),
            nn.BatchNorm2d(64).to('cuda'),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1).to('cuda'),
            nn.BatchNorm2d(128).to('cuda'),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1).to('cuda'),
            nn.BatchNorm2d(256).to('cuda'),
            nn.ReLU()
        )
        
        # Hierarchical graph convolutional structure
        self.hgc = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1).to('cuda'),
            nn.BatchNorm2d(256).to('cuda'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1).to('cuda'),
            nn.BatchNorm2d(256).to('cuda'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1).to('cuda'),
            nn.BatchNorm2d(256).to('cuda'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1).to('cuda'),
            nn.BatchNorm2d(256).to('cuda'),
            nn.ReLU()
        )
        
        # Output module
        self.output = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1).to('cuda'),
            nn.BatchNorm2d(128).to('cuda'),
            nn.ReLU(),
            nn.Conv2d(128, self.out_channel, kernel_size=1).to('cuda')
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1).to('cuda'),
            nn.BatchNorm2d(128).to('cuda'),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1).to('cuda'),
            nn.BatchNorm2d(128).to('cuda'),
            nn.ReLU(),
            nn.Conv2d(128, self.out_channel, kernel_size=1).to('cuda'),
            nn.Softmax(dim=2)
        )
        
    def forward(self, x):
        # Long-term encoding
        x_l = self.encoder_l(x)
        x_l = x_l.mean(dim=2, keepdim=True)
        
        # Short-term encoding
        x_s = self.encoder_s(x)
        
        # Concatenating long-term and short-term encodings
        x = torch.cat((x_l.repeat(1, 1, self.num_node, 1), x_s), dim=1)

        # Hierarchical graph convolutional structure
        x = self.hgc(x)
        
        # Output module
        x = self.output(x)
        
        # Attention mechanism
        attn = self.attention(x_s)
        x = x * attn
        return x