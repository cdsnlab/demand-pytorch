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