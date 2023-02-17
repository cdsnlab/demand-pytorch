import torch
import torch.nn as nn
import torch.nn.functional as F

def graph_conv(inputs, supports, dim_in, dim_out, scope='gcn'):
    dtype = inputs.dtype
    num_nodes = inputs.shape[1]
    assert num_nodes == supports.shape[0]
    assert dim_in == inputs.shape[2]
    order = int(supports.shape[1] / num_nodes)

    x_new = inputs.permute(0, 2, 1)
    x_new = x_new.reshape(-1, num_nodes)
    x_new = torch.matmul(x_new, supports)
    x_new = x_new.reshape(-1, dim_in, order, num_nodes)
    x_new = x_new.permute(0, 3, 1, 2)
    x_new = x_new.reshape(-1, order * dim_in)

    with torch.no_grad():
        weights = nn.Parameter(torch.zeros(order * dim_in, dim_out, dtype=dtype))
        biases = nn.Parameter(torch.zeros(dim_out, dtype=dtype))

    outputs = torch.add(torch.matmul(x_new, weights), biases)
    return outputs.reshape(-1, num_nodes, dim_out)

class Conv_ST(nn.Module):
    def __init__(self, kt, dim_in, dim_out, activation='GLU'):
        super().__init__()
        self.kt = kt
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation
        if (dim_in > dim_out):
            self.w_input = nn.Parameter(torch.zeros(1, 1, dim_in, dim_out))
            nn.init.xavier_normal_(self.w_input)
        elif (dim_in < dim_out):
            self.res_input = nn.Sequential(
                nn.Identity(),
                nn.ConstantPad3d((0, 0, 0, 0, 0, dim_out - dim_in), 0)
            )
        else:
            self.res_input = nn.Identity()
  
    def forward(self, inputs, supports):
        T = inputs.size(1)
        num_nodes = inputs.size(2)
        assert inputs.size(3) == self.dim_in
        if (self.dim_in > self.dim_out):
            res_input = F.conv2d(inputs, self.w_input, stride=1, padding=0)
        else:
            res_input = self.res_input(inputs)
        # padding zero
        padding = torch.zeros(inputs.size(0), self.kt - 1, num_nodes, self.dim_in)
        # extract spatial-temporal relationships at the same time
        inputs = torch.cat([padding, inputs], dim=1)
        x_input = torch.stack([inputs[:, i:i + self.kt, :, :] for i in range(0, T)], dim=1)    #[B*T, kt, N, C]
        x_input = x_input.view(-1, self.kt, num_nodes, self.dim_in)
        x_input = x_input.permute(0, 2, 1, 3)

        if (self.activation == 'GLU'):
            conv_out = graph_conv(x_input.view(-1, num_nodes, self.kt * self.dim_in),
                                  supports, self.kt * self.dim_in, 2 * self.dim_out)
            conv_out = conv_out.view(-1, T, num_nodes, 2 * self.dim_out)
            out = (conv_out[:, :, :, :self.dim_out] + res_input) * \
                  torch.sigmoid(conv_out[:, :, :, self.dim_out:2 * self.dim_out])
        if (self.activation == 'sigmoid'):
            conv_out = graph_conv(torch.reshape(x_input, [-1, num_nodes, self.kt * self.dim_in]), supports)
            out = torch.reshape(conv_out, [-1, T, num_nodes, self.dim_out])
        return out

