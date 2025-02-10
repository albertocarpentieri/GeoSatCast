"""
This is a Pytorch implementation of PredRNN++, a recurrent model for video prediction as described in the following paper:  
PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma in Spatiotemporal Predictive Learning, by Yunbo Wang, Zhifeng Gao, 
Mingsheng Long, Jianmin Wang and Philip S. Yu. 

Adapted from https://github.com/zhangyanbiao/predrnn-_pytorch
"""


import torch
import torch.nn as nn

def tensor_layer_norm(num_features):
	return nn.LayerNorm(num_features)

class GHU(nn.Module):
    def __init__(self, layer_name, num_features, tln=False):
        super(GHU,self).__init__()
        """Initialize the Gradient Highway Unit.
        """
        self.layer_name = layer_name
        self.num_features = num_features
        self.layer_norm = tln

        self.bn_z_concat = tensor_layer_norm(self.num_features*2) if tln else None
        self.bn_x_concat = tensor_layer_norm(self.num_features*2) if tln else None

        self.z_concat_conv = nn.Conv2d(self.num_features,self.num_features*2,5,1,2)
        self.x_concat_conv = nn.Conv2d(self.num_features,self.num_features*2,5,1,2)
        nn.init.xavier_uniform_(self.z_concat_conv.weight)
        nn.init.zeros_(self.z_concat_conv.bias)
        nn.init.xavier_uniform_(self.x_concat_conv.weight)
        nn.init.zeros_(self.x_concat_conv.bias)


    def forward(self,x,z):
        if z is None:
            z = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        z_concat = self.z_concat_conv(z)
        if self.layer_norm:
            z_concat = self.bn_z_concat(z_concat.permute(0,2,3,1)).permute(0,3,1,2)

        x_concat = self.x_concat_conv(x)
        if self.layer_norm:
            x_concat = self.bn_x_concat(x_concat.permute(0,2,3,1)).permute(0,3,1,2)

        gates = torch.add(x_concat, z_concat)
        p, u = torch.split(gates, self.num_features, 1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1 - u) * z
        return z_new


class CausalLSTMCell(nn.Module):
    def __init__(
        self, 
        layer_name,
        x_ch,
        num_hidden_in,
        num_hidden_out,
        forget_bias, 
        tln=True):
        super(CausalLSTMCell, self).__init__()
        """Initialize the Causal LSTM cell.
        Args:
            layer_name: layer names for different lstm layers.
            filter_size: int tuple thats the height and width of the filter.
            num_hidden_in: number of units for input tensor.
            num_hidden_out: number of units for output tensor.
            seq_shape: shape of a sequence.
            forget_bias: float, The bias added to forget gates.
            tln: whether to apply tensor layer normalization
        """
        self.layer_name = layer_name
        self.x_ch = x_ch
        self.num_hidden_in = num_hidden_in
        self.num_hidden_out = num_hidden_out
        self.layer_norm = tln
        self._forget_bias = forget_bias

        self.bn_h_cc = tensor_layer_norm(self.num_hidden_out * 4) if tln else None
        self.bn_c_cc = tensor_layer_norm(self.num_hidden_out * 3) if tln else None
        self.bn_m_cc = tensor_layer_norm(self.num_hidden_out * 3) if tln else None
        self.bn_x_cc = tensor_layer_norm(self.num_hidden_out * 7) if tln else None
        self.bn_c2m = tensor_layer_norm(self.num_hidden_out * 4) if tln else None
        self.bn_o_m = tensor_layer_norm(self.num_hidden_out) if tln else None

        self.h_cc_conv = nn.Conv2d(self.num_hidden_out,self.num_hidden_out*4,5,1,2)
        nn.init.xavier_uniform_(self.h_cc_conv.weight)
        nn.init.zeros_(self.h_cc_conv.bias)
        
        self.c_cc_conv = nn.Conv2d(self.num_hidden_out,self.num_hidden_out*3,5,1,2)
        nn.init.xavier_uniform_(self.c_cc_conv.weight)
        nn.init.zeros_(self.c_cc_conv.bias)
        
        self.m_cc_conv = nn.Conv2d(self.num_hidden_in,self.num_hidden_out*3,5,1,2)
        nn.init.xavier_uniform_(self.m_cc_conv.weight)
        nn.init.zeros_(self.m_cc_conv.bias)
        
        self.x_cc_conv = nn.Conv2d(self.x_ch,self.num_hidden_out*7,5,1,2)
        nn.init.xavier_uniform_(self.x_cc_conv.weight)
        nn.init.zeros_(self.x_cc_conv.bias)
        
        self.c2m_conv  = nn.Conv2d(self.num_hidden_out,self.num_hidden_out*4,5,1,2)
        nn.init.xavier_uniform_(self.c2m_conv.weight)
        nn.init.zeros_(self.c2m_conv.bias)
        
        self.o_m_conv = nn.Conv2d(self.num_hidden_out,self.num_hidden_out,5,1,2)
        nn.init.xavier_uniform_(self.o_m_conv.weight)
        nn.init.zeros_(self.o_m_conv.bias)
        
        self.o_conv = nn.Conv2d(self.num_hidden_out, self.num_hidden_out, 5, 1, 2)
        nn.init.xavier_uniform_(self.o_conv.weight)
        nn.init.zeros_(self.o_conv.bias)
        
        self.cell_conv = nn.Conv2d(self.num_hidden_out*2,self.num_hidden_out,1,1,0)
        nn.init.xavier_uniform_(self.cell_conv.weight)
        nn.init.zeros_(self.cell_conv.bias)
        

    def forward(self,x,h,c,m):
        if h is None:
            h = torch.zeros((x.shape[0], self.num_hidden_out, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
        if c is None:
            c = torch.zeros((x.shape[0], self.num_hidden_out, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
        if m is None:
            m = torch.zeros((x.shape[0], self.num_hidden_in, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
        
        h_cc = self.h_cc_conv(h)
        c_cc = self.c_cc_conv(c)
        m_cc = self.m_cc_conv(m)
        
        if self.layer_norm:
            h_cc = self.bn_h_cc(h_cc.permute(0,2,3,1)).permute(0,3,1,2)
            c_cc = self.bn_c_cc(c_cc.permute(0,2,3,1)).permute(0,3,1,2)
            m_cc = self.bn_m_cc(m_cc.permute(0,2,3,1)).permute(0,3,1,2)


        i_h, g_h, f_h, o_h = torch.split(h_cc, self.num_hidden_out, 1)
        i_c, g_c, f_c = torch.split(c_cc, self.num_hidden_out, 1)
        i_m, f_m, m_m = torch.split(m_cc, self.num_hidden_out, 1)
        
        if x is None:
            i = torch.sigmoid(i_h + i_c)
            f = torch.sigmoid(f_h + f_c + self._forget_bias)
            g = torch.tanh(g_h + g_c)
        else:
            x_cc = self.x_cc_conv(x)
            if self.layer_norm:
                x_cc = self.bn_x_cc(x_cc.permute(0,2,3,1)).permute(0,3,1,2)

            i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = torch.split(x_cc,self.num_hidden_out, 1)
            i = torch.sigmoid(i_x + i_h+ i_c)
            f = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
            g = torch.tanh(g_x + g_h + g_c)
        c_new = f * c + i * g
        c2m = self.c2m_conv(c_new)
        if self.layer_norm:
            c2m = self.bn_c2m(c2m.permute(0,2,3,1)).permute(0,3,1,2)

        i_c, g_c, f_c, o_c = torch.split(c2m,self.num_hidden_out, 1)

        if x is None:
            ii = torch.sigmoid(i_c + i_m)
            ff = torch.sigmoid(f_c + f_m + self._forget_bias)
            gg = torch.tanh(g_c)
        else:
            ii = torch.sigmoid(i_c + i_x_ + i_m)
            ff = torch.sigmoid(f_c + f_x_ + f_m + self._forget_bias)
            gg = torch.tanh(g_c + g_x_)
        m_new = ff * torch.tanh(m_m) + ii * gg
        o_m = self.o_m_conv(m_new)

        if self.layer_norm:
             o_m = self.bn_o_m(o_m.permute(0,2,3,1)).permute(0,3,1,2)
        if x is None:
            o = torch.tanh(o_c + o_m)
        else:
            o = torch.tanh(o_x + o_c + o_m)
        o = self.o_conv(o)
        cell = torch.cat([c_new, m_new],1)
        cell = self.cell_conv(cell)
        h_new = o * torch.tanh(cell)
        return h_new, c_new, m_new


class PredRNN(nn.Module):
    def __init__(self, in_steps, in_ch, out_ch, num_hidden, tln=True):
        super(PredRNN, self).__init__()
        
        self.in_steps = in_steps
        self.num_hidden = num_hidden
        self.num_layers = len(self.num_hidden)
        cell_list = []
        ghu_list = []

        for i in range(self.num_layers):
            if i == 0:
                num_hidden_in = self.num_hidden[-1]
                in_ch = in_ch
            else:
                in_ch = num_hidden_in = self.num_hidden[i-1]
            cell_list.append(CausalLSTMCell(
                'lstm_' + str(i + 1),
                in_ch, 
                num_hidden_in,
                num_hidden[i],
                1.0, 
                tln=tln))
        self.cell_list = nn.ModuleList(cell_list)
        
        ghu_list.append(GHU('highway', self.num_hidden[0], tln=tln))
        self.ghu_list = nn.ModuleList(ghu_list)

        self.conv_last = nn.Conv2d(self.num_hidden[-1], out_ch, 1, 1, 0)
        nn.init.xavier_uniform_(self.conv_last.weight)
        nn.init.zeros_(self.conv_last.bias)


    def forward(self, images, num_steps):
        # [batch, length, channel, width, height]
        total_length = num_steps + self.in_steps
        batch = images.shape[0]
        height = images.shape[3]
        width = images.shape[4]

        next_images = []
        h_t = []
        c_t = []
        z_t = None
        m_t = None

        for i in range(self.num_layers):
            h_t.append(None)
            c_t.append(None)

        for t in range(total_length):
            if t < self.in_steps:
                net = images[:,:,t]
            else:
                net = x_gen

            h_t[0], c_t[0], m_t = self.cell_list[0](net, h_t[0], c_t[0], m_t)
            z_t = self.ghu_list[0](h_t[0],z_t)
            h_t[1], c_t[1], m_t = self.cell_list[1](z_t, h_t[1], c_t[1], m_t)
            
            for i in range(2, self.num_layers):
                h_t[i], c_t[i], m_t = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], m_t)

            x_gen = self.conv_last(h_t[self.num_layers-1])
            if t >= self.in_steps:
                next_images.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_images = torch.stack(next_images, dim=2)
        out = next_images
        next_images = []

        return out

if __name__ == "__main__":
    x = torch.randn(1,11,2,256,256)
    predrnn = PredRNN(2, 11, 11, [128,64,64,64],  True)
    print(predrnn)
    predict = predrnn(x, num_steps=1)
    print(predict.shape)
