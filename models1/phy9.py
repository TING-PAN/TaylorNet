import torch
import torch.nn as nn


class PhyCell9(nn.Module):
    def __init__(self, device, input_dim = 64, F_hidden_dim = 49, kernel_size = (7, 7), bias = 1):
        super(PhyCell9, self).__init__()
        self.input_dim = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.device = device

        self.F = nn.Sequential()
        self.F.add_module('conv1', nn.Conv2d(in_channels = input_dim, out_channels = F_hidden_dim, kernel_size = self.kernel_size, stride = (1, 1), padding = self.padding))
        self.F.add_module('bn1', nn.GroupNorm(7, F_hidden_dim))
        self.F.add_module('f_act1', nn.LeakyReLU(negative_slope = 0.1))
        self.F.add_module('conv2', nn.Conv2d(in_channels = F_hidden_dim, out_channels = input_dim, kernel_size = (1, 1), stride = (1, 1), padding = (0, 0)))

        self.on_gru = ON_ConvGRU(hidden_dim = input_dim, input_dim = input_dim)

        self.convgate = nn.Conv2d(in_channels = self.input_dim + self.input_dim,
                                  out_channels = self.input_dim,
                                  kernel_size = (3, 3),
                                  padding = (1, 1), bias = self.bias)

        self.h0, self.h1, self.h2, self.time = 0, 0, 0, 0
        self.E_tilde = 0

    def forward(self, Ep, first_timestep = False):  # x [batch_size, hidden_dim, height, width]

        if first_timestep:
            self.h0 = Ep
            self.h1 = self.F(self.h0)
            self.h2 = self.F(self.h1)
            self.time = 0
            self.E_tilde = torch.zeros_like(Ep).to(self.device)

        # hidden_tilde
        self.time += 1
        hidden_tilde = self.h0 + self.time*self.h1 + (self.time**2)*self.h2/2.0

        # E_tilde
        self.E_tilde = self.on_gru(self.E_tilde, Ep)

        # K
        combined = torch.cat([self.E_tilde, hidden_tilde], dim = 1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)

        next_hidden = hidden_tilde + K * (self.E_tilde - hidden_tilde)  # correction , Haddamard product
        # 可视化
        mcu = K * self.E_tilde
        tpu = (1-K) * hidden_tilde

        return next_hidden, mcu, tpu


class ON_ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ON_ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding = 1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, E_tilde, Ep):
        hx = torch.cat([E_tilde, Ep], dim = 1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * E_tilde, Ep], dim = 1)))

        h = (1 - z) * Ep + z * q
        return h