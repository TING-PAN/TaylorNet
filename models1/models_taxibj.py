import torch.nn as nn
import torch


class EncoderRNN_1(torch.nn.Module):
    def __init__(self, phycell, convcell, device):
        super(EncoderRNN_1, self).__init__()
        self.image_cnn_enc = nn.Sequential()  # 32x32x2 -> 16x16x64
        self.image_cnn_enc.add_module('conv1', nn.Conv2d(in_channels = 2, out_channels = 32, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)))
        self.image_cnn_enc.add_module('bn1', nn.GroupNorm(16, 32))
        self.image_cnn_enc.add_module('f_act1', nn.LeakyReLU(negative_slope = 0.1))
        self.image_cnn_enc.add_module('conv2', nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (5, 5), stride = (2, 2), padding = (2, 2)))
        self.image_cnn_enc.add_module('bn2', nn.GroupNorm(16, 64))
        self.image_cnn_enc.add_module('f_act2', nn.LeakyReLU(negative_slope = 0.1))
        self.image_cnn_enc = self.image_cnn_enc.to(device)

        self.image_cnn_dec = nn.Sequential()  # 16x16x64 -> 32x32x2
        self.image_cnn_dec.add_module('conv2d_transposed_1', nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = (5, 5), stride = (2, 2), padding = 2, output_padding = 1))
        self.image_cnn_dec.add_module('bn1', nn.GroupNorm(16, 32))
        self.image_cnn_dec.add_module('activation_1', nn.LeakyReLU(negative_slope = 0.1))
        self.image_cnn_dec.add_module('conv2d_transposed_2', nn.ConvTranspose2d(in_channels = 32, out_channels = 2, kernel_size = (3, 3), stride = (1, 1), padding = 1))
        self.image_cnn_dec = self.image_cnn_dec.to(device)

        self.phycell = phycell.to(device)
        self.convcell = convcell.to(device)

    def forward(self, input, first_timestep = False):
        output = self.image_cnn_enc(input)
        output1,_,_ = self.phycell(output, first_timestep)
        hidden2, output2 = self.convcell(output, first_timestep)

        out_phys = torch.sigmoid(self.image_cnn_dec(output1))  # partial reconstructions for vizualization
        out_conv = torch.sigmoid(self.image_cnn_dec(output2[-1]))

        output_image = self.image_cnn_dec(output1 + output2[-1])
        output_image = torch.sigmoid(output_image)
        return output, hidden2, output_image, (out_phys, out_conv),()


