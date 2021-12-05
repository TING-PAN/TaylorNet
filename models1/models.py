import torch.nn as nn
import torch
import torch.distributions as distrib
import torch.nn.functional as F
from torch.autograd import Variable

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3,3), stride=stride, padding=1),
                nn.GroupNorm(16,nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if (stride ==2):
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channels=nin,out_channels=nout,kernel_size=(3,3), stride=stride,padding=1,output_padding=output_padding),
                nn.GroupNorm(16,nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)


class encoder_E(nn.Module):
    def __init__(self, nc = 1, nf = 32):
        super(encoder_E, self).__init__()
        # input is (1) x 64 x 64
        self.c1 = dcgan_conv(nc, nf, stride = 2)  # (32) x 32 x 32
        self.c2 = dcgan_conv(nf, nf, stride = 1)  # (32) x 32 x 32
        self.c3 = dcgan_conv(nf, 2 * nf, stride = 2)  # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        return h3


class decoder_D(nn.Module):
    def __init__(self, nc = 1, nf = 32):
        super(decoder_D, self).__init__()
        self.upc1 = dcgan_upconv(2 * nf, nf, stride = 2)  # (32) x 32 x 32
        self.upc2 = dcgan_upconv(nf, nf, stride = 1)  # (32) x 32 x 32
        self.upc3 = nn.ConvTranspose2d(in_channels = nf, out_channels = nc, kernel_size = (3, 3), stride = 2,
                                       padding = 1, output_padding = 1)  # (nc) x 64 x 64

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        return d3


class encoder_specific(nn.Module):
    def __init__(self, nc = 64, nf = 64):
        super(encoder_specific, self).__init__()
        self.c1 = dcgan_conv(nc, nf, stride = 1)  # (64) x 16 x 16
        self.c2 = dcgan_conv(nf, nf, stride = 1)  # (64) x 16 x 16

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        return h2


class decoder_specific(nn.Module):
    def __init__(self, nc = 64, nf = 64):
        super(decoder_specific, self).__init__()
        self.upc1 = dcgan_upconv(nf, nf, stride = 1)  # (64) x 16 x 16
        self.upc2 = dcgan_upconv(nf, nc, stride = 1)  # (64) x 16 x 16

    def forward(self, input):
        d1 = self.upc1(input)
        d2 = self.upc2(d1)
        return d2


class EncoderRNN_on_5(torch.nn.Module):
    def __init__(self, phycell, convcell, device):
        super(EncoderRNN_on_5, self).__init__()
        self.encoder_Ep = encoder_specific().to(device)  # specific image encoder 32x32x32 -> 16x16x64
        self.encoder_Er = encoder_specific().to(device)
        self.decoder_Dp = decoder_specific().to(device)  # specific image decoder 16x16x64 -> 32x32x32
        self.decoder_Dr = decoder_specific().to(device)
        self.encoder_E = encoder_E().to(device)  # general encoder 64x64x1 -> 32x32x32
        self.decoder_D = decoder_D().to(device)

        self.phycell = phycell.to(device)
        self.convcell = convcell.to(device)
        self.device = device

    def forward(self, input, first_timestep = False):

        input = self.encoder_E(input)

        input_conv = self.encoder_Er(input)
        input_phys = self.encoder_Ep(input)

        output1, mcu, tpu = self.phycell(input_phys, first_timestep)
        hidden2, output2 = self.convcell(input_conv, first_timestep)

        decoded_Dp = self.decoder_Dp(output1)
        decoded_Dr = self.decoder_Dr(output2[-1])

        output_image = torch.sigmoid(self.decoder_D(decoded_Dp+decoded_Dr))

        # partial reconstructions for vizualization
        out_phys = torch.sigmoid(self.decoder_D(decoded_Dp))
        out_conv = torch.sigmoid(self.decoder_D(decoded_Dr))

        # out_mcu = torch.sigmoid(self.decoder_D(self.decoder_Dp(mcu)))
        # out_tpu = torch.sigmoid(self.decoder_D(self.decoder_Dp(tpu)))

        return out_phys, hidden2, output_image, (out_phys, out_conv), ()


