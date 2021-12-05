import torch

from models1.lstm import ConvLSTM

from models1.models_taxibj import EncoderRNN_1
from models1.phy9 import PhyCell9


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

phycell9 = PhyCell9(device=device)
convcell = ConvLSTM(input_shape = (16, 16), input_dim = 64, hidden_dims = [256, 256, 256, 64], n_layers = 4, kernel_size = (3, 3), device = device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def on_11():
    encoder = EncoderRNN_1(phycell9, convcell, device)
    print('phycell ', count_parameters(phycell9))
    print('convcell ', count_parameters(convcell))
    print('encoder ', count_parameters(encoder))
    return encoder


