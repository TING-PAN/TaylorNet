import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import numpy as np
import random
import time
import os
import argparse

from models import on_11
from dataloader import load_mmnist
from constrain_moments import K2M

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = 'mmnist', help = '')
parser.add_argument('--batch_size', type = int, default = 16, help = 'batch_size')
parser.add_argument('--nepochs', type = int, default = 2001, help = 'nb of epochs')
parser.add_argument('--print_every', type = int, default = 1, help = '')
parser.add_argument('--eval_every', type = int, default = 10, help = '')
parser.add_argument('--target_len', type = int, default = 10, help = 'batch_size')

parser.add_argument('--Continue', type = bool, default = False, help = '')
parser.add_argument('--load_epoch', type = int, default = 370, help = '')
parser.add_argument('--experiment', type = str, default = 'on_11_mmnist', help = '')
parser.add_argument('--mode', type = str, default = 'on_11', help = '')
args = parser.parse_args()

# dataset
if args.dataset == 'mmnist':
    train_loader, test_loader = load_mmnist(args)


print('BEGIN TRAIN')
# initial model
if args.mode == 'on_11':
    encoder = on_11()
else:
    print('Something Wrong!')

constraints = torch.zeros((49, 7, 7)).to(device)
ind = 0
for i in range(0, 7):
    for j in range(0, 7):
        constraints[ind, i, j] = 1
        ind += 1


def save_networks(i_epoch, model):
    save_filename = 'epoch_{:<d}.pth'.format(i_epoch)
    path = os.path.join('./weights', args.experiment)
    if not os.path.exists(path):
        os.mkdir(path)
    save_path = os.path.join(path, save_filename)
    torch.save(model.state_dict(), save_path)
    print(' * save model (epoch {}) '.format(i_epoch))


def load_networks(load_epoch, model):
    load_filename = 'epoch_{:<d}.pth'.format(load_epoch)
    load_path = os.path.join('./weights', args.experiment, load_filename)
    state_dict = torch.load(load_path)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    model.load_state_dict(state_dict)


def train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion, teacher_forcing_ratio):
    encoder_optimizer.zero_grad()
    # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)
    loss = 0


    for ei in range(input_length - 1):
        encoder_output, encoder_hidden, output_image, _, inp = encoder(input_tensor[:, ei, :, :, :], (ei == 0))
        loss += criterion(output_image, input_tensor[:, ei + 1, :, :, :])




    decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input = last image of input sequence


    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    for di in range(target_length):
        decoder_output, decoder_hidden, output_image, _, inp = encoder(decoder_input)
        target = target_tensor[:, di, :, :, :]

        loss += criterion(output_image, target)

        if use_teacher_forcing:
            decoder_input = target  # Teacher forcing
        else:
            decoder_input = output_image




    # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
    k2m = K2M([7, 7]).to(device)
    for b in range(0, encoder.phycell.input_dim):
        filters = encoder.phycell.F.conv1.weight[:, b, :, :]  # (nb_filters,7,7)
        m = k2m(filters.double())
        m = m.float()
        loss += criterion(m, constraints)  # constrains is a precomputed matrix


    # k2m = K2M([7, 7]).to(device)
    # for b in range(0, encoder.phycell.input_dim):
    #     filters1 = encoder.phycell.F1.conv1.weight[:, b, :, :]  # (nb_filters,7,7)
    #     filters2 = encoder.phycell.F2.conv1.weight[:, b, :, :]  # (nb_filters,7,7)
    #     m1 = k2m(filters1.double()).float()
    #     m2 = k2m(filters2.double()).float()
    #     loss += criterion(m1, constraints)
    #     loss += criterion(m2, constraints)


    loss.backward()
    encoder_optimizer.step()
    return loss.item() / target_length


def trainIters(encoder, nepochs, print_every = 10, eval_every = 10):
    train_losses = []
    best_mse = float('inf')

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = 0.001)
    scheduler_enc = ReduceLROnPlateau(encoder_optimizer, mode = 'min', patience = 3, factor = 0.1, verbose = True)
    criterion = nn.MSELoss()

    # load model
    if args.Continue:
        load_networks(args.load_epoch, encoder)
        base_epoch = args.load_epoch
        print(f'*load model [epoch {base_epoch}]')
    else:
        base_epoch = 0

    for epoch in range(base_epoch, nepochs):
        t0 = time.time()
        loss_epoch = 0
        teacher_forcing_ratio = np.maximum(0, 1 - epoch * 0.003)  # on le diminue de 0.01 a chaque epoch

        for i, out in enumerate(train_loader, 0):
            input_tensor = out[1].to(device)
            target_tensor = out[2].to(device)
            loss = train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion,
                                  teacher_forcing_ratio)
            loss_epoch += loss

        train_losses.append(loss_epoch)
        if (epoch + 1) % print_every == 0:
            print('epoch ', epoch, ' loss ', loss_epoch, ' time epoch ', time.time() - t0)

        if (epoch + 1) % eval_every == 0:
            mse, mae, ssim = evaluate(encoder, test_loader)
            scheduler_enc.step(mse)
            save_networks(epoch + 1, encoder)
    return train_losses


def evaluate(encoder, loader):
    total_mse, total_mae, total_ssim, total_bce, total_psnr = 0, 0, 0, 0, 0
    t0 = time.time()
    with torch.no_grad():
        for i, out in enumerate(loader, 0):
            input_tensor = out[1].to(device)
            target_tensor = out[2].to(device)
            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]


            for ei in range(input_length - 1):
                encoder_output, encoder_hidden, _, _, _ = encoder(input_tensor[:, ei, :, :, :], (ei == 0))

            decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input= last image of input sequence
            predictions = []

            for di in range(target_length):
                decoder_output, decoder_hidden, output_image, _, _ = encoder(decoder_input, False)
                decoder_input = output_image
                predictions.append(output_image.cpu())

            input = input_tensor.cpu().numpy()
            target = target_tensor.cpu().numpy()
            predictions = np.stack(predictions)  # (10, batch_size, 1, 64, 64)
            predictions = predictions.swapaxes(0, 1)  # (batch_size,10, 1, 64, 64)

            mse_batch = np.mean((predictions - target) ** 2, axis = (0, 1, 2)).sum()
            mae_batch = np.mean(np.abs(predictions - target), axis = (0, 1, 2)).sum()
            total_mse += mse_batch
            total_mae += mae_batch

            for a in range(0, target.shape[0]):
                for b in range(0, target.shape[1]):
                    total_ssim += ssim(target[a, b, 0,], predictions[a, b, 0,]) / (target.shape[0] * target.shape[1])
                    total_psnr += psnr(target[a, b, 0,], predictions[a, b, 0,]) / (target.shape[0] * target.shape[1])

            cross_entropy = -target * np.log(predictions) - (1 - target) * np.log(1 - predictions)
            cross_entropy = cross_entropy.sum()
            cross_entropy = cross_entropy / (args.batch_size * target_length)
            total_bce += cross_entropy

    print('eval mse ', total_mse / len(loader), ' eval mae ', total_mae / len(loader), ' eval ssim ',
          total_ssim / len(loader), ' eval bce ', total_bce / len(loader), ' eval psnr ', total_psnr / len(loader),
          ' time= ', time.time() - t0)
    return total_mse / len(loader), total_mae / len(loader), total_ssim / len(loader)


# train
# evaluate(encoder, test_loader)
plot_losses = trainIters(encoder, args.nepochs, print_every = args.print_every, eval_every = args.eval_every)
print(plot_losses)
