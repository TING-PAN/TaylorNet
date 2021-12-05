import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import numpy as np
import random
import argparse
from constrain_moments import K2M
import os
import time
import cv2

from dataloader import load_human
from models_human import on_11


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = 'human', help = '')
parser.add_argument('--save_path', type = str, default = './visual', help = '')
parser.add_argument('--batch_size', type = int, default = 4, help = 'batch_size')
parser.add_argument('--nepochs', type = int, default = 131, help = 'nb of epochs')

parser.add_argument('--Continue', type = bool, default = False, help = '')
parser.add_argument('--load_epoch', type = int, default = 70, help = '')
parser.add_argument('--experiment', type = str, default = 'on_11_human', help = '')
parser.add_argument('--mode', type = str, default = 'on_11', help = '')
parser.add_argument('--visual', type = bool, default = True, help = '')

args = parser.parse_args()

# dataset
if args.dataset == 'human':
    train_loader, test_loader = load_human(args)

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
    # state_dict = torch.load(load_path)
    state_dict = torch.load(load_path, map_location={'cuda:0':"cuda:1"})
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    model.load_state_dict(state_dict)


def train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion, teacher_forcing_ratio):
    encoder_optimizer.zero_grad()
    # input_tensor : torch.Size([batch_size, input_length, 1, 64, 64])
    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)
    loss_recons, loss_regularisation = 0, 0

    for ei in range(input_length - 1):
        encoder_output, encoder_hidden, output_image, _, _ = encoder(input_tensor[:, ei, :, :, :], (ei == 0))
        loss_recons += criterion(output_image, input_tensor[:, ei + 1, :, :, :])

    decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input= last image of input sequence

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    for di in range(target_length):
        decoder_output, decoder_hidden, output_image, _, _ = encoder(decoder_input)
        target = target_tensor[:, di, :, :, :]
        loss_recons += criterion(output_image, target)

        if use_teacher_forcing:
            decoder_input = target  # Teacher forcing
        else:
            decoder_input = output_image

    # # Regularisation sur les moments # encoder.physcell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
    # k2m = K2M([7, 7]).to(device)
    # for b in range(0, encoder.phycell.cell_list[0].input_dim):
    #     filters = encoder.phycell.cell_list[0].F.conv1.weight[:, b, :, :]  # (nb_filters,7,7)
    #     m = k2m(filters.double())
    #     m = m.float()
    #     loss_regularisation += criterion(m, constraints)  # constrains is a precomputed matrix



    # Regularisation sur les moments # encoder.physcell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
    k2m = K2M([7, 7]).to(device)
    for b in range(0, encoder.phycell.input_dim):
        filters = encoder.phycell.F.conv1.weight[:, b, :, :]  # (nb_filters,7,7)
        m = k2m(filters.double())
        m = m.float()
        loss_regularisation += criterion(m, constraints)  # constrains is a precomputed matrix



    loss = loss_recons + 10 * loss_regularisation
    loss.backward()
    encoder_optimizer.step()

    return loss.item() / target_length


def train(encoder, nepochs = args.nepochs, print_every = 10, eval_every = 10):
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr = 0.0001)
    best_mse = float('inf')
    scheduler_enc = ReduceLROnPlateau(encoder_optimizer, mode = 'min', patience = 3, factor = 0.5, verbose = True)
    criterion = nn.MSELoss(reduction = 'sum')

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
        teacher_forcing_ratio = np.maximum(0, 1 - epoch * 0.01)  # on le diminue de 0.01 a chaque epoch

        for i, input_batch in enumerate(train_loader, 0):  # input_batch =  torch.Size([batch, 8, 2, 32, 32])
            input_tensor = input_batch[:, 0:4, :, :, :].to(device)  # second half of the batch
            target_tensor = input_batch[:, 4:8, :, :, :].to(device)
            loss = train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion,
                                                   teacher_forcing_ratio)

            loss_epoch += loss

        if (epoch + 1) % print_every == 0:
            print('epoch ', epoch, ' loss ', loss_epoch, ' time epoch ', time.time() - t0)

        if (epoch + 1) % eval_every == 0:
            mse, mae, ssim = evaluate(encoder)
            scheduler_enc.step(mse)
            save_networks(epoch + 1, encoder)


def evaluate(encoder):
    total_mse, total_mae, total_ssim, total_psnr, total_bce = 0, 0, 0, 0, 0
    t0 = time.time()
    with torch.no_grad():

        for i, input_batch in enumerate(test_loader, 0):
            # input_batch = torch.Size([batch size, 20, 1, 64, 64])
            input_tensor = input_batch[:, 0:4, :, :, :].to(device)
            target_tensor = input_batch[:, 4:8, :, :, :].to(device)  # (batch_size,4,2,32,32)
            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]

            for ei in range(input_length - 1):
                encoder_output, encoder_hidden, _, _, _ = encoder(input_tensor[:, ei, :, :, :], (ei == 0))

            decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input= last image of input sequence
            predictions = []

            for di in range(target_length):
                decoder_output, decoder_hidden, output_image, _, _ = encoder(decoder_input)
                decoder_input = output_image
                predictions.append(output_image.cpu())

            input = input_tensor.cpu().numpy()
            target = target_tensor.cpu().numpy()
            predictions = np.stack(predictions)  # (4, batch_size, 3, 128, 128)
            predictions = predictions.swapaxes(0, 1)  # (batch_size,4, 3, 128, 128)

            mse_batch = np.mean((predictions - target) ** 2, axis = (0, 1)).sum()
            mae_batch = np.mean(np.abs(predictions - target), axis = (0, 1)).sum()
            total_mse += mse_batch
            total_mae += mae_batch

            for a in range(0, target.shape[0]):
                for b in range(0, target.shape[1]):
                    total_ssim += ssim(target[a,b,0,], predictions[a,b,0,], multichannel=True) / (target.shape[0]*target.shape[1])
                    total_psnr += psnr(target[a,b,0,], predictions[a,b,0,]) / (target.shape[0] * target.shape[1])

            cross_entropy = -target * np.log(predictions) - (1 - target) * np.log(1 - predictions)
            cross_entropy = cross_entropy.sum()
            cross_entropy = cross_entropy / (args.batch_size * target_length)
            total_bce += cross_entropy

    print('eval mse ', total_mse / len(test_loader), ' eval mae ', total_mae / len(test_loader), ' eval ssim ',
          total_ssim / len(test_loader), ' eval bce ', total_bce / len(test_loader), ' eval psnr ',
          total_psnr / len(test_loader),
          ' time= ', time.time() - t0)
    return total_mse / len(test_loader), total_mae / len(test_loader), total_ssim / len(test_loader)


def tensor2numpy(img):
    img = img.detach().cpu().numpy() * 255.
    img = img.transpose((1, 2, 0))
    return img


# encoder.load_state_dict(torch.load('./weights/on_11_human/epoch_70.pth'))
# encoder.eval()
# evaluate(encoder)

if args.visual:
    # evaluate or visualization
    load_networks(args.load_epoch, encoder)
    encoder.eval()
    evaluate(encoder)
else:
    # train
    train(encoder,print_every=1,eval_every=5)

