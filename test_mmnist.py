import numpy as np
import torch
import os
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import argparse
import cv2

from models import on_11
from dataloader import load_mmnist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = 'mmnist', help = '')
parser.add_argument('--target_len', type=int, default=10, help='batch_size')
parser.add_argument('--save_path', type = str, default = './visual', help = '')
parser.add_argument('--vis_num', type = int, default = 10, help = '')
parser.add_argument('--batch_size', type=int, default=300, help='batch_size')

parser.add_argument('--load_epoch', type = int, default = 1110, help = '')
parser.add_argument('--visual', type = bool, default = False, help = '')
parser.add_argument('--experiment', type = str, default = 'on_11_mmnist', help = '')
parser.add_argument('--mode', type = str, default = 'on_11', help = '')

args = parser.parse_args()

# dataset
if args.dataset == 'mmnist':
    train_loader, test_loader = load_mmnist(args)

if args.mode == 'on_11':
    encoder = on_11()
else:
    print('Something Wrong!')


def load_networks(load_epoch, model):
    experiment = args.experiment
    load_filename = 'epoch_{:<d}.pth'.format(load_epoch)
    load_path = os.path.join('./weights', experiment, load_filename)
    state_dict = torch.load(load_path)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    model.load_state_dict(state_dict)


def tensor2numpy(img):
    img = img.detach().cpu().numpy() * 255.
    img = img.transpose((1, 2, 0))
    return img

def evaluate(encoder, loader):
    total_mse, total_mae, total_ssim, total_bce, total_psnr = 0, 0, 0, 0, 0
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
                # if args.mode == 'on_5'or args.mode == 'on_7' or args.mode == 'on_8':
                decoder_output, decoder_hidden, output_image, _, _ = encoder(decoder_input, False)
                # elif args.mode == 'on_3':
                #     decoder_output, decoder_hidden, (output_image, predic1, predic2), _, _ = encoder(decoder_input, False)
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
          total_ssim / len(loader), ' eval bce ', total_bce / len(loader), ' eval psnr ', total_psnr / len(loader))
    return total_mse / len(loader), total_mae / len(loader), total_ssim / len(loader)



def visualization(encoder, loader, vis_num = 10):
    with torch.no_grad():
        for i, out in enumerate(loader, 0):
            # input_batch = torch.Size([8, 20, 1, 64, 64])
            input_tensor = out[1].to(device)
            target_tensor = out[2].to(device)

            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]



            for ei in range(input_length - 1):
                encoder_output, encoder_hidden, _, vis, _ = encoder(input_tensor[:, ei, :, :, :], (ei == 0))
                len_vis = len(vis)

            decoder_input = input_tensor[:, -1, :, :, :]  # first decoder input= last image of input sequence
            predictions = []

            list = []
            for i in range(len_vis):
                list.append([])

            for di in range(target_length):
                decoder_output, decoder_hidden, output_image, vis, _ = encoder(decoder_input, False)
                decoder_input = output_image
                predictions.append(output_image)

                for i in range(len_vis):
                    list[i].append(vis[i])


            predictions = torch.stack(predictions, dim = 1)  # for MM: (batch_size, 10, 1, 64, 64)

            vis = []
            for i in range(len_vis):
                vis.append(torch.stack(list[i], dim = 1))


            for num in range(vis_num):
                real = []
                fake = []
                img_list = []
                for i in range(len_vis):
                    img_list.append([])
                for frame in range(input_tensor.size(1)):
                    real_img = tensor2numpy(input_tensor[num, frame])
                    fake_img = real_img
                    real.append(real_img)
                    fake.append(fake_img)

                    for i in range(len_vis):
                        img_list[i].append(real_img)

                for frame in range(target_tensor.size(1)):
                    real_img = tensor2numpy(target_tensor[num, frame])  # [H,W,C]
                    fake_img = tensor2numpy(predictions[num, frame])
                    real.append(real_img)
                    fake.append(fake_img)

                    for i in range(len_vis):
                        # ******  为了更加清晰 *10  ******
                        img = tensor2numpy(vis[i][num, frame])
                        img_list[i].append(img)


                real_imgs = np.concatenate(real, axis = 1)
                fake_imgs = np.concatenate(fake, axis = 1)

                imgs_list = []
                for i in range(len_vis):
                    imgs_list.append(np.concatenate(img_list[i], axis = 1))


                real_fake = np.concatenate([real_imgs, fake_imgs], axis = 0)
                for i in range(len_vis):
                    real_fake = np.concatenate([real_fake, imgs_list[i]], axis = 0)
                path = os.path.join(args.save_path, args.experiment)
                if not os.path.exists(path):
                    os.mkdir(path)
                cv2.imwrite(path + f'/{num}.jpg', real_fake)

            break


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6



print('BEGIN TEST')
load_networks(args.load_epoch, encoder)

if args.visual:
    visualization(encoder, test_loader, vis_num = args.vis_num)
else:
    evaluate(encoder, test_loader)

