import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from data.moving_mnist import MovingMNIST
from data.sst import SSTSeq
from data import human
from data.datasets_factory import data_provider


def load_mmnist(args):
    root = './dataset/mmnist'
    mm = MovingMNIST(root = root, is_train = True, n_frames_input = 10, n_frames_output = 10, num_objects = [2])
    train_loader = DataLoader(dataset = mm, batch_size = args.batch_size, shuffle = True, num_workers = 0)

    mm = MovingMNIST(root = root, is_train = False, n_frames_input = 10, n_frames_output = args.target_len, num_objects = [2])
    test_loader = DataLoader(dataset = mm, batch_size = args.batch_size, shuffle = False, num_workers = 0)

    return train_loader, test_loader



def load_human(args):
    dataset_name = 'human'
    train_data_paths = './dataset/human'
    valid_data_paths = './dataset/human'

    n_gpu = 1
    img_width = 128
    total_length = 8

    train_input_handle, test_input_handle = data_provider(dataset_name, train_data_paths, valid_data_paths, 1 * n_gpu, img_width, seq_length = total_length, is_training = True)
    # test_input_handle = data_provider(dataset_name, train_data_paths, valid_data_paths, 1 * n_gpu, img_width, seq_length = total_length, is_training = False)

    class CustomDataset(Dataset):
        def __init__(self, input_handle):
            self.input_handle = input_handle

        def __len__(self):
            return self.input_handle.total()

        def __getitem__(self, idx):
            input_batch = self.input_handle.get_batch()  # shape (batch_size,8,128,128,3)
            input_batch = np.transpose(input_batch, [0, 1, 4, 2, 3])
            self.input_handle.next()
            return input_batch[0, :, :, :, :]  # removes batch dimension

    train_dataset = CustomDataset(train_input_handle)
    test_dataset = CustomDataset(test_input_handle)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 0)
    print('-------------------------- num iter train_loader ', len(train_loader))
    print('-------------------------- num iter test_loader ', len(test_loader))
    return train_loader, test_loader



def load_taxibj(args):
    train_data_paths = './dataset/TaxiBJ'
    valid_data_paths = './dataset/TaxiBJ'
    img_width = 32
    total_length = 8

    train_data, test_data = data_provider(args.dataset, train_data_paths, valid_data_paths, args.batch_size, img_width, seq_length = total_length, is_training = True)

    print('train data ', train_data.shape)  # (19560,8,32,32,2)
    print('test data ', test_data.shape)  # (1344,8,32,32,2)

    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            im = self.data[idx, :, :, :, :]
            im = np.transpose(im, [0, 3, 1, 2])
            im = im.astype('float32')
            return im

    train_dataset = CustomDataset(train_data)
    test_dataset = CustomDataset(test_data)

    train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 0)
    test_loader = DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 0)
    print('-------------------------- num iter train_loader ', len(train_loader))
    print('-------------------------- num iter test_loader ', len(test_loader))


    return train_loader, test_loader



