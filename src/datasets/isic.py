from torch.utils.data import Subset
from PIL import Image
import torch
import os
import numpy as np
from skimage import io, transform
import io as file_io
import pandas as pd
from torchvision.datasets import MNIST
# from .torchvision_dataset import TorchvisionDataset
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms


class ISIC_Dataset(TorchvisionDataset):
    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 9))
        self.outlier_classes.remove(normal_class)
        self.csv_file = '../data/ISIC_2019_Training_GroundTruth.csv'

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-0.8826567065619495, 9.001545489292527),
                   (-0.6661464580883915, 20.108062262467364),
                   (-0.7820454743183202, 11.665100841080346),
                   (-0.7645772083211267, 12.895051191467457),
                   (-0.7253923114302238, 12.683235701611533),
                   (-0.7698501867861425, 13.103278415430502),
                   (-0.778418217980696, 10.457837397569108),
                   (-0.7129780970522351, 12.057777597673047),
                   (-0.8280402650205075, 10.581538445782988),
                   (-0.7369959242164307, 10.697039838804978)]

        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        ])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        self.train_set = ISIC(csv_file=self.csv_file, train=True, root_dir=root, transform=transform, target_transform=target_transform)

        # train_size = int(0.8 * len(self.total_dataset))
        # test_size = len(self.total_dataset) - train_size
        # train_set, test_set = torch.utils.data.random_split(self, [train_size, test_size])
        # print(train_set, type(self.total_dataset))
        # train_set = MyMNIST(root=self.root, train=True, download=True,
        #                     transform=transform, target_transform=target_transform)
        # Subset train_set to normal class
        train_label_list = []
        # print(self.train_set.train_labels)
        # for i in range(len(self.train_set)):
        #     train_label_list.append(self.train_set[i][1])
        #     # print(self.train_set[i][1], self.train_set[i][2])
        #     # if i > 10:
        #     #     break

        train_idx_normal = get_target_label_idx(self.train_set.train_labels, self.normal_classes)
        self.train_set = Subset(self.train_set, train_idx_normal)

        self.test_set = ISIC(csv_file=self.csv_file, train=False, root_dir=root, transform=transform, target_transform=target_transform)

        # self.test_set = MyMNIST(root=self.root, train=False, download=True,
        #                         transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.total_dataset.target)
