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
        self.normal_classes = tuple([1,2,3,4,5,6,7,8])
        self.outlier_classes = list(range(9,10))

        #self.outlier_classes.remove([0,1,2,3])

        # self.normal_classes = tuple([normal_class])
        # self.outlier_classes = list(range(0,8))
        # self.outlier_classes.remove(normal_class)

        print('Normal class:', self.normal_classes)
        print('outliers: ', self.outlier_classes)
        self.csv_file = '/home/fnunnari/skincaredata/ISIC_Challenge_2019/ISIC_2019_Training_GroundTruth.csv'

        self.test_csv_file = '/home/haal01/Desktop/Projects/Deep-SVDD-PyTorch-master/data/ISIC19_test_data.csv'

        self.test_root_dir = '/home/haal01/Desktop/Projects/val2017'

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
        transform = transforms.Compose([
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
       # print(len(self.train_set))
        train_idx_normal = get_target_label_idx(self.train_set.train_labels, self.normal_classes)
        self.train_set = Subset(self.train_set, train_idx_normal)
       # print(len(self.train_set))

        self.test_set = ISIC(csv_file=self.test_csv_file, train=False, root_dir=self.test_root_dir, transform=transform, target_transform=target_transform) # for unk class
        # self.test_set = ISIC(csv_file=self.csv_file, train=False, root_dir=root, transform=transform, target_transform=target_transform)



    def __len__(self):
        return len(self.total_dataset.target)

class ISIC(Dataset):

    def __init__(self, csv_file, root_dir, train=True, transform=None, target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.csv_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.target = []
        self.test_split = 0.2
        self.split_len = int((1 - self.test_split) * len(self.csv_data))

     #   if train:
        #    for i in range(self.split_len):
           #     for j in range(1,9):
         #           if self.csv_data.iloc[i,j] == 1:
           #             self.target.append(j)


        if train:
            for i in range(1, 20000): #20000
                for j in range(1,10): #change it to 10 for including outlier class
                    if self.csv_data.iloc[i,j] == 1:
                        self.target.append(j)
                        break
        else:
            for i in range(1, len(self.csv_data)): #len(self.csv_data)
                for j in range(1,10):
                    if self.csv_data.iloc[i,j] == 1:
                        self.target.append(j)
                        break


    def __len__(self):
        return len(self.target)

    @property
    def train_labels(self):
        # warnings.warn("test_labels has been renamed targets")

        return self.target

    @property
    def test_labels(self):
        return self.target



    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()




        if 'ISIC' in self.csv_data.iloc[idx, 0]:
            img_name = os.path.join(self.root_dir, self.csv_data.iloc[idx, 0] + '.jpg')
        else:
            img_name = os.path.join(self.root_dir, self.csv_data.iloc[idx, 0] + '.jpg')



        pic = Image.open(img_name, mode='r')
        pic = pic.resize((224,224),Image.NEAREST)
        #img = np.array(pic.getdata()).reshape(pic.size[0], pic.size[1], 3)
        img = np.asarray(pic).copy()
        t = (224,224,3)
        if img.shape != t:
            print(img.shape)
            print(img_name)
        #print(img.shape)
        # print(type(img), img.shape)
        # img = img[0:32][0:32][:]
        img = Image.fromarray(img)

        # img = ImgByteArr.getvalue()


        target = self.target[idx]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, idx

