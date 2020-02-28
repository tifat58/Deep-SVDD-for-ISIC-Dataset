import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from base.base_net import BaseNet


class ISIC_VGG16(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 980
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv1 = nn.Conv2d(3, 64, 3, bias=False, padding=1)
        self.bn2d1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2d2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2d3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2d4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn2d5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2d6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2d7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn2d8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn2d9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn2d10 = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn2d11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn2d12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 980, 3, padding=1)
        self.bn2d13 = nn.BatchNorm2d(980)

        # self.fc1 = nn.Linear(512 * 7 * 7, 4096, bias=True)
        # self.fc2 = nn.Linear(4096, 4096, bias=True)
        # self.fc3 = nn.Linear(4096, self.rep_dim, bias=True)


        self.fc1 = nn.Linear(980 * 7 * 7, self.rep_dim, bias=True)

        # self.fc2 = nn.Linear(2000, self.rep_dim, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn2d1(x)
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.bn2d3(x)
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))
        x = self.conv5(x)
        x = self.bn2d5(x)
        x = self.conv6(x)
        x = self.bn2d6(x)
        x = self.conv7(x)
        x = self.pool(F.leaky_relu(self.bn2d7(x)))
        x = self.conv8(x)
        x = self.bn2d8(x)
        x = self.conv9(x)
        x = self.bn2d9(x)
        x = self.conv10(x)
        x = self.pool(F.leaky_relu(self.bn2d10(x)))
        x = self.conv11(x)
        x = self.bn2d11(x)
        x = self.conv12(x)
        x = self.bn2d12(x)
        x = self.conv13(x)
        x = self.pool(F.leaky_relu(self.bn2d13(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features



class ISIC_VGG16_Autoencoder(BaseNet):
    # vgg 16 au
    def __init__(self):
        super().__init__()

        self.rep_dim = 980
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv1 = nn.Conv2d(3, 64, 3, bias=False, padding=1)
        self.bn2d1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2d2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2d3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2d4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn2d5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2d6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2d7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn2d8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn2d9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn2d10 = nn.BatchNorm2d(512)

        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn2d11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn2d12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 980, 3, padding=1)
        self.bn2d13 = nn.BatchNorm2d(980)

        # self.fc1 = nn.Linear(512 * 7 * 7, 4096, bias=True)
        # self.fc2 = nn.Linear(4096, 4096, bias=True)
        # self.fc3 = nn.Linear(4096, self.rep_dim, bias=True)


        self.fc1 = nn.Linear(980 * 7 * 7, self.rep_dim, bias=True)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        # self.fc2 = nn.Linear(2000, self.rep_dim, bias=True)

        #         # decoder
        # self.dfc1 = nn.Linear(self.rep_dim, 4096)
        # self.dfc2 = nn.Linear(4096, 4096)
        # self.dfc3 = nn.Linear(4096, 25088)
        #
        # self.dfc1 = nn.Linear(self.rep_dim, 2000)
        #
        # self.dfc2 = nn.Linear(2000, 25088)

        #         for testing block

        #self.fc1 = nn.Linear(512 * 7 * 7, self.rep_dim, bias=True)
        #self.dfc1 = nn.Linear(self.rep_dim, 25088)

        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (7 * 7)), 980, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d14 = nn.BatchNorm2d(980)

        self.deconv2 = nn.ConvTranspose2d(980, 512, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain(('leaky_relu')))
        self.bn2d15 = nn.BatchNorm2d(512)

        self.deconv3 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain(('leaky_relu')))
        self.bn2d16 = nn.BatchNorm2d(512)

        self.deconv4 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain(('leaky_relu')))
        self.bn2d17 = nn.BatchNorm2d(512)

        self.deconv5 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv5.weight, gain=nn.init.calculate_gain(('leaky_relu')))
        self.bn2d18 = nn.BatchNorm2d(512)

        self.deconv6 = nn.ConvTranspose2d(512, 512, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv6.weight, gain=nn.init.calculate_gain(('leaky_relu')))
        self.bn2d19 = nn.BatchNorm2d(512)

        self.deconv7 = nn.ConvTranspose2d(512, 256, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv7.weight, gain=nn.init.calculate_gain(('leaky_relu')))
        self.bn2d20 = nn.BatchNorm2d(256)

        self.deconv8 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv8.weight, gain=nn.init.calculate_gain(('leaky_relu')))
        self.bn2d21 = nn.BatchNorm2d(256)

        self.deconv9 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv9.weight, gain=nn.init.calculate_gain(('leaky_relu')))
        self.bn2d22 = nn.BatchNorm2d(256)

        self.deconv10 = nn.ConvTranspose2d(256, 128, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv10.weight, gain=nn.init.calculate_gain(('leaky_relu')))
        self.bn2d23 = nn.BatchNorm2d(128)

        self.deconv11 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv11.weight, gain=nn.init.calculate_gain(('leaky_relu')))
        self.bn2d24 = nn.BatchNorm2d(128)

        self.deconv12 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv12.weight, gain=nn.init.calculate_gain(('leaky_relu')))
        self.bn2d25 = nn.BatchNorm2d(64)

        self.deconv13 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv13.weight, gain=nn.init.calculate_gain(('leaky_relu')))
        self.bn2d26 = nn.BatchNorm2d(64)

        self.deconv14 = nn.ConvTranspose2d(64, 3, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv14.weight, gain=nn.init.calculate_gain(('leaky_relu')))
        self.bn2d27 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn2d1(x)
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.bn2d3(x)
        x = self.conv4(x)
        x = self.pool(F.leaky_relu(self.bn2d4(x)))
        x = self.conv5(x)
        x = self.bn2d5(x)
        x = self.conv6(x)
        x = self.bn2d6(x)
        x = self.conv7(x)
        x = self.pool(F.leaky_relu(self.bn2d7(x)))
        x = self.conv8(x)
        x = self.bn2d8(x)
        x = self.conv9(x)
        x = self.bn2d9(x)
        x = self.conv10(x)
        x = self.pool(F.leaky_relu(self.bn2d10(x)))
        x = self.conv11(x)
        x = self.bn2d11(x)
        x = self.conv12(x)
        x = self.bn2d12(x)
        x = self.conv13(x)
        x = self.pool(F.leaky_relu(self.bn2d13(x)))
        x = x.view(x.size(0), -1)
        x = self.bn1d(self.fc1(x))

        # x = self.fc2(x)
        # x = self.fc3(x)

        # x = self.dfc1(x)
        # x = self.dfc2(x)
        # x = self.dfc3(x)

        x = x.view(x.size(0), int(self.rep_dim / (7 * 7)), 7, 7)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d14(x)), scale_factor=2)
        x = self.deconv2(x)
        x = self.bn2d15(x)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d16(x)), scale_factor=2)
        x = self.deconv4(x)
        x = self.bn2d17(x)
        x = self.deconv5(x)
        x = self.bn2d18(x)
        x = self.deconv6(x)
        x = F.interpolate(F.leaky_relu(self.bn2d19(x)), scale_factor=2)
        x = self.deconv7(x)
        x = self.bn2d20(x)
        x = self.deconv8(x)
        x = self.bn2d21(x)
        x = self.deconv9(x)
        x = F.interpolate(F.leaky_relu(self.bn2d22(x)), scale_factor=2)
        x = self.deconv10(x)
        x = self.bn2d23(x)
        x = self.deconv11(x)
        x = F.interpolate(F.leaky_relu(self.bn2d24(x)), scale_factor=2)
        x = self.deconv12(x)
        x = self.bn2d25(x)
        x = self.deconv13(x)
        x = self.bn2d26(x)
        x = self.deconv14(x)
        x = self.bn2d27(x)
        x = torch.sigmoid(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
