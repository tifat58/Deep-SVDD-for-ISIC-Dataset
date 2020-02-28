import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class ISIC_AlexNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 512
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv1 = nn.Conv2d(3, 96, 7, bias=False, padding=1, stride=4)
        self.bn2d1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 120, 5, padding=2)
        self.bn2d2 = nn.BatchNorm2d(120)
        self.conv3 = nn.Conv2d(120, 120, 3, padding=1)
        self.bn2d3 = nn.BatchNorm2d(120)
        self.conv4 = nn.Conv2d(120, 120, 3, padding=1)
        self.bn2d4 = nn.BatchNorm2d(120)
        self.conv5 = nn.Conv2d(120, 150, 3, padding=1)
        self.bn2d5 = nn.BatchNorm2d(150)
        self.fc1 = nn.Linear(150 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)

        x = self.conv4(x)

        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn2d5(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class ISIC_AlexNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 512
        self.pool = nn.MaxPool2d(2, 2)

        # encoder
        self.conv1 = nn.Conv2d(3, 96, 7, bias=False, padding=1, stride=4)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 120, 5, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(120)
        self.conv3 = nn.Conv2d(120, 120, 3, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(120)
        self.conv4 = nn.Conv2d(120, 120, 3, padding=1)
        nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(120)
        self.conv5 = nn.Conv2d(120, 150, 3, padding=1)
        nn.init.xavier_uniform_(self.conv5.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(150)
        self.fc1 = nn.Linear(150 * 8 * 8, 9600)
        self.fc2 = nn.Linear(9600, 2048)
        self.fc3 = nn.Linear(2048, self.rep_dim)

        # decoder
        self.dfc1 = nn.Linear(self.rep_dim, 2048)
        self.dfc2 = nn.Linear(2048, 9600)

        self.deconv1 = nn.ConvTranspose2d(int(9600 / (8 * 8)), 150, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(150)

        self.deconv2 = nn.ConvTranspose2d(150, 120, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d7 = nn.BatchNorm2d(120)

        self.deconv3 = nn.ConvTranspose2d(120, 120, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d8 = nn.BatchNorm2d(120)

        self.deconv4 = nn.ConvTranspose2d(120, 120, 3, padding=1)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d9 = nn.BatchNorm2d(120)

        self.deconv5 = nn.ConvTranspose2d(120, 96, 5, padding=2)
        nn.init.xavier_uniform_(self.deconv5.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d10 = nn.BatchNorm2d(96)

        self.deconv6 = nn.ConvTranspose2d(96, 3, 7, padding=1, stride=4)
        nn.init.xavier_uniform_(self.deconv6.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(F.leaky_relu(self.bn2d5(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.dfc1(x)
        x = self.dfc2(x)
        x = x.view(x.size(0), int(9600 / (8 * 8)), 8, 8)
        x = F.leaky_relu(x)
        x = self.deconv1(x)

        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d7(x)), scale_factor=2)
        x = self.deconv3(x)

        x = self.deconv4(x)

        x = self.deconv5(x)
        x = F.interpolate(F.leaky_relu(self.bn2d10(x)), scale_factor=2)
        x = self.deconv6(x)
        x = torch.sigmoid(x)
        return x