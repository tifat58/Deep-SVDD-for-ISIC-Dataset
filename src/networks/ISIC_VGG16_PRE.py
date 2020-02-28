import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary

from base.base_net import BaseNet



class ISIC_VGG16_PRE_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.model_ft = models.vgg16()


class ISIC_VGG16_PRE(BaseNet):

    def __init__(self):
        super().__init__()

        print(torch.cuda.device_count())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_ft = models.vgg16().to(self.device)

        # self.model_ext = nn.Sequential()

        # self.modules = list(self.model_ft.children())[:-1]
        # self.model = nn.Sequential().to(self.device)
        # self.model.features = nn.Sequential(*self.modules)
        #
        # self.model.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(in_features=1024, out_features=512, bias=True),
        #     nn.Dropout(),
        #     nn.ReLU(),
        #     nn.Linear(in_features=512, out_features=512, bias=True)
        # )



        summary(self.model_ft, (3, 224, 224))

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn2d1(x)
            x = self.conv2(x)
            x = self.pool(F.leaky_relu(self.bn2d2(x)))

            return x
