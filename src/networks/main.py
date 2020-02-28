from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .isic_alexNet import ISIC_AlexNet, ISIC_AlexNet_Autoencoder
from .ISIC_VGG16 import ISIC_VGG16, ISIC_VGG16_Autoencoder
from .ISIC_VGG16_PRE import ISIC_VGG16_PRE_Autoencoder, ISIC_VGG16_PRE


def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'isic_AlexNet', 'ISIC_VGG16', 'ISIC_VGG16_PRE')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()

    if net_name == 'isic_AlexNet':
        net = ISIC_AlexNet()

    if net_name == 'ISIC_VGG16':
        net = ISIC_VGG16()

    if net_name == 'ISIC_VGG16_PRE':
        net = ISIC_VGG16_PRE()

    return net

def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'isic_AlexNet', 'ISIC_VGG16', 'ISIC_VGG16_PRE')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    if net_name == 'isic_AlexNet':
        ae_net = ISIC_AlexNet_Autoencoder()

    if net_name == 'ISIC_VGG16':
        ae_net = ISIC_VGG16_Autoencoder()

    if net_name == 'ISIC_VGG16_PRE':
        ae_net = ISIC_VGG16_PRE_Autoencoder()

    return ae_net
