
import torch.nn as nn
import torch
from torchvision import models
from efficientnet_pytorch import EfficientNet
import pretrainedmodels
from pathlib import Path
from glob import glob
import models_resnet_autenc

class ResNet18(nn.Module):
    def __init__(self, config):
        super(ResNet18, self).__init__()
        self.output_size = config['num_out']
        weights = None
        if config['pretrained']: weights = models.ResNet18_Weights.DEFAULT
        self.net = models.resnet18(weights=weights)
        self.net.avgpool = nn.AdaptiveAvgPool2d(1)
        self.net.fc = nn.Linear(512, self.output_size)

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, input_size=(512, 7, 7), output_size=(1, 64, 64)):
        super(Decoder, self).__init__()

        # Initial input size (512, 7, 7)
        self.initial_conv_transpose = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.initial_batchnorm = nn.BatchNorm2d(256)

        # Upsampling layers
        self.layer1 = self.create_decoder_block(256, 128)
        #self.layer1_0 = self.create_decoder_block(128, 128) # TODO
        self.layer2 = self.create_decoder_block(128, 64)

        self.extra_layer1 = self.create_decoder_block(64, 64, kernel_size=3, stride=1, padding=1)  # Add the first additional layer
        self.extra_layer2 = self.create_decoder_block(64, 64, kernel_size=3, stride=1, padding=1)  # Add the second additional layer

        self.layer3 = self.create_decoder_block(64, 32)

        # Output layer
        self.final_conv_transpose = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        self.output_activation = nn.Sigmoid()

    def create_decoder_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.initial_conv_transpose(x)
        x = self.initial_batchnorm(x)
        x = nn.ReLU(inplace=True)(x)

        x = self.layer1(x)
        #x = self.layer1_0(x)
        x = self.layer2(x)

        x = self.extra_layer1(x)  # Pass through the first additional layer
        x = self.extra_layer2(x)  # Pass through the second additional layer

        x = self.layer3(x)

        x = self.final_conv_transpose(x)
        x = self.output_activation(x)

        return x

# Create the custom autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def create_autoencoder2(config):

    # Load the pretrained ResNet18 model from a ".pt" file
    save_path = Path('./data/train_and_test', config['encoder_group'], config['name'])
    save_path_cv = save_path / ('cv_' + str(config["num_cv"]))
    for path in glob(str(save_path_cv / '*.pt')):
        if "checkpoint_best" in checkpoint_path:
            checkpoint_path = path
    
    resnet = ResNet18(config)

    checkpoint = torch.load(checkpoint_path)
    resnet.load_state_dict(checkpoint['Model']) # TODO: remove hashtag

    # Define the encoder using the first layers of ResNet18
    encoder = nn.Sequential(*list(resnet.net.children())[:-1])

    # Set the encoder layers to be non-trainable
    for param in encoder.parameters():
        param.requires_grad = False

    # Create the autoencoder model
    decoder = Decoder()
    autoencoder = Autoencoder(encoder, decoder)

    return autoencoder

def create_autoencoder(config):

    # Load the pretrained ResNet18 model from a ".pt" file
    save_path = Path('./data/train_and_test', config['encoder_group'], config['encoder_name'])
    save_path_cv = save_path / ('cv_' + str(config["num_cv"]))
    for path in glob(str(save_path_cv / '*.pt')):
        if "checkpoint_best" in path:
            checkpoint_path = path
    
    resnet = ResNet18(config)

    checkpoint = torch.load(checkpoint_path)
    resnet.load_state_dict(checkpoint['Model'])

    # Define the encoder using the first layers of ResNet18
    encoder = nn.Sequential(*list(resnet.net.children())[:-2])

    # Set the encoder layers to be non-trainable
    for param in encoder.parameters():
        param.requires_grad = False

    arch, bottleneck = models_resnet_autenc.get_configs('resnet18')
    decoder = models_resnet_autenc.ResNetDecoder(arch[::-1], bottleneck=bottleneck)
    
    #print(arch[::-1])
    # encoder2 = models_resnet_autenc.ResNetEncoder(arch, bottleneck=bottleneck)
    # print(encoder)
    # exit()

    autoencoder = Autoencoder(encoder, decoder)

    return autoencoder
    
class AlexNet(nn.Module):
    def __init__(self, config):
        super(AlexNet, self).__init__()
        self.output_size = config['num_out']
        # self.layer1 = nn.Linear(1000,1)
        weights = None
        if config['pretrained']: weights = models.AlexNet_Weights.DEFAULT
        self.net = models.alexnet(weights=weights)
        # for p in self.net.parameters():
        #   p.requires_grad=False
        self.net.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.output_size))

    def forward(self, x):
        return self.net(x)

class WideResNet50(nn.Module):
    def __init__(self, config):  # adjustedpool = False,
        super(WideResNet50, self).__init__()
        self.output_size = config['num_out']
        # self.layer1 = nn.Linear(1000,1)
        weights = None
        if config['pretrained']: weights = models.Wide_ResNet50_2_Weights.DEFAULT
        self.net = models.wide_resnet50_2(weights=weights)
        # self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net.avgpool = nn.AdaptiveAvgPool2d(1)
        # =============================================================================
        #         if self.adjusted == True:
        #             self.net.avgpool = nn.AvgPool2d(kernel_size=16, stride=1, padding=0)
        # =============================================================================
        self.net.fc = nn.Linear(2048, self.output_size)

    def forward(self, x):
        return self.net(x)

class ResNext50(nn.Module):
    def __init__(self, config):  # adjustedpool = False,
        super(ResNext50, self).__init__()
        self.output_size = config['num_out']
        # self.layer1 = nn.Linear(1000,1)
        weights = None
        if config['pretrained']: weights = models.ResNeXt50_32X4D_Weights.DEFAULT
        self.net = models.resnext50_32x4d(weights=weights)
        # self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net.avgpool = nn.AdaptiveAvgPool2d(1)
        # =============================================================================
        #         if self.adjusted == True:
        #             self.net.avgpool = nn.AvgPool2d(kernel_size=16, stride=1, padding=0)
        # =============================================================================
        self.net.fc = nn.Linear(2048, self.output_size)

    def forward(self, x):
        return self.net(x)

class DenseNet121(nn.Module):
    def __init__(self, config):
        super(DenseNet121, self).__init__()
        self.output_size = config['num_out']
        weights = None
        if config['pretrained']: weights = models.DenseNet121_Weights.DEFAULT
        self.net = models.densenet121(weights=weights)
        self.net.classifier = nn.Linear(1024, self.output_size)

    def forward(self, x):
        return self.net(x)


