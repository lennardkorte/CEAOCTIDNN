
# Inspired by: https://github.com/Horizon2333/imagenet-autoencoder/blob/536633ec1c0e9afe2dd91ce74b56e6e13479b6bd/models/resnet.py
# (This file manually builds Resnets with different configurations and their respective autoencoders)

import torch.nn as nn
import torch
from torchvision import models
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

def create_autoenc_resnet18(config):

    # Load the pretrained ResNet18 model from a ".pt" file
    save_path_ae_cv = Path('./data/train_and_test', config['encoder_group'], config['encoder_name'], ('cv_' + str(config["num_cv"])))
    for path in glob(str(save_path_ae_cv / '*.pt')):
        if "checkpoint_best" in path:
            checkpoint_path = path
    
    resnet = ResNet18(config)

    checkpoint = torch.load(checkpoint_path)
    resnet.load_state_dict(checkpoint['Model'])

    # Define the encoder using the first layers of the model
    encoder = nn.Sequential(*list(resnet.net.children())[:-2])

    # Set the encoder layers to be non-trainable
    for param in encoder.parameters():
        param.requires_grad = False

    arch, bottleneck = models_resnet_autenc.get_configs('resnet18')
    decoder = models_resnet_autenc.ResNetDecoder(arch[::-1], bottleneck=bottleneck)

    autoencoder = Autoencoder(encoder, decoder)

    return autoencoder
