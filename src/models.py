
import torch.nn as nn
import torch
from torchvision import models
from efficientnet_pytorch import EfficientNet
import pretrainedmodels
from pathlib import Path
from glob import glob


# Define the BasicBlock class
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

# Define the ResNet class
class MyResNet18(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1000):
        super(MyResNet18, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, config):
        super(ResNet18, self).__init__()
        self.output_size = config['num_out']
        self.net = models.resnet18(config['pretrained'])
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

def create_autoencoder(config):

    # Load the pretrained ResNet18 model from a ".pt" file
    save_path = Path('./data/train_and_test', config['encoder_group'], config['name'])
    save_path_cv = save_path / ('cv_' + str(config["num_cv"]))
    for checkpoint_path in glob(str(save_path_cv / '*.pt')):
        if "checkpoint_best" in checkpoint_path:
            print("Load encoder for autoencoder from: ", checkpoint_path)
    
    resnet = ResNet18(config)

    checkpoint = torch.load(checkpoint_path)
    resnet.load_state_dict(checkpoint['Model']) # TODO: remove hashtag

    # Define the encoder using the first layers of ResNet18
    encoder = nn.Sequential(*list(resnet.net.children())[:-1])

    # Set the encoder layers to be non-trainable
    for param in encoder.parameters():
        param.requires_grad = False # TODO: remove hashtag

    # Create the autoencoder model
    decoder = Decoder()
    autoencoder = Autoencoder(encoder, decoder)

    return autoencoder
    
class AlexNet(nn.Module):
    def __init__(self, config):
        super(AlexNet, self).__init__()
        self.output_size = config['num_out']
        # self.layer1 = nn.Linear(1000,1)
        self.net = models.alexnet(config['pretrained'])
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
        self.net = models.wide_resnet50_2(config['pretrained'])
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
        self.net = models.resnext50_32x4d(config['pretrained'])
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
        self.net = models.densenet121(config['pretrained'])
        self.net.classifier = nn.Linear(1024, self.output_size)

    def forward(self, x):
        return self.net(x)

class EfficientNetB6(nn.Module):
    def __init__(self, config):
        super(EfficientNetB6, self).__init__()
        self.output_size = config['num_out']
        if config['pretrained']:
            self.net = EfficientNet.from_pretrained('efficientnet-b6')
        else:
            self.net = EfficientNet.from_name('efficientnet-b6')
        self.net._fc = nn.Linear(2304, self.output_size)

    def forward(self, x):
        return self.net(x)

class SENet(nn.Module):
    def __init__(self, config):
        super(SENet, self).__init__()
        self.output_size = config['num_out']
        if config['pretrained']:
            self.net = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')  # 'se_resnext50_32x4d'
        else:
            self.net = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=None)  # 'se_resnext50_32x4d'
        
        self.net.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.net.last_linear = nn.Linear(2048, self.output_size)

    def forward(self, x):
        #        x = F.relu(self.net(x))
        #        x = self.fc(x)
        return self.net(x)



