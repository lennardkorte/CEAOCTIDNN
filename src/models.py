
import torch.nn as nn
import torch
from torchvision import models
from efficientnet_pytorch import EfficientNet
import pretrainedmodels
from pathlib import Path
from glob import glob

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
        self.layer2 = self.create_decoder_block(128, 64)
        self.layer3 = self.create_decoder_block(64, 32)

        # Output layer
        self.final_conv_transpose = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        self.output_activation = nn.Sigmoid()

    def create_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.initial_conv_transpose(x)
        x = self.initial_batchnorm(x)
        x = nn.ReLU(inplace=True)(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.final_conv_transpose(x)
        x = self.output_activation(x)

        return x

class ResNet18AutEnc(nn.Module):
    def __init__(self, config):
        super(ResNet18AutEnc, self).__init__()
        
        save_path = Path('./data/train_and_test', config['encoder_group'], config['name'])
        save_path_cv = save_path / ('cv_' + str(config["num_cv"]))
        for checkpoint_path in glob(str(save_path_cv / '*.pt')):
            if "checkpoint_best" in checkpoint_path:
                print("PathTest: ", checkpoint_path)
        
        resnet18 = ResNet18(config)
        checkpoint = torch.load(checkpoint_path)
        resnet18.load_state_dict(checkpoint['Model'])

        self.encoder = nn.Sequential(*list(resnet18.children())[:-2])
        self.decoder = Decoder()

        # Set the encoder layers to be non-trainable
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
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



