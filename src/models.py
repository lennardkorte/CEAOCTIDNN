
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
import pretrainedmodels

class AlexNet(nn.Module):
    def __init__(self, num_out, pretrained):
        super(AlexNet, self).__init__()
        self.output_size = num_out
        # self.layer1 = nn.Linear(1000,1)
        self.net = models.alexnet(pretrained)
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

class ResNet18(nn.Module):
    def __init__(self, num_out, pretrained):  # adjustedpool = False,
        super(ResNet18, self).__init__()
        self.output_size = num_out
        # self.layer1 = nn.Linear(1000,1)
        self.net = models.resnet18(pretrained)
        # self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net.avgpool = nn.AdaptiveAvgPool2d(1)
        # =============================================================================
        #         if self.adjusted == True:
        #             self.net.avgpool = nn.AvgPool2d(kernel_size=16, stride=1, padding=0)
        # =============================================================================
        self.net.fc = nn.Linear(512, self.output_size)
        
        #self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #x = self.dropout(x)
        #x = self.net.fc(x)
        return self.net(x)

class WideResNet50(nn.Module):
    def __init__(self, num_out, pretrained):  # adjustedpool = False,
        super(WideResNet50, self).__init__()
        self.output_size = num_out
        # self.layer1 = nn.Linear(1000,1)
        self.net = models.wide_resnet50_2(pretrained)
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
    def __init__(self, num_out, pretrained):  # adjustedpool = False,
        super(ResNext50, self).__init__()
        self.output_size = num_out
        # self.layer1 = nn.Linear(1000,1)
        self.net = models.resnext50_32x4d(pretrained)
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
    def __init__(self, num_out, pretrained):
        super(DenseNet121, self).__init__()
        self.output_size = num_out
        self.net = models.densenet121(pretrained)
        self.net.classifier = nn.Linear(1024, self.output_size)

    def forward(self, x):
        return self.net(x)

class EfficientNetB6(nn.Module):
    def __init__(self, num_out, pretrained):
        super(EfficientNetB6, self).__init__()
        self.output_size = num_out
        if pretrained:
            self.net = EfficientNet.from_pretrained('efficientnet-b6')
        else:
            self.net = EfficientNet.from_name('efficientnet-b6')
        self.net._fc = nn.Linear(2304, self.output_size)

    def forward(self, x):
        return self.net(x)

class SENet(nn.Module):
    def __init__(self, num_out, pretrained):
        super(SENet, self).__init__()
        self.output_size = num_out
        if pretrained:
            self.net = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')  # 'se_resnext50_32x4d'
        else:
            self.net = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=None)  # 'se_resnext50_32x4d'
        
        self.net.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.net.last_linear = nn.Linear(2048, self.output_size)

    def forward(self, x):
        #        x = F.relu(self.net(x))
        #        x = self.fc(x)
        return self.net(x)


