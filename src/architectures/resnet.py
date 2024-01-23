import torch
import torch.nn as nn

def get_configs(arch):

    # True or False means wether to use BottleNeck

    if arch == 'resnet18':
        return [2, 2, 2, 2], False
    elif arch == 'resnet34':
        return [3, 4, 6, 3], False
    elif arch == 'resnet50':
        return [3, 4, 6, 3], True
    elif arch == 'resnet101':
        return [3, 4, 23, 3], True
    elif arch == 'resnet152':
        return [3, 8, 36, 3], True
    else:
        raise ValueError("Undefined model")

class ResNetAutoEncoder(nn.Module):

    def __init__(self, layer_cfg, version, bottleneck):

        super(ResNetAutoEncoder, self).__init__()

        self.encoder = ResNetEncoder(layer_cfg=layer_cfg, version=version, bottleneck=bottleneck)
        self.decoder = ResNetDecoder(layer_cfg=layer_cfg[::-1], version=version, bottleneck=bottleneck)
    
    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

class ResNet(nn.Module):

    def __init__(self, layer_cfg, version, bottleneck, num_classes, dropout):
        super(ResNet, self).__init__()

        self.encoder = ResNetEncoder(layer_cfg, version, bottleneck)

        self.avpool = nn.AdaptiveAvgPool2d((1,1))

        if bottleneck:
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features=2048, out_features=num_classes)
            )
            
        else:
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features=512, out_features=num_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        x = self.encoder(x)

        x = self.avpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


class ResNetEncoder(nn.Module):

    def __init__(self, layer_cfg, version, bottleneck):
        super(ResNetEncoder, self).__init__()

        if len(layer_cfg) != 4:
            raise ValueError("Only 4 layers can be configued")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        if bottleneck:

            self.conv2 = EncoderBottleneckBlock(in_channels=64,   hidden_channels=64,  up_channels=256,  layers=layer_cfg[0], downsample_method="pool", version=version)
            self.conv3 = EncoderBottleneckBlock(in_channels=256,  hidden_channels=128, up_channels=512,  layers=layer_cfg[1], downsample_method="conv", version=version)
            self.conv4 = EncoderBottleneckBlock(in_channels=512,  hidden_channels=256, up_channels=1024, layers=layer_cfg[2], downsample_method="conv", version=version)
            self.conv5 = EncoderBottleneckBlock(in_channels=1024, hidden_channels=512, up_channels=2048, layers=layer_cfg[3], downsample_method="conv", version=version)

        else:

            self.conv2 = EncoderResidualBlock(in_channels=64,  hidden_channels=64,  layers=layer_cfg[0], downsample_method="pool", version=version)
            self.conv3 = EncoderResidualBlock(in_channels=64,  hidden_channels=128, layers=layer_cfg[1], downsample_method="conv", version=version)
            self.conv4 = EncoderResidualBlock(in_channels=128, hidden_channels=256, layers=layer_cfg[2], downsample_method="conv", version=version)
            self.conv5 = EncoderResidualBlock(in_channels=256, hidden_channels=512, layers=layer_cfg[3], downsample_method="conv", version=version)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

class ResNetDecoder(nn.Module):

    def __init__(self, layer_cfg, version, bottleneck):
        super(ResNetDecoder, self).__init__()

        if len(layer_cfg) != 4:
            raise ValueError("Only 4 layers can be configued")

        if bottleneck:

            self.conv1 = DecoderBottleneckBlock(in_channels=2048, hidden_channels=512, down_channels=1024, layers=layer_cfg[0], version=version)
            self.conv2 = DecoderBottleneckBlock(in_channels=1024, hidden_channels=256, down_channels=512,  layers=layer_cfg[1], version=version)
            self.conv3 = DecoderBottleneckBlock(in_channels=512,  hidden_channels=128, down_channels=256,  layers=layer_cfg[2], version=version)
            self.conv4 = DecoderBottleneckBlock(in_channels=256,  hidden_channels=64,  down_channels=64,   layers=layer_cfg[3], version=version)


        else:

            self.conv1 = DecoderResidualBlock(hidden_channels=512, output_channels=256, layers=layer_cfg[0], version=version)
            self.conv2 = DecoderResidualBlock(hidden_channels=256, output_channels=128, layers=layer_cfg[1], version=version)
            self.conv3 = DecoderResidualBlock(hidden_channels=128, output_channels=64,  layers=layer_cfg[2], version=version)
            self.conv4 = DecoderResidualBlock(hidden_channels=64,  output_channels=64,  layers=layer_cfg[3], version=version)

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
        )

        self.gate = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x

class EncoderResidualBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, layers, downsample_method, version):
        super(EncoderResidualBlock, self).__init__()

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, version=version, downsample=True)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, version=version, downsample=False)
                
                self.add_module('%02d EncoderLayer' % i, layer)
        
        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, version=version, downsample=False)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, version=version, downsample=False)
                
                self.add_module('%02d EncoderLayer' % (i+1), layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class EncoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, up_channels, layers, downsample_method, version):
        super(EncoderBottleneckBlock, self).__init__()

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=True, version=version)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False, version=version)
                
                self.add_module('%02d EncoderLayer' % i, layer)
        
        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False, version=version)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False, version=version)
                
                self.add_module('%02d EncoderLayer' % (i+1), layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x


class DecoderResidualBlock(nn.Module):

    def __init__(self, hidden_channels, output_channels, layers, version):
        super(DecoderResidualBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=output_channels, upsample=True, version=version)
            else:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=hidden_channels, upsample=False, version=version)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class DecoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, layers, version):
        super(DecoderBottleneckBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=down_channels, upsample=True, version=version)
            else:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=in_channels, upsample=False, version=version)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x


class EncoderResidualLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, downsample, version):
        super(EncoderResidualLayer, self).__init__()

        self.version = version

        if downsample:
            if version != 2.0:
                self.weight_layer1 = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                self.weight_layer1 = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
                )

        else:
            if version != 2.0:
                self.weight_layer1 = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                self.weight_layer1 = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                )

        if version != 2.0:
            self.weight_layer2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                
            )

        if downsample:
            if version != 2.0:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=hidden_channels),
                )
            else:
                self.downsample = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                    
                )
        else:
            self.downsample = None

        if version != 2.0:
            self.relu = nn.Sequential(
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        if self.version != 2.0:
            x = self.relu(x)

        return x

class EncoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, up_channels, downsample, version):
        super(EncoderBottleneckLayer, self).__init__()
        
        self.version = version

        if downsample:
            if version != 2.0:
                self.weight_layer1 = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                self.weight_layer1 = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                )
        else:
            if version != 2.0:
                self.weight_layer1 = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                self.weight_layer1 = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
                )

        if version != 2.0:
            self.weight_layer2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )

        if version != 2.0:
            self.weight_layer3 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
            )

        if downsample:
            if version != 2.0:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=up_channels),
                )
            else:
                self.downsample = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=2, padding=0, bias=False),
                )

        elif (in_channels != up_channels):
            self.downsample = None
            if version != 2.0:
                self.up_scale = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=up_channels),
                )
            else:
                self.up_scale = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
                )

        else:
            self.downsample = None
            self.up_scale = None

        if version != 2.0:
            self.relu = nn.Sequential(
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        elif self.up_scale is not None:
            identity = self.up_scale(identity)

        x = x + identity

        if self.version != 2.0:
            x = self.relu(x)

        return x

class DecoderResidualLayer(nn.Module):

    def __init__(self, hidden_channels, output_channels, upsample, version):
        super(DecoderResidualLayer, self).__init__()

        if version != 2.0:
            self.weight_layer1 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        if upsample:
            if version != 2.0:
                self.weight_layer2 = nn.Sequential(
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),              
                )
            else:
                self.weight_layer2 = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                    nn.BatchNorm2d(num_features=output_channels),
                    nn.ReLU(inplace=True),               
                )

        else:
            if version != 2.0:
                self.weight_layer2 = nn.Sequential(
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False),
                )
            else:
                self.weight_layer2 = nn.Sequential(
                    nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(num_features=output_channels),
                    nn.ReLU(inplace=True),
                )

        if upsample:
            if version != 2.0:
                self.upsample = nn.Sequential(
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1, stride=2, output_padding=1, bias=False)   
                )
            else:
                self.upsample = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1, stride=2, output_padding=1, bias=False),
                    nn.BatchNorm2d(num_features=output_channels),
                    nn.ReLU(inplace=True),
                )
        else:
            self.upsample = None
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity

        return x

class DecoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, upsample, version):
        super(DecoderBottleneckLayer, self).__init__()

        self.version = version

        if version != 2.0:
            self.weight_layer1 = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        if version != 2.0:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        if upsample:
            if version != 2.0:
                self.weight_layer3 = nn.Sequential(
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
                )
            else:
                self.weight_layer3 = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False),
                    nn.BatchNorm2d(num_features=down_channels),
                    nn.ReLU(inplace=True),
                )
        else:
            if version != 2.0:
                self.weight_layer3 = nn.Sequential(
                    nn.BatchNorm2d(num_features=hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
                )
            else:
                self.weight_layer3 = nn.Sequential(
                    nn.Conv2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=down_channels),
                    nn.ReLU(inplace=True),
                )


        if upsample:
            if version != 2.0:
                self.upsample = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
                )
            else:
                self.upsample = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False),
                    nn.BatchNorm2d(num_features=down_channels),
                    nn.ReLU(inplace=True),
                )
        elif (in_channels != down_channels):
            self.upsample = None
            if version != 2.0:
                self.down_scale = nn.Sequential(
                    nn.BatchNorm2d(num_features=in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
                )
            else:
                self.down_scale = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_features=down_channels),
                    nn.ReLU(inplace=True),
                )
        else:
            self.upsample = None
            self.down_scale = None
        
        if version == 2.0:
            self.relu = nn.Sequential(
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.upsample is not None:
            identity = self.upsample(identity)
        elif self.down_scale is not None:
            identity = self.down_scale(identity)

        x = x + identity

        if self.version == 2.0:
            x = self.relu(x)

        return x

if __name__ == "__main__":

    version = 2.0

    layer_cfg, bottleneck = get_configs("resnet50")

    encoder = ResNetEncoder(layer_cfg, version, bottleneck)

    input = torch.randn((5,3,224,224))

    print(input.shape)

    output = encoder(input)

    print(output.shape)

    decoder = ResNetDecoder(layer_cfg[::-1], version, bottleneck)

    output = decoder(output)

    print(output.shape)

    #from torchsummary import summary
    #summary(encoder)
    #summary(decoder)