from . import vgg, resnet, densenet

def architecture_builder(arch='resnet50', version=2.0, dropout=0.2, num_classes=2):

    if arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        layer_cfg = vgg.get_configs(arch)
        model = vgg.VGG(layer_cfg, num_classes, dropout)

    elif arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        layer_cfg, bottleneck = resnet.get_configs(arch)
        model = resnet.ResNet(layer_cfg, bottleneck, version, num_classes, dropout)

    elif arch in ["vgg11autenc", "vgg13autenc", "vgg16autenc", "vgg19autenc"]:
        layer_cfg = vgg.get_configs(arch)
        model = vgg.VGGAutoEncoder(layer_cfg)

    elif arch in ["resnet18autenc", "resnet34autenc", "resnet50autenc", "resnet101autenc", "resnet152autenc"]:
        layer_cfg, bottleneck = resnet.get_configs(arch)
        model = resnet.ResNetAutoEncoder(layer_cfg, bottleneck, version)
    
    elif arch in ["densenet121","densenet169","densenet201","densenet264"]:
        layer_cfg = densenet.get_configs(arch)
        model = densenet.DenseNet(layer_cfg, num_classes, dropout)

    elif arch in ["densenet121autenc","densenet169autenc","densenet201autenc","densenet264autenc"]:
        layer_cfg = densenet.get_configs(arch)
        model = densenet.DenseNetAutoEncoder(layer_cfg)

    else:
        raise ValueError("Undefined model")
    
    return model
    