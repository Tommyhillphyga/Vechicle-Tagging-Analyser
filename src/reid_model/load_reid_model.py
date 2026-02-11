import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import timm
from .mixstyle import MixStyle

import os
import yaml
import sys


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        # For old pytorch, you may use kaiming_normal.
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear > 0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x
        

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False, linear_num=512,
                 model_subtype="50", mixstyle=True):
        super(ft_net, self).__init__()
        if model_subtype in ("50", "default"):
            if ibn:
                model_ft = torch.hub.load(
                    'XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
            else:
                model_ft = models.resnet50(weights="IMAGENET1K_V2")
        elif model_subtype == "101":
            if ibn:
                model_ft = torch.hub.load("XingangPan/IBN-Net", "resnet101_ibn_a", pretrained=True)
            else:
                model_ft = models.resnet101(weights="IMAGENET1K_V2")
        elif model_subtype == "152":
            if ibn:
                raise ValueError("Resnet152 has no IBN variants available.")
            model_ft = models.resnet152(weights="IMAGENET1K_V2")
        else:
            raise ValueError(f"Resnet model subtype: {model_subtype} is invalid, choose from: ['50','101','152'].")

        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(
            2048, class_num, droprate, linear=linear_num, return_f=circle)
        self.mixstyle = MixStyle(alpha=0.3) if mixstyle else None

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        if self.training and self.mixstyle:
            x = self.mixstyle(x)
        x = self.model.layer2(x)
        if self.training and self.mixstyle:
            x = self.mixstyle(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
    
def load_weights(model, ckpt_path):

    state = torch.load(ckpt_path, map_location=torch.device("cpu"))
    if model.classifier.classifier[0].weight.shape != state["classifier.classifier.0.weight"].shape:
        state["classifier.classifier.0.weight"] = model.classifier.classifier[0].weight
        state["classifier.classifier.0.bias"] = model.classifier.classifier[0].bias
    model.load_state_dict(state)
    return model

def create_model(n_classes, kind="resnet", **kwargs):
    """Creates a model of a given kind and number of classes"""
    if kind == "resnet":
        return ft_net(n_classes, **kwargs)

    else:
        raise ValueError("Model type cannot be created: {}".format(kind))
    

def load_model_from_opts(opts_file, ckpt=None, return_feature=False, remove_classifier=False):

    with open(opts_file, "r") as stream:
        opts = yaml.load(stream, Loader=yaml.FullLoader)
    n_classes = opts["nclasses"]
    droprate = opts["droprate"]
    stride = opts["stride"]
    linear_num = opts["linear_num"]

    model_subtype = opts.get("model_subtype", "default")
    model_type = opts.get("model", "resnet_ibn")
    mixstyle = opts.get("mixstyle", False)

    if model_type in ("resnet", "resnet_ibn"):
        model = create_model(n_classes, "resnet", droprate=droprate, ibn=(model_type == "resnet_ibn"),
                             stride=stride, circle=return_feature, linear_num=linear_num,
                             model_subtype=model_subtype, mixstyle=mixstyle)

    else:
        raise ValueError("Unsupported model type: {}".format(model_type))

    if ckpt:
        load_weights(model, ckpt)
    if remove_classifier:
        model.classifier.classifier = nn.Sequential()
        model.eval()
    return model

