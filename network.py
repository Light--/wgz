import os, sys, time

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.models as models

class LinearLayer(nn.Module):
    def __init__(self):
        super(LinearLayers,self).__init__()

    def forward(self,x):
        pass

class VGGEmbNet(nn.Module):
    def __init__(self):
        super(VGGEmbNet,self).__init__()
        self.preNet = models.vgg16(pretrained=True).features
        self.vgg = nn.Sequential()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.NSlices = 5
        for x in range(4):
            self.slice1.add_module(str(x), self.preNet[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), self.preNet[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), self.preNet[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), self.preNet[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), self.preNet[x])
        for param in self.preNet.parameters():
            param.requires_grad = False
        self.finalClassifier = nn.Sequential()

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out

class AlexEmbNet(nn.Module):
    def __init__(self):
        super(AlexEmbNet,self).__init__()

    def forward(self,x):
        pass

class SQEEmbNet(nn.Module):
    def __init__(self):
        super(SQEEmbNet,self).__init__()
        self.sqeNet = models.squeezenet1_1(pretrained=True)
    def forward(self,x):
        pass

# class MobilenetEmbNet(nn.Module):
#     def __init__(self):
#         super(MobilenetEmbNet,self).__init__()
#         self.mobileNet = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
#         self.transform = None
#     def forward(self,x):
#         pass

class ResnetEmbNet(nn.Module):
    def __init__(self):
        super(ResnetEmbNet,self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.NSlices = 5
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.endClassifier = nn.Sequential()

    def forward(self,x):
        pass


class SENETEmbNet(nn.Module):
    def __init__(self):
        super(SENETEmbNet,self).__init__()

    def forward(self,x):
        pass

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet,self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1,x2):
        out1 = self.embedding_net(x1)
        out2 = self.embedding_net(x2)
        return out1, out2

class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet,self).__init__()
        self.embedding_net = embedding_net

    def forward(self,x1,x2,x3):
        out1 = self.embedding_net(x1)
        out2 = self.embedding_net(x2)
        out3 = self.embedding_net(x3)
        return x1, x2, x3

