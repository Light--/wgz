import time, sys, os
from itertools import product
from collections import OrderedDict, namedtuple
import logging

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision as tv
import torchvision.models as models
import torchvision.datasets as dset
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms as transforms
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


import numpy as np

from custom_datasets import siameseDataset,tripletDataset, datasetGen
from network import VGGEmbNet

# from pytorch_metric_learning import losses as losses
# from pytorch_metric_learning import miners
# import lpips

import PIL
from PIL import Image

from network import ResnetEmbNet, AlexEmbNet, SQEEmbNet, VGGEmbNet, SiameseNet, TripletNet
from losses import ContrastiveLoss, TripletLoss

params = OrderedDict(
    lr = [.01,.001,.0001],
    batch_size = [10,100,250,500],
    arch_type = ['siameseNet','tripletNet'],
    embs_net = ['Resnet','vgg','alex','sqe'],
    lpips_like = [True, False]
)

class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run',params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# N_epoch = 15
# learning_rate = 0.001

class Trainer():
    pass
def main():
    NEpoch = 30
    logs_filename = 'results.log'
    logging.basicConfig(filename=logs_filename, level= logging.DEBUG,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    tb = SummaryWriter()
    
    def train_epoch(train_loader):
        embedding_net.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            embs = embedding_net(img)
            loss_output = loss(sig(embs),sig(lbl))
            total_loss += loss_output
            optimizer.zero_grad()
            loss_output.backward()
            optimizer.step()
        return total_loss
    def test_epoch():
        with torch.no_grad():
            pass
    for epoch in range(N_epoch):
        embedding_net = ResnetEmbNet()
        optimizer = optim.Adam(embedding_net.endClassifier.parameters(),learning_rate)

if __name__ == "__main__":
    main()