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
    batch_size = [10,25,100],
    arch_type = ['siameseNet','tripletNet'],
    emb_net = ['sqe','alex', 'resnet']
    # lpips_like = [True, False],
    margin = [0.5,1.,1.5]
)

class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run',params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs

class RunManager():
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_epoch = 15
learning_rate = 0.001
batch_size =15

def main():
    logs_filename = 'results.log'
    logging.basicConfig(filename=logs_filename, level= logging.DEBUG,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    tb = SummaryWriter()
    
    # embedding_net = SQEEmbNet()
    # embedding_net = AlexEmbNet()
    embedding_net = ResnetEmbNet()
    embedding_net = embedding_net.to(device)

    # if siam
    # model = SiameseNet(embedding_net)
    # Dset = siameseDataset()
    # train_set, test_set = torch.utils.data.random_split(Dset,[10000,3203])
    
    #if triplet
    model = TripletNet(embedding_net)
    Dset = tripletDataset()
    train_set, test_set = torch.utils.data.random_split(Dset,[120,43])

    model = model.to(device)
    trainLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    def train_epoch(trainLoader):
        embedding_net.train()
        total_loss = 0
        optimizer = optim.Adam(model.parameters(),learning_rate)
        for batch_idx, data in enumerate(trainLoader):
            img1 = data[0][0].to(device)
            img2 = data[0][1].to(device)

            #if siam
            # target = data[1].to(device)
            # preds = model(img1,img2)
            # constLoss = ContrastiveLoss(1.)
            # loss_output = constLoss(preds[0],preds[1],target)
            
            #if trip
            img3 = data[0][2].to(device)
            preds = model(img1,img2,img3)
            tripLoss = TripletLoss(1.)
            loss_output = tripLoss(preds[0],preds[1],preds[2])
            
            total_loss += loss_output
            optimizer.zero_grad()
            loss_output.backward()
            optimizer.step()
        return total_loss
    def test_epoch():
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for batch_idx, data in enumerate(testLoader):
                img1 = data[0][0].to(device)
                img2 = data[0][1].to(device)

                #if siam
                # target = data[1].to(device)
                # preds = model(img1,img2)
                # constLoss = ContrastiveLoss(1.)
                # loss_output = constLoss(preds[0],preds[1],target)
                
                #if trip
                img3 = data[0][2].to(device)
                preds = model(img1,img2,img3)
                tripLoss = TripletLoss(1.)
                loss_output = tripLoss(preds[0],preds[1],preds[2])
                
                val_loss += loss_output
            return val_loss
    for epoch in range(N_epoch):
        epoch_start_time = time.time()
        train_loss = train_epoch(trainLoader)
        message = 'Epoch: {}/{}, Train Loss: {:.4f}, Accuracy:'.format(epoch+1,N_epoch,train_loss)
        # val_loss = test_epoch()
        # message += '\nEpoch: {}/{}, Test Loss: {:.4f}, Accuracy:'.format(epoch+1,N_epoch,val_loss)
        epoch_end = time.time() - epoch_start_time
        message += '\nEpoch: {}/{}, Duration: {:.4f}'.format(epoch+1, N_epoch, epoch_end)
        logging.info(message)
        tb.add_scalar('SQE_Siam Training Loss',train_loss,epoch)
        tb.close()


if __name__ == "__main__":
    main()