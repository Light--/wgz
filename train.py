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


logs_filename = 'results.log'
logging.basicConfig(filename=logs_filename, level= logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

params = OrderedDict(
    model = ['siameseNet', 'tripletNet'],
    network = ['sqe','alex','resnet'],
    lr = [.01, .001, .0001],
    if model == 'siameseNet':
        batch_size = [100, 150, 200],
    elif model == 'tripletNet':
        batch_size = [12,24,36],
    # lpips_like = [True, False], ####### NEXT STEP ######
    test_shuffle = [True, False],
    margin = [0.5, 1., 1.5]
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
    def __init__(self):
        self.epoch_number = 0
        self.epoch_trainLoss = 0
        self.epoch_testLoss = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_number = 0
        self.run_data = []
        self.run_start_time = 0

        self.network = None
        self.tb = None

    def begin_run(self, run):
        self.run_start_time = time.time()

        self.run_params = run
        self.run_number += 1

        self.tb = SummaryWriter(comment=f'-{run}')
    
    def end_run(self):
        self.tb.close()
        self.epoch_number = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_number += 1
        self.epoch_trainLoss = 0
        self.epoch_testLoss = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        trainLoss = self.epoch_trainLoss.item()
        testLoss = self.epoch_testLoss.item()

        self.tb.add_scalar('Train Loss:', trainLoss, self.epoch_number)
        self.tb.add_scalar('Test Loss:', testLoss, self.epoch_number)

        results = OrderedDict()
        results['run'] = self.run_number
        results['epoch'] = self.epoch_number
        results['train loss'] = trainLoss
        results['test loss'] = testLoss
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

        message = f"{self.run_params}\n Run Data: {results}"
        logging.info(message)
    def track_trainLoss(self, trainLoss):
        self.epoch_trainLoss = trainLoss # edittable

    def track_testLoss(self, testLoss):
        self.epoch_testLoss = testLoss # edittable

    def save_model(self):
        pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # logs_filename = 'results.log'
    # logging.basicConfig(filename=logs_filename, level= logging.DEBUG,
    #                     format='%(asctime)s:%(levelname)s:%(message)s')
    # tb = SummaryWriter(logs_filename)

    m = RunManager()
    for run in RunBuilder.get_runs(params):
        if run.network == 'sqe':
            embedding_net = SQEEmbNet()
        elif run.network == 'alex':
            embedding_net = AlexEmbNet()
        elif run.network == 'resnet':
            embedding_net = ResnetEmbNet()
        embedding_net = embedding_net.to(device)

        if run.model == 'siameseNet':
            model = SiameseNet(embedding_net)
            Dset = siameseDataset()
            train_set, test_set = torch.utils.data.random_split(Dset,[10000,3203])
        elif run.model == 'tripletNet':
            model = TripletNet(embedding_net)
            Dset = tripletDataset()
            train_set, test_set = torch.utils.data.random_split(Dset,[120,43])
        model = model.to(device)
        trainLoader = DataLoader(train_set, batch_size=run.batch_size, shuffle=True)
        testLoader = DataLoader(test_set, batch_size=run.batch_size, shuffle = run.test_shuffle)
        m.begin_run(run)
        def train_epoch(trainLoader):
            embedding_net.train()
            train_loss = 0
            optimizer = optim.Adam(model.parameters(),run.lr)
            for batch_idx, data in enumerate(trainLoader):
                img1 = data[0][0].to(device)
                img2 = data[0][1].to(device)
                if run.model == 'siameseNet':
                    target = data[1].to(device)
                    preds = model(img1,img2)
                    constLoss = ContrastiveLoss(run.margin)
                    loss_output = constLoss(preds[0],preds[1],target)
                elif run.model == 'tripletNet':
                    img3 = data[0][2].to(device)
                    preds = model(img1,img2,img3)
                    tripLoss = TripletLoss(run.margin)
                    loss_output = tripLoss(preds[0],preds[1],preds[2])
                train_loss += loss_output
                optimizer.zero_grad()
                loss_output.backward()
                optimizer.step()
            return train_loss
        def test_epoch(testLoader):
            with torch.no_grad():
                model.eval()
                test_loss = 0
                for batch_idx, data in enumerate(testLoader):
                    img1 = data[0][0].to(device)
                    img2 = data[0][1].to(device)
                    if run.model == 'siameseNet':
                        target = data[1].to(device)
                        preds = model(img1,img2)
                        constLoss = ContrastiveLoss(run.margin)
                        loss_output = constLoss(preds[0],preds[1],target)
                    elif run.model == 'tripletNet':
                        img3 = data[0][2].to(device)
                        preds = model(img1,img2,img3)
                        tripLoss = TripletLoss(run.margin)
                        loss_output = tripLoss(preds[0],preds[1],preds[2])
                    test_loss += loss_output
                return test_loss
        for epoch in range(5):
            m.begin_epoch()
            train_loss = train_epoch(trainLoader)
            test_loss = test_epoch(testLoader)
            m.track_trainLoss(train_loss)
            m.track_testLoss(test_loss)
            # message = 'Epoch: {}/{}, Train Loss: {:.4f}'.format(epoch+1,N_epoch,train_loss)
            # message += '\nEpoch: {}/{}, Test Loss: {:.4f}'.format(epoch+1,N_epoch,val_loss)
            # message += '\nEpoch: {}/{}, Duration: {:.4f}'.format(epoch+1, N_epoch, epoch_end)
            # logging.info(message)
            m.end_epoch()
            # m.save_log(run)
        m.end_run()

if __name__ == "__main__":
    main()