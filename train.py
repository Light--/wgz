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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np

from custom_datasets import siameseDataset,tripletDataset, datasetGen
from network import VGGEmbNet

from pytorch_metric_learning import losses as losses
from pytorch_metric_learning import miners
import lpips

import PIL
from PIL import Image

tb = SummaryWriter()

dset = datasetGen()
dsetLoader = DataLoader(dset,5,True)
emb_net = VGGEmbNet()

batch = next(iter(dsetLoader))
pred = emb_net(batch)