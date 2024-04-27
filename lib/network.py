"""
Project 4: Deep VIO


Author(s):
Uday Girish Maradana
Pradnya Shinde
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

# Don't generate pyc codes
# sys.dont_write_bytecode = True


class DeepVONet(pl.LightningModule):
    def __init__(self, InputSize, OutputSize):
        """                                                                         
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output        
        """
        super().__init__()
        self.ConvLayers = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1),
        nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 7) # 6 DOF Pose
        )

    def forward(self, x):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        x = self.ConvLayers(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)

        return x
   
    def training_step(self, batch, batch_idx):
        x, y = batch 
        output = self.forward(x)
        loss = F.mse_loss(output, y)
        self.log("train", loss)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class DeepIONet(DeepVONet):
    def __init__(self, InputSize, OutputSize):
        # super().__init__(hparams)
        self.ConvLayers = nn.Sequential(
        nn.Conv1d(in_channels=3, out_channels=256, kernel_size=2, stride=2, padding=3),
        nn.ReLU(),
        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=1),
        nn.ReLU(),
        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=1),
        nn.ReLU(),
        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=1),
        nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.ConvLayers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch 
        output = self.forward(x)
        loss = F.mse_loss(output, y)
        self.log("train", loss)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer





class DeepVIONet(DeepIONet):
    def __init__(self, InputSize, OutputSize):
        # super().__init__(hparams)
        self.net1 = DeepVONet()
        self.net2 = DeepIONet()
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
            nn.ReLU(),
        )

    def forward(self, x1, x2):

        xVO = self.net1.forward(x1)
        xIO = self.net2.forward(x2)
        concat_vio = torch.cat((xVO, xIO), dim=1)
        output = self.fc_layers(concat_vio)

        return output
    
    def training_step(self, batch1, batch2, batch1_idx, batch2_idx):
        # x: data, y:label
        x1, y1 = batch1
        x2, y2 = batch2 
        output = self.forward(x1, x2)
        loss = F.mse_loss(output, y1, y2)
        self.log("train", loss)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer