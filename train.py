import sys 
import os 
import glob
import time
import math

from os import listdir

import shutil 
import string
import argparse

import random 
import numpy as np
import cv2 
import skimage
import PIL
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms as tf
from torch.optim import AdamW
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from tqdm.notebook import tqdm
from os import listdir
from lib.network import VONet, IONet, VIONet
import pandas as pd
from torchsummary import summary 
from lib.helpers import * 
from lib.data_helpers import *


def TrainOperation(Batches, Epochs, lr, latestmodelpath, checkpointpath, logspath, device, modeltype):
    """_summary_

    Args:
        Batches (toorch list): _input generated batches
        epochs (float): for hoow many epochs the model should run
        lr (Float): learning rate in optimizer
        latestmodelpath (str): if we want to continue training from a saved model
        checkpointpath (str):where the checkpoints should be saved
        logspath (str): where the tensorlogs should be saved
        device (str): device to run the model on
    """
    
    total_batches = len(Batches)
    lossvalues_eachepoch = []
    
    #initialize model
    if modeltype == "vonet":
        model = VONet(input_channels=2) # 2 input channels  # For RGB use 6 channels
        model = model.to(device)
        
    
    if modeltype == "ionet":
        model = IONet(6, 64, 2, 4) # 2 input channels
        model = model.to(device)
        
    if modeltype == "vionet":
        model = VIONet(6, 64,2, 4) # 2 input channels
        model = model.to(device)

    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    #summary(model, (2, 512, 512))
    
    print("Model Initialized")
    
    if latestmodelpath is not None:
        checkpoint = torch.load(latestmodelpath)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    print("training started...")
        
    for epochs in tqdm(range(Epochs)):
        cumulative_Loss = 0
        for batch in Batches:
            
            #get the batch
            if modeltype == "vonet":
                images = batch[0].to(device)
            if modeltype == "ionet":
                IMU = batch[1].to(device)
                
            if modeltype == "vionet":
                images = batch[0].to(device)
                IMU = batch[1].to(device)
            
            pos_t = batch[2].to(device)
            pos_q = batch[3].to(device)
            
            #forward pass
            if modeltype == "vonet":
                pred_q, pred_t = model(images)
            if modeltype == "ionet":
                pred_q, pred_t = model(IMU)
            if modeltype == "vionet":
                pred_q, pred_t = model(images, IMU)
            
            
            #backward pass
            optimizer.zero_grad()
            
            loss = loss_function_1(pred_q, pred_t, pos_q, pos_t)
            loss.backward()
            optimizer.step() 
            
            cumulative_Loss = cumulative_Loss+loss
        
        epoch_avg_loss = cumulative_Loss/total_batches
        
        print(f"Epoch: {epochs}, Loss: {epoch_avg_loss}")
        
        #saving the model checkpoint
        savename = checkpointpath + "checkpoint_"+ modeltype + str(epochs) + ".ckpt"
        
        if epochs % 20 == 0:
            torch.save({'epoch': Epochs,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_avg_loss},
                        savename)
        print('\n' + savename + ' Model Saved...')
        
        #saving the loss values in an array for plotting
        lossvalues_eachepoch.append(epoch_avg_loss)
        
    return lossvalues_eachepoch

def plot_loss(Loss):
    Loss = [l.item() for l in Loss]
    plt.plot(np.arange(len(Loss)), Loss, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.legend()
    plt.show()
            
    
   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="Path to the images folder", default=r"data_t/images")
    parser.add_argument("--IMU", type=str, help="path to imu data", default=r"data_t/imu.csv")
    parser.add_argument("--rotations", type=str, help="path to relative pose data", default=r"data_t/relative_pose.csv")
    parser.add_argument("--minibatch", type=int, help="Minibatch size", default=32)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=10000)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.001)
    parser.add_argument("--latestmodelpath", type=str, help="folder to load checkpoint and start training from that epoch", default=None)
    parser.add_argument("--checkpointpath", type=str, help="where should the checkpoints save", default="checkpoints/")
    parser.add_argument("--logspath", type=str, help="where the tensorlogs should save", default="tensor_logs/")
    parser.add_argument("--model", type=str, help="which model to run", default="ionet")
    args = parser.parse_args()

    #loading data
    images = LoadImages(args.images)
    IMU = pd.read_csv(args.IMU)
    IMU_array = IMU.values # IMU with time values
    pose = pd.read_csv(args.rotations)
    pose_array = pose.values
    minibatch = args.minibatch # rotation with time values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #generating batches
    Batches = BatchGen(images, IMU_array, pose_array, minibatch)
    
    #train operation
    Loss = TrainOperation(Batches, args.epochs, args.lr, args.latestmodelpath, args.checkpointpath, args.logspath, device, args.model)
    
    #plotting losses
    plot_loss(Loss)
    
    
    
if __name__ == "__main__":
    main()