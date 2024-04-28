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

# import pytorch_lightning as pl

# Don't generate pyc codes
sys.dont_write_bytecode = True


class DeepVONet(nn.Module):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()


## Yet to build VIO net with Super point based concept
