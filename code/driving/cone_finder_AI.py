import os
import numpy as np
import pickle
import random
import cv2
from time import time

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from PIL import Image

# Data settings

IN_IMG_DIMENSIONS = (672, 376)
OUT_IMG_DIMENSIONS = (336, 188)

# Loss weights
WEIGHT = 1.0

# Remaps values of masks to indices for use in CE loss
mapper_dict = {0:0, 100:1, 200:2} 
def remap_target(entry):
    return mapper_dict[entry]
remap_target = np.vectorize(remap_target)

dataTransformations = transforms.Compose([
    transforms.ColorJitter(brightness=(0.2,1.0)),
    transforms.ToTensor(),    
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def convert_to_tensor(image):
    return dataTransformations(image)

def clamp(value, minValue, maxValue):
    return max(min(maxValue, value), minValue)

class FaceNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc_in_conv1_ds = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)

        self.enc_b1_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)  
        self.enc_b1_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)    

        self.enc_b2_conv1_ds = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc_b2_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)    
        self.enc_b2_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) 
        self.enc_b2_conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  
        self.enc_b2_conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)     

        self.enc_latent_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.dec_deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)  

        self.dec_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1) 
        self.dec_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1) 
        
        self.dec_out_conv1 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)  
        self.dec_out_conv2 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)  

        self.layers = [self.enc_in_conv1_ds, 
                       self.enc_b1_conv1,
                       self.enc_b1_conv2,
                       self.enc_b2_conv1_ds,
                       self.enc_b2_conv2,
                       self.enc_b2_conv3,
                       self.enc_b2_conv4,
                       self.enc_b2_conv5,
                       self.enc_latent_conv1,
                       self.dec_deconv1,
                       self.dec_conv1,
                       self.dec_conv2,
                       self.dec_out_conv1,
                       self.dec_out_conv2]

    def forward(self, x):    
        x = F.elu(self.enc_in_conv1_ds(x))

        x = F.elu(self.enc_b1_conv1(x))
        x = F.elu(self.enc_b1_conv2(x))

        skip_x_1 = x 
        x = F.elu(self.enc_b2_conv1_ds(x))
        x = F.elu(self.enc_b2_conv2(x))
        x = F.elu(self.enc_b2_conv3(x)) 
        x = F.elu(self.enc_b2_conv4(x))
        x = F.elu(self.enc_b2_conv5(x))  

        x = F.elu(self.enc_latent_conv1(x))

        x = F.elu(self.dec_deconv1(x)) 
        x = F.elu(self.dec_conv1(x))
        x = F.elu(self.dec_conv2(x))
        x = x + skip_x_1

        x = F.elu(self.dec_out_conv1(x))
        x = self.dec_out_conv2(x)

        #x = torch.squeeze(x) NOTE: not needed since we have 3 channels


        return x

def load_model(path):
    model = FaceNet()
    model.load_state_dict(torch.load(path))
    model.eval()

    return model
            