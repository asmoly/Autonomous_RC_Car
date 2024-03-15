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

from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from utils import *

# Data settings
IN_IMG_SIZE = 1024
OUT_IMG_SIZE = 256

IN_IMG_DIMENSIONS = (672, 376)
OUT_IMG_DIMENSIONS = (336, 188)

# Loss weights
WEIGHT = 1.0

dataTransformations = transforms.Compose([transforms.ColorJitter(brightness=(0.5,1.0)), transforms.ToTensor()])

def convert_to_tensor(image):
    return dataTransformations(image)

def clamp(value, minValue, maxValue):
    return max(min(maxValue, value), minValue)

# def draw_labels(image, result):    
#     for i in range(0, MAX_NUM_OF_FACES):
#         image = cv2.rectangle(image, (int(result[0 + i*4]), int(result[1 + i*4])), (int(result[0 + i*4]) + int(result[2 + i*4]), int(result[1 + i*4]) + int(result[3 + i*4])), (0, 255, 0), 2)

#     return image

class FacesDataset(Dataset):
    def __init__(self, pathToTargets, pathToImages, imageTransform=None):
        self.imageTransform = imageTransform
        self.pathToImages = pathToImages
        self.pathToTargets = pathToTargets

        self.dataset_length = pickle.load(open("dataset/image_count", "rb"))

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        #image = cv2.imread(f"{self.pathToImages}/image_{idx}.png")
        image = Image.open(f"{self.pathToImages}/image_{idx}.png")
        image_as_tensor = self.imageTransform(image)

        target = cv2.imread(f"{self.pathToTargets}/target_{idx}.png", cv2.IMREAD_GRAYSCALE)
        target = cv2.resize(target, (0, 0), fx=0.5, fy=0.5)
        target = target/255.0

        return image_as_tensor, torch.Tensor(target)

class FaceNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc_in_conv1_ds = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)

        self.enc_b1_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)  
        self.enc_b1_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)    

        self.enc_b2_conv1_ds = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc_b2_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)    
        self.enc_b2_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)    

        self.enc_latent_conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.dec_deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)  

        self.dec_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1) 
        self.dec_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1) 
        
        self.dec_out_conv1 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)  
        self.dec_out_conv2 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)  

        self.layers = [self.enc_in_conv1_ds, 
                       self.enc_b1_conv1,
                       self.enc_b1_conv2,
                       self.enc_b2_conv1_ds,
                       self.enc_b2_conv2,
                       self.enc_b2_conv3,
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

        x = F.elu(self.enc_b2_conv1_ds(x))
        x = F.elu(self.enc_b2_conv2(x))
        x = F.elu(self.enc_b2_conv3(x)) 
        #skip_x_1 = x  

        x = F.elu(self.enc_latent_conv1(x))

        x = F.elu(self.dec_deconv1(x)) 
        x = F.elu(self.dec_conv1(x))
        x = F.elu(self.dec_conv2(x))
        #x = x + skip_x_1

        x = F.elu(self.dec_out_conv1(x))
        x = self.dec_out_conv2(x)

        x = torch.squeeze(x)

        return x

def load_model(path):
    model = FaceNet()
    model.load_state_dict(torch.load(path))
    model.eval()

    return model
            
# start_epoch = 1 at the very begining!
def train(device, start_epoch, n_epochs, pathToTargets, pathToImages, pathToLogs, pathToModel = None):
    writer = SummaryWriter(log_dir=pathToLogs)

    dataLoader = DataLoader( FacesDataset(imageTransform=dataTransformations, 
        pathToTargets=pathToTargets, 
        pathToImages=pathToImages), 
        batch_size=16, shuffle=True)
    
    model = FaceNet().to(device)
    if pathToModel != None :
        model = load_model(pathToModel).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    startTime = time()
    idx_counter = 0
    for epoch_index in range (start_epoch, start_epoch + n_epochs):
        print(f"Epoch {epoch_index}, lr {optimizer.param_groups[0]['lr']}")

        #if epoch_index % 5 == 0:
            # Log weight and gradient histograms
            #log_weight_histograms(writer, epoch_index, model)
            #log_gradient_histograms(writer, epoch_index, model)

        for images, targets in dataLoader:
            batchImages = images.to(device)
            batchTargets = targets.to(device)

            optimizer.zero_grad()

            batchOuput = model(batchImages)
            loss_ce = nn.CrossEntropyLoss(reduction="mean")(batchOuput, batchTargets)
            total_loss = WEIGHT*loss_ce

            total_loss.backward()
            optimizer.step()

            if idx_counter % 20 == 0:
                # Logging
                image_to_log = images[0].permute(1, 2, 0).cpu().detach().numpy()
                image_to_log *= 255.0   # de-normalize
                image_to_log = image_to_log.astype(np.uint8).copy()

                writer.add_scalar("Loss_ce", loss_ce, idx_counter)
                
                writer.add_image("image", transforms.ToTensor()(image_to_log), idx_counter)
                writer.add_image("result_mask", targets[0].cpu()*255.0, idx_counter, dataformats="HW")
                writer.add_image("target_mask", batchOuput[0]*255.0, idx_counter, dataformats="HW")
                writer.flush()
                print("Logged data")
            
            idx_counter += 1
            print("Epoch: ", epoch_index, "Idx", idx_counter, " Total Loss: ", total_loss.item())

        # Save model checkpoint every 5th epoch
        if epoch_index % 5 == 0:
            torch.save(model.state_dict(), os.path.join(pathToLogs, "conenet_{}.pt".format(epoch_index))) 

    writer.close()
    print("Training Time: ", (time() - startTime)/60)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train(device, start_epoch = 0, n_epochs = 100,
          pathToTargets="dataset/targets",
          pathToImages="dataset/images",
          pathToLogs="logs",
          pathToModel = "logs/conenet_30.pt")

if __name__ == "__main__":
    main()