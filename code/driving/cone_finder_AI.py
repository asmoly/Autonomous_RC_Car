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

# def draw_labels(image, result):    
#     for i in range(0, MAX_NUM_OF_FACES):
#         image = cv2.rectangle(image, (int(result[0 + i*4]), int(result[1 + i*4])), (int(result[0 + i*4]) + int(result[2 + i*4]), int(result[1 + i*4]) + int(result[3 + i*4])), (0, 255, 0), 2)

#     return image

class FacesDataset(Dataset):
    def __init__(self, path_to_targets, path_to_images, path_to_dataset_length, imageTransform=None):
        self.imageTransform = imageTransform
        self.path_to_images = path_to_images
        self.path_to_targets = path_to_targets

        self.dataset_length = pickle.load(open(path_to_dataset_length, "rb"))
        print(self.dataset_length)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        #image = cv2.imread(f"{self.path_to_images}/image_{idx}.png")
        image = Image.open(f"{self.path_to_images}/image_{idx}.png")
        image_as_tensor = self.imageTransform(image)

        target = cv2.imread(f"{self.path_to_targets}/target_{idx}.png", cv2.IMREAD_GRAYSCALE)
        target = cv2.resize(target, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST) 
        target = remap_target(target)
        target = torch.Tensor(target)
        target = target.type(torch.LongTensor) # Casting to int

        return image_as_tensor, target

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
            
# start_epoch = 1 at the very begining!
def train(device, start_epoch, n_epochs, path_to_targets, path_to_images, path_to_val_targets, path_to_val_images, path_to_data_length, path_to_val_data_length, path_to_logs, path_to_model = None):
    writer = SummaryWriter(log_dir=path_to_logs)

    data_loader = DataLoader(FacesDataset(imageTransform=dataTransformations, path_to_targets=path_to_targets, path_to_images=path_to_images, path_to_dataset_length=path_to_data_length), batch_size=16, shuffle=True)
    validation_data_loader = DataLoader(FacesDataset(imageTransform=dataTransformations, path_to_targets=path_to_val_targets, path_to_images=path_to_val_images, path_to_dataset_length=path_to_val_data_length), batch_size=16, shuffle=True)

    model = FaceNet().to(device)
    if path_to_model != None :
        model = load_model(path_to_model).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    startTime = time()
    idx_counter = 0
    for epoch_index in range (start_epoch, start_epoch + n_epochs):
        print(f"Epoch {epoch_index}, lr {optimizer.param_groups[0]['lr']}")

        loss_list = []
        for images, targets in data_loader:
            batchImages = images.to(device)
            batchTargets = targets.to(device)

            optimizer.zero_grad()

            batchOuput = model(batchImages)
            loss_ce = nn.CrossEntropyLoss(reduction="mean")(batchOuput, batchTargets)
            total_loss = WEIGHT*loss_ce
            loss_list.append(total_loss)

            total_loss.backward()
            optimizer.step()

            if idx_counter % 20 == 0:
                # Logging
                image_to_log = images[0].permute(1, 2, 0).cpu().detach().numpy()
                image_to_log *= 255.0   # de-normalize
                image_to_log = image_to_log.astype(np.uint8).copy()

                writer.add_scalar("Loss_ce", loss_ce, idx_counter)
                
                output = torch.argmax(batchOuput[0], dim=0)
                writer.add_image("image", transforms.ToTensor()(image_to_log), idx_counter)
                writer.add_image("target_mask", targets[0].cpu()/2.0, idx_counter, dataformats="HW")
                writer.add_image("result", output.cpu()/2, idx_counter, dataformats="HW")
                writer.flush()
            
            idx_counter += 1
            print("Epoch: ", epoch_index, "Idx", idx_counter, " Total Loss: ", total_loss.item())

        average_loss = 0
        for loss in loss_list:
            average_loss += loss
        average_loss = average_loss/len(loss_list)
        writer.add_scalar("Loss_ce_Per_Epoch", average_loss, epoch_index)
        writer.flush()

        loss_list = []
        for images, targets in validation_data_loader:
            batchImages = images.to(device)
            batchTargets = targets.to(device)

            with torch.no_grad():
                batchOuput = model(batchImages)
                loss_ce = nn.CrossEntropyLoss(reduction="mean")(batchOuput, batchTargets)
                total_loss = WEIGHT*loss_ce
                loss_list.append(total_loss)

        average_loss = 0
        for loss in loss_list:
            average_loss += loss
        average_loss = average_loss/len(loss_list)
        writer.add_scalar("Validation_Loss_ce_Per_Epoch", average_loss, epoch_index)
        writer.flush()

        # Save model checkpoint every 5th epoch
        #if epoch_index % 5 == 0:
        torch.save(model.state_dict(), os.path.join(path_to_logs, "conenet_{}.pt".format(epoch_index))) 

    writer.close()
    print("Training Time: ", (time() - startTime)/60)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train(device, start_epoch = 0, n_epochs = 200,
          path_to_targets="dataset/targets",
          path_to_images="dataset/images",
          path_to_logs="logs",
          path_to_model = None,
          path_to_val_targets="dataset/validation_targets",
          path_to_val_images="dataset/validation_images",
          path_to_data_length="dataset/image_count",
          path_to_val_data_length="dataset/validation_image_count")

if __name__ == "__main__":
    main()