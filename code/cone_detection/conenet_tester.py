import sys
import cv2
import pyzed.sl as sl
import numpy as np
import keyboard
import torchvision.transforms.functional as fn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as f
from PIL import Image
from cone_finder_AI import *

IN_IMG_DIMENSIONS = (672, 376)
OUT_IMG_DIMENSIONS = (336, 188)

cam = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.VGA
init_params.camera_fps = 60

err = cam.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Camera failed to open")
    sys.exit()

print("Opened Camera")

image = sl.Mat()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model("logs/conenet_30.pt").to(device)

while keyboard.is_pressed("a") == False:
    if cam.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(image, sl.VIEW.LEFT)
        image_as_array = image.get_data()[:, :, 0:3]
        image_as_array_rgb = cv2.cvtColor(image_as_array, cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(image_as_array_rgb)
        image_tensor = transforms.ToTensor()(PIL_image)

        with torch.no_grad():
            model_output = model(image_tensor.to(device))
            result_mask = torch.argmax(model_output, dim=0).cpu().detach().numpy().astype(np.uint8)
            
            cv2.imshow("Video", image_as_array)
            cv2.imshow("Mask", (result_mask/2))
            cv2.waitKey(1)

cam.close()
print("Closed Camera")