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

init_params.sdk_verbose = True # Enable verbose logging
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Set the depth mode to performance (fastest)
init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units

err = cam.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Camera failed to open")
    sys.exit()

print(cam.getCameraInformation())
print("Opened Camera")

image = sl.Mat()
depth = sl.Mat()
normalized_depth = sl.Mat()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model("logs/conenet_30.pt").to(device)

while keyboard.is_pressed("a") == False:
    if cam.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
        # Normal view
        cam.retrieve_image(image, sl.VIEW.LEFT)
        image_as_array = image.get_data()
        
        # Depth
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
        depth_as_array = depth.get_data()

        # Normalized Depth for viewing
        cam.retrieve_image(normalized_depth, sl.VIEW.DEPTH)
        normalized_depth_as_array = normalized_depth.get_data()
        
        # Preping input
        image_as_array_rgb = cv2.cvtColor(image_as_array[:, :, 0:3], cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(image_as_array_rgb)
        image_tensor = transforms.ToTensor()(PIL_image)

        # Running model
        with torch.no_grad():
            model_output = model(image_tensor.to(device))
            model_output = model_output.softmax(dim=0)
            #model_output[1] = nn.Threshold(0.3, 0.0)(model_output[1])
            #model_output[2] = nn.Threshold(0.5, 0.0)(model_output[2])

            result_mask = torch.argmax(model_output, dim=0).cpu().detach().numpy().astype(np.uint8)
            result_mask = cv2.resize(result_mask, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
            
            contours, hierarchy = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(0, len(contours)):
                if cv2.contourArea(contours[i]) > 100: # Minimum number of pixels in countour
                    x, y, w, h = cv2.boundingRect(contours[i])
                    x, y, w, h = x, y, w, h

                    if result_mask[contours[i][0][0][1], contours[i][0][0][0]] == 2: # orange
                        cv2.rectangle(image_as_array, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    elif result_mask[contours[i][0][0][1], contours[i][0][0][0]] == 1: # green
                        cv2.rectangle(image_as_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    distance = depth_as_array[int(y + h/2), int(x + w/2)]
                    try:
                        image_as_array = cv2.putText(image_as_array, f"{int(distance)} mm", (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX ,  0.5, (255, 0, 0), 1, cv2.LINE_AA) 
                    except:
                        print("Error with depth")

            cv2.imshow("Video", image_as_array)
            cv2.imshow("Mask", (result_mask/2))
            cv2.waitKey(1)

cam.close()
print("Closed Camera")