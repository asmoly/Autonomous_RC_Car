import cv2
import numpy as np
import torchvision.transforms.functional as fn
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as f
from PIL import Image
import math
from cone_finder_AI import *

SESSION = 3
IMAGE_WIDTH = 672
VGA_FOCAL_LENGTH = 367

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model("conenet_30.pt").to(device)

image_counter = 0
while True:
    image = cv2.imread(f"sessions/session_{SESSION}/images/image_{image_counter}.png")

    image_as_array_rgb = cv2.cvtColor(image[:, :, 0:3], cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(image_as_array_rgb)
    image_tensor = transforms.ToTensor()(PIL_image)

    with torch.no_grad():
        model_output = model(image_tensor.to(device))
        model_output = model_output.softmax(dim=0)

        result_mask = torch.argmax(model_output, dim=0).cpu().detach().numpy().astype(np.uint8)
        result_mask = cv2.resize(result_mask, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)

        closest_cones = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

        contours, hierarchy = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(0, len(contours)):
            if cv2.contourArea(contours[i]) > 20: # Minimum number of pixels in countour
                x, y, w, h = cv2.boundingRect(contours[i])
                x, y, w, h = x, y, w, h

                if cv2.contourArea(contours[i]) > closest_cones[result_mask[contours[i][0][0][1], contours[i][0][0][0]] - 1][0]:
                    closest_cones[result_mask[contours[i][0][0][1], contours[i][0][0][0]] - 1] = [cv2.contourArea(contours[i]), x, y, w, h]

        center_point = [((closest_cones[0][1] + closest_cones[0][3]/2) + (closest_cones[1][1] + closest_cones[1][3]/2))/2, ((closest_cones[0][2] + closest_cones[0][4]/2) + (closest_cones[1][2] + closest_cones[1][4]/2))/2]
        angle_to_center_point = math.atan2(center_point[0] - IMAGE_WIDTH/2, VGA_FOCAL_LENGTH)

        cv2.rectangle(image, (closest_cones[0][1], closest_cones[0][2]), (closest_cones[0][1] + closest_cones[0][3], closest_cones[0][2] + closest_cones[0][4]), (0, 0, 255), 2)
        cv2.rectangle(image, (closest_cones[1][1], closest_cones[1][2]), (closest_cones[1][1] + closest_cones[1][3], closest_cones[1][2] + closest_cones[1][4]), (0, 0, 255), 2)

        cv2.line(image, (int(closest_cones[0][1] + closest_cones[0][3]/2), int(closest_cones[0][2] + closest_cones[0][4]/2)), (int(closest_cones[1][1] + closest_cones[1][3]/2), int(closest_cones[1][2] + closest_cones[1][4]/2)), (0, 0, 255), 2)
        cv2.line(image, (int(IMAGE_WIDTH/2), 0), (int(IMAGE_WIDTH/2), 375), (0, 255, 0), 2)
        cv2.circle(image, (int(center_point[0]), int(center_point[1])), 2, (0, 255, 0), 3)

    cv2.imshow("drive debugger", image)
    cv2.imshow("mask", result_mask*255)
    cv2.waitKey(20)
    image_counter += 1