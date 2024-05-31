import os
import cv2
import numpy as np

os.makedirs("cone_masks")

for color in os.listdir("cones"):
    os.makedirs(f"cone_masks/{color}")

    for cone_picture in os.listdir(f"cones/{color}"):
        image = cv2.imread(f"cones/{color}/{cone_picture}", cv2.IMREAD_UNCHANGED)
        image_mask = np.zeros(image.shape[0:2])

        for y in range (0, image.shape[0]):
            for x in range (0, image.shape[1]):
                if image[y, x][3] == 255:
                    if color == "orange":
                        image_mask[y, x] = 200
                    elif color == "green":
                        image_mask[y, x] = 100

        cv2.imwrite(f"cone_masks/{color}/{cone_picture}", image_mask)