import os
import cv2
import numpy as np
import random
import pickle

PATH_TO_RAW_IMAGES = "raw_images"
PATH_TO_IMAGES = "images"
PATH_TO_TARGETS = "targets"
PATH_TO_DATASET_LENGTH = "image_count"

NUMBER_OF_COLORS = 2
COLOR_DICT = {1:"green", 2:"orange"}

PICTURES_PER_COLOR = 20

MAX_NUMBER_OF_CONES = 9
RANDOM_CONE_SIZE_RANGE = (0.01, 0.4)

RANDOM_BRIGHTNESS_RANGE = (100, 255)

def overlay_image(base_image, overlay_image, position):
    alpha_channel = overlay_image[:, :, 3]/255.0

    if position[1] + overlay_image.shape[0] < base_image.shape[0] and position[0] + overlay_image.shape[1] < base_image.shape[1]:
        for c in range (0, 3):
            base_image[position[1]:position[1] + overlay_image.shape[0], position[0]:position[0] + overlay_image.shape[1], c] = (alpha_channel * overlay_image[:, :, c] + (1 - alpha_channel) * base_image[position[1]:position[1] + overlay_image.shape[0], position[0]:position[0] + overlay_image.shape[1], c])

    return base_image

def change_image_brightness(image, value):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    final_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return final_image

image_counter = 12521
for session in os.listdir(PATH_TO_RAW_IMAGES):
    print(f"Creating dataset for {session}")
    for raw_image_name in os.listdir(f"{PATH_TO_RAW_IMAGES}/{session}"):
        raw_image = cv2.imread(f"{PATH_TO_RAW_IMAGES}/{session}/{raw_image_name}")
        image_mask = np.zeros(raw_image.shape[0:2])

        number_of_cones = random.randint(0, MAX_NUMBER_OF_CONES + 1)
        for i in range (number_of_cones):
            random_color = random.randrange(1, NUMBER_OF_COLORS + 1)
            random_color = COLOR_DICT[random_color]
            random_picture = random.randrange(1, PICTURES_PER_COLOR + 1)

            random_size = random.uniform(RANDOM_CONE_SIZE_RANGE[0], RANDOM_CONE_SIZE_RANGE[1])

            cone_image = cv2.imread(f"cones/{random_color}/{random_picture}.png", cv2.IMREAD_UNCHANGED)
            cone_image = cv2.resize(cone_image, (0,0), fx=random_size, fy=random_size)

            #random_brightness = random.randint(RANDOM_BRIGHTNESS_RANGE[0], RANDOM_BRIGHTNESS_RANGE[1])
            #cone_image = change_image_brightness(cone_image, random_brightness)

            cone_mask = cv2.imread(f"cone_masks/{random_color}/{random_picture}.png", cv2.IMREAD_GRAYSCALE)
            cone_mask = cv2.resize(cone_mask, (0,0), fx=random_size, fy=random_size, interpolation=cv2.INTER_NEAREST) 

            random_position = (random.randrange(0, raw_image.shape[1]), random.randrange(0, raw_image.shape[0]))
            overlay_image(raw_image, cone_image, random_position)

            if random_position[1] + cone_mask.shape[0] < image_mask.shape[0] and random_position[0] + cone_mask.shape[1] < image_mask.shape[1]:
                image_mask[random_position[1]:random_position[1] + cone_mask.shape[0], random_position[0]:random_position[0] + cone_mask.shape[1]] = np.where(cone_mask!=0, cone_mask, image_mask[random_position[1]:random_position[1] + cone_mask.shape[0], random_position[0]:random_position[0] + cone_mask.shape[1]])
                
        cv2.imwrite(f"{PATH_TO_IMAGES}/image_{image_counter}.png", raw_image)
        cv2.imwrite(f"{PATH_TO_TARGETS}/target_{image_counter}.png", image_mask)
        image_counter += 1

print(f"Created {image_counter} images")
pickle.dump(image_counter, open(PATH_TO_DATASET_LENGTH, "wb"))