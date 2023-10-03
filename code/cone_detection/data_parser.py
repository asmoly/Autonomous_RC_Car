import numpy as np
import cv2
import base64
import json

file = open("data/amz/ann/amz_00016.jpg.json")
data = json.load(file)
file.close()

# objects contains cones
# objects is list of all cones
# each object is a dictionary
# bitmap is dictionary - origin is position - data is a string

print(data["objects"][0]["bitmap"]["data"])
image_string = data["objects"][0]["bitmap"]["data"]
