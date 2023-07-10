import os
import json
import pickle
from PIL import Image
import numpy as np

class FSOCO_Dataset:
    def __init__(self, path_to_data) -> None:
        #self.dataset = json.load(open(path_to_annotations))
        #self.annotations = self.dataset["annotations"]

        self.path_to_data = path_to_data

        for filename in os.listdir(self.path_to_data):
            print(filename)

    def get_data(self):
        pass


#dataset = FSOCO_Dataset("data")

dataset = json.load(open("data/ampera/ann/amz_00805.png.json"))

print(dataset.keys())
#print(dataset["objects"])

cone = dataset["objects"][0]
print(cone.keys())
print(cone["bitmap"]["data"])
