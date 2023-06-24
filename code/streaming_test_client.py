import cv2
import socket
import pickle
import numpy as np

MAX_SIZE = 65000

UDP_IP = "0.0.0.0"
UDP_PORT = 5006

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

buffer = 0

while True:
    data, addr = sock.recvfrom(MAX_SIZE)
    
    image_info = pickle.loads(data)
    number_of_packs = image_info["packs"]
    image_dimensions = [image_info["width"], image_info["height"]]

    for i in range (0, number_of_packs):
        data, addr = sock.recvfrom(MAX_SIZE)

        if i == 0:
            buffer = data
        else:
            buffer += data

    compressed_image = np.frombuffer(buffer, dtype=np.uint8)
    compressed_image = np.reshape(image.shape[0], 1)

    image = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)

    cv2.imshow("Stream", image)
    cv2.waitKey(1)