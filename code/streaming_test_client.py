import cv2
import socket
import pickle
import numpy as np

X_PIECES = 12
Y_PIECES = 12

UDP_IP = "0.0.0.0"
UDP_PORT = 5006

pieces = []
for i in range (0, X_PIECES*Y_PIECES):
    pieces.append(0)

def rebuild_image(x_pieces, y_pieces, pieces):
    x_size = pieces[0].shape[1]
    y_size = pieces[0].shape[0]
    
    image_shape = [pieces[0].shape[0]*y_pieces, pieces[0].shape[1]*x_pieces]
    image_array = np.zeros(shape=(image_shape[0], image_shape[1], 4))

    counter = 0
    for y in range (0, y_pieces):
        for x in range (0, x_pieces):
            image_array[y*y_size:y*y_size + y_size, x*x_size:x*x_size + x_size, :] = pieces[counter]
            counter += 1

    return image_array

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while True:
    piece_as_bytes, addr = sock.recvfrom(10000)
    piece = pickle.loads(piece_as_bytes)

    pieces[piece[1]] = piece[0]

    image_as_array = rebuild_image(X_PIECES, Y_PIECES, pieces)
    
    cv2.imshow("Stream", image_as_array)
    cv2.waitKey(1)