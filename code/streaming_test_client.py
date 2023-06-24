import cv2
import socket
import pickle

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while True:
    image_as_bytes, addr = sock.recvfrom(1024)
    image_as_array = pickle.loads(image_as_bytes)

    cv2.imshow("Stream", image_as_array)
    cv2.waitKey(1)