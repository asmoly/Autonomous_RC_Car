import sys
import pyzed.sl as sl
import socket
import pickle
import keyboard
import cv2
import numpy

UDP_IP = "192.168.1.4"
UDP_PORT = 5006

X_PIECES = 12
Y_PIECES = 12

def split_image(x_pieces, y_pieces, image_array):
    pieces = []

    x_size = int(image_array.shape[1]/x_pieces)
    y_size = int(image_array.shape[0]/y_pieces)

    for y in range (0, y_pieces):
        for x in range (0, x_pieces):
            pieces.append(image_array[y_size*y:y_size*y + y_size, x_size*x:x_size*x + x_size, :])

    return pieces

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera failed to open")
        sys.exit()

    print("Opened Camera")

    image = sl.Mat()

    while keyboard.is_pressed("a") == False:
        if zed.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)

            image_as_array = image.get_data()
            image_pieces = split_image(X_PIECES, Y_PIECES, image_as_array)
            
            for i in range (0, len(image_pieces)):
                piece = [image_pieces[i], i]
                piece_as_bytes = pickle.dumps(piece)
                sock.sendto(piece_as_bytes, (UDP_IP, UDP_PORT))

            print("Sent image")

            #cv2.imshow("Video", image_as_array)
            #cv2.waitKey(1)
        else:
            print("Failed to grab image")


    zed.close()
    print("Closed Camera")

if __name__ == "__main__":
    main()