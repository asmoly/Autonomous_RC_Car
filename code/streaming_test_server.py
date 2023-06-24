import sys
import pyzed.sl as sl
import socket
import pickle
import keyboard
import cv2
import numpy

UDP_IP = "192.168.1.4"
UDP_PORT = 5006

MAX_SIZE = 65000

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

            compressed_image = cv2.imencode(".jpg", image_as_array)[1]
            compressed_image_bytes = compressed_image.tobytes()
            number_of_packs = int(len(compressed_image_bytes)/MAX_SIZE) + 1

            image_info = {"packs":number_of_packs, "height":image.get_height(), "width":image.get_width()}
            image_info_as_bytes = pickle.dumps(image_info)
            sock.sendto(image_info_as_bytes, (UDP_IP, UDP_PORT))

            counter = 0
            for i in range (0, number_of_packs):
                data_to_send = compressed_image_bytes[counter:counter + MAX_SIZE]
                counter += MAX_SIZE

                sock.sendto(data_to_send, (UDP_IP, UDP_PORT))

            print(f"Sent image with {number_of_packs} packs")
        else:
            print("Failed to grab image")


    zed.close()
    print("Closed Camera")

if __name__ == "__main__":
    main()