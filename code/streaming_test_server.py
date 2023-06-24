import pyzed.sl as sl
import socket
import pickle
import keyboard

UDP_IP = "10.42.0.120"
UDP_PORT = 5005

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    print("Opened Camera")

    image = sl.Mat()

    while keyboard.is_pressed("a") == False:
        if zed.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)

            image_as_array = image.get_data()
            image_as_bytes = pickle.dumps(image_as_array)

            sock.sendto(image_as_bytes, (UDP_IP, UDP_PORT))

    zed.close()
    print("Closed Camera")

if __name__ == "__main__":
    main()