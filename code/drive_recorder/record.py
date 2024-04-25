import os
import sys
import cv2
import numpy as np
import pyzed.sl as sl
import keyboard
from flask import Flask, render_template, Response
import pickle
import socket
import threading
import Car_Controller

SESSION_NUMBER = pickle.load(open("session_count", "rb"))
print(f"Session Number: {SESSION_NUMBER}")

os.makedirs(f"sessions/session_{SESSION_NUMBER}")
os.makedirs(f"sessions/session_{SESSION_NUMBER}/images")
os.makedirs(f"sessions/session_{SESSION_NUMBER}/depth_images")

pickle.dump(SESSION_NUMBER + 1, open("session_count", "wb"))

cam = sl.Camera()
app = Flask(__name__)

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.VGA
init_params.camera_fps = 60

#init_params.sdk_verbose = True # Enable verbose logging
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Set the depth mode to performance (fastest)
init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units

err = cam.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Camera failed to open")
    sys.exit()

print("Opened Camera")

image = sl.Mat()
depth = sl.Mat()
normalized_depth = sl.Mat()

def car_controls():
    PORT = 6000

    car_controller = Car_Controller.Car_Controller()

    reciever_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    reciever_socket.setblocking(False)
    reciever_socket.bind(("", PORT))
    missing_data = 0

    while missing_data < 100000:
        try:
            data, address = reciever_socket.recvfrom(1024)
            data = data.decode("utf-8")
            data = data.split(",")

            car_controller.set_speed_and_steering(int(data[0]), int(data[1]))

            missing_data = 0
        except:
            pass
            #missing_data += 1

def generate_frames():
    threading.Thread(target=car_controls).start()
    
    image_counter = 0
    while True:
        if cam.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
            # Normal view
            cam.retrieve_image(image, sl.VIEW.LEFT)
            image_as_array = image.get_data()

            # Depth
            cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
            depth_as_array = depth.get_data()

            cv2.imwrite(f"sessions/session_{SESSION_NUMBER}/images/image_{image_counter}.png", image_as_array)
            cv2.imwrite(f"sessions/session_{SESSION_NUMBER}/depth_images/depth_image_{image_counter}.png", depth_as_array)
            image_counter += 1

            # Normalized Depth for viewing
            cam.retrieve_image(normalized_depth, sl.VIEW.DEPTH)
            normalized_depth_as_array = normalized_depth.get_data()

            width = image_as_array.shape[1]
            height = image_as_array.shape[0]
            image_to_stream = np.zeros((height, width*2, 4))

            # Top Left
            image_to_stream[0:height, 0:width, :] = image_as_array
            # Top Right
            image_to_stream[0:height, width:, :] = normalized_depth_as_array

            _, buffer = cv2.imencode('.jpg', image_to_stream)
            frame_bytes = buffer.tobytes()
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)