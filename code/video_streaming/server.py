import sys
import cv2
import numpy as np
import pyzed.sl as sl
import keyboard
from flask import Flask, render_template, Response

cam = sl.Camera()
app = Flask(__name__)

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.VGA
init_params.camera_fps = 60

init_params.sdk_verbose = True # Enable verbose logging
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

def generate_frames():
    while keyboard.is_pressed("a") == False:
        if cam.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
            # Normal view
            cam.retrieve_image(image, sl.VIEW.LEFT)
            image_as_array = image.get_data()

            # Depth
            cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
            depth_as_array = depth.get_data()

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