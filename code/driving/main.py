import sys
import cv2
import numpy as np
import pyzed.sl as sl
import torchvision.transforms.functional as fn
import torch
from flask import Flask, render_template, Response
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as f
from PIL import Image
import math

import PI_Controller
import Car_Controller
from cone_finder_AI import load_model, clamp

SESSION = 3
IMAGE_WIDTH = 672
IMAGE_HEIGHT = 376
VGA_FOCAL_LENGTH = 367

app = Flask(__name__)

car_controller = Car_Controller.Car_Controller()
pi_controller = PI_Controller.PI(0.1, 0.2)

cam = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.VGA
init_params.camera_fps = 60

init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Set the depth mode to performance (fastest)
init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units

err = cam.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Camera failed to open")
    sys.exit()

image = sl.Mat()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model("conenet_30.pt").to(device)

def generate_frames():
    while True:
        if cam.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(image, sl.VIEW.LEFT)
        
        image_as_array = image.get_data()

        image_as_array_rgb = cv2.cvtColor(image_as_array[:, :, 0:3], cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(image_as_array_rgb)
        image_tensor = transforms.ToTensor()(PIL_image)

        with torch.no_grad():
            model_output = model(image_tensor.to(device))
            model_output = model_output.softmax(dim=0)

            result_mask = torch.argmax(model_output, dim=0).cpu().detach().numpy().astype(np.uint8)
            result_mask = cv2.resize(result_mask, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)

            closest_cones = [[0, 0, int(IMAGE_HEIGHT/2), 0, 0], [0, IMAGE_WIDTH - 1, int(IMAGE_HEIGHT/2), 0, 0]] # green, orange

            contours, hierarchy = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(0, len(contours)):
                if cv2.contourArea(contours[i]) > 300: # Minimum number of pixels in countour
                    x, y, w, h = cv2.boundingRect(contours[i])
                    x, y, w, h = x, y, w, h

                    if cv2.contourArea(contours[i]) > closest_cones[result_mask[contours[i][0][0][1], contours[i][0][0][0]] - 1][0] and cv2.contourArea(contours[i]) < 10000:
                        closest_cones[result_mask[contours[i][0][0][1], contours[i][0][0][0]] - 1] = [cv2.contourArea(contours[i]), x, y, w, h]

            center_point = [((closest_cones[0][1] + closest_cones[0][3]/2) + (closest_cones[1][1] + closest_cones[1][3]/2))/2, ((closest_cones[0][2] + closest_cones[0][4]/2) + (closest_cones[1][2] + closest_cones[1][4]/2))/2]
            angle_to_center_point = math.atan2(center_point[0] - IMAGE_WIDTH/2, VGA_FOCAL_LENGTH)

            steering = pi_controller.calculate_error(angle_to_center_point)
            steering = clamp(steering, -1.0, 1.0)*-1.0
            #print(steering)
            car_controller.set_speed_and_steering(1, steering, translate_values=True)

            cv2.rectangle(image_as_array, (closest_cones[0][1], closest_cones[0][2]), (closest_cones[0][1] + closest_cones[0][3], closest_cones[0][2] + closest_cones[0][4]), (0, 0, 255), 2)
            cv2.rectangle(image_as_array, (closest_cones[1][1], closest_cones[1][2]), (closest_cones[1][1] + closest_cones[1][3], closest_cones[1][2] + closest_cones[1][4]), (0, 0, 255), 2)

            cv2.line(image_as_array, (int(closest_cones[0][1] + closest_cones[0][3]/2), int(closest_cones[0][2] + closest_cones[0][4]/2)), (int(closest_cones[1][1] + closest_cones[1][3]/2), int(closest_cones[1][2] + closest_cones[1][4]/2)), (0, 0, 255), 2)
            cv2.line(image_as_array, (int(IMAGE_WIDTH/2), 0), (int(IMAGE_WIDTH/2), 375), (0, 255, 0), 2)
            cv2.circle(image_as_array, (int(center_point[0]), int(center_point[1])), 2, (0, 255, 0), 3)

        _, buffer = cv2.imencode('.jpg', image_as_array)
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
