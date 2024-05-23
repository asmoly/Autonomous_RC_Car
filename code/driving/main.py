import sys
import time
import math
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as fn
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as f
import pyzed.sl as sl
from PIL import Image
from flask import Flask, render_template, Response

import PI_Controller
import Car_Controller
from cone_finder_AI import load_model, clamp

SESSION = 3
IMAGE_WIDTH = 672
IMAGE_HEIGHT = 376
CAMERA_VGA_FOCAL_LENGTH = 367

CONE_HEIGHT = 0.175 # meters
NON_VISIBLE_CONE_OFFSET = 0.5 # meters
MAX_ANGLE_TO_CONE = 45

CONE_MIN_BLOB_AREA = 300 # in pixels
CONE_MAX_HEIGHT_PERCENT = 1.0 # in percents
LEFT_MARGIN_POINT = (0, 0)
RIGHT_MARGIN_POINT = (IMAGE_WIDTH - 1, 0)

STEERING_PROPORTIONAL_GAIN = 0.9   # PID's P term gain
STEERING_INTEGRAL_GAIN = 0.1   # PID's I term gain
STEERING_TRIM = 0.0 # from -1 - 1

SPEED_PROPORTIONAL_GAIN = 0.2
SPEED_INTEGRAL_GAIN = 0.05

SPEED_MIN = 0.65
SPEED_MAX = 0.8
SPEED_MULTIPLYER = 2

streaming_app = Flask(__name__)

car_controller = Car_Controller.Car_Controller()
steering_pi_controller = PI_Controller.PI(STEERING_PROPORTIONAL_GAIN, STEERING_INTEGRAL_GAIN, buffer_length=10)
speed_pi_controller = PI_Controller.PI(SPEED_PROPORTIONAL_GAIN, SPEED_INTEGRAL_GAIN, buffer_length=15)

camera = sl.Camera()

camera_params = sl.InitParameters()
camera_params.camera_resolution = sl.RESOLUTION.VGA
camera_params.camera_fps = 60
camera_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Set the depth mode to performance (fastest)
camera_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units

err = camera.open(camera_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("ERROR: Camera failed to open!")
    sys.exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model("conenet_30.pt").to(device)

frame = sl.Mat()

def angle_to_speed(angle):
    proccessed_angle = -angle + 1
    return proccessed_angle

def process_frames():
    prev_time = time.time()
    while True:
        current_time = time.time()
        # print(int((current_time - prev_time)*1000))
        prev_time = current_time

        if camera.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
            camera.retrieve_image(frame, sl.VIEW.LEFT)
        
        frame_as_array = frame.get_data()
        frame_as_array_rgb = cv2.cvtColor(frame_as_array[:, :, 0:3], cv2.COLOR_BGR2RGB)
        PIL_image = Image.fromarray(frame_as_array_rgb)
        image_tensor = transforms.ToTensor()(PIL_image)

        with torch.no_grad():
            model_output = model(image_tensor.to(device))
            model_output = model_output.softmax(dim=0)
            result_mask = torch.argmax(model_output, dim=0).cpu().detach().numpy().astype(np.uint8)
            result_mask = cv2.resize(result_mask, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)   # upscale 2x

            #closest_cones = [[0, LEFT_MARGIN_POINT[0], LEFT_MARGIN_POINT[1], 0, 0], [0, RIGHT_MARGIN_POINT[0], RIGHT_MARGIN_POINT[1], 0, 0]] # green, orange
            closest_cones = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]


            contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(0, len(contours)):
                blob_area = cv2.contourArea(contours[i])
                if blob_area > CONE_MIN_BLOB_AREA:   # TODO: this is in pixels! Must be changed to percent of total frame pixels
                    x, y, w, h = cv2.boundingRect(contours[i])

                    blob_class_id = result_mask[contours[i][0][0][1], contours[i][0][0][0]]  # 0 - background, 1 - green, 2 - red
                    if blob_area > closest_cones[blob_class_id - 1][0] and h <= CONE_MAX_HEIGHT_PERCENT*IMAGE_HEIGHT:
                        closest_cones[blob_class_id - 1] = [blob_area, x, y, w, h]

            # Compute center point between nearest cones to drive to and then the heading angle to that point
            left_cone_center = ((closest_cones[0][1] + closest_cones[0][3]/2), (closest_cones[0][2] + closest_cones[0][4]/2))
            right_cone_center = ((closest_cones[1][1] + closest_cones[1][3]/2), (closest_cones[1][2] + closest_cones[1][4]/2))

            center_point = [0, 0]
            angle_to_center_point = 0

            if closest_cones[0] == [0, 0, 0, 0, 0] and closest_cones[1] == [0, 0, 0, 0, 0]: # no cones visible
                center_point = [IMAGE_WIDTH/2, IMAGE_HEIGHT/2]
            elif closest_cones[0] == [0, 0, 0, 0, 0]: # green cone not visible
                height_of_visible_cone = closest_cones[1][4] # pixels
                distance_to_visible_cone = (CONE_HEIGHT/height_of_visible_cone)*CAMERA_VGA_FOCAL_LENGTH # meters
                cone_offset_pixels = (CAMERA_VGA_FOCAL_LENGTH/distance_to_visible_cone)*NON_VISIBLE_CONE_OFFSET
                center_point = [right_cone_center[0] - cone_offset_pixels, right_cone_center[1]]
                angle_to_center_point = math.atan2(center_point[0] - IMAGE_WIDTH/2, CAMERA_VGA_FOCAL_LENGTH)
            elif closest_cones[1] == [0, 0, 0, 0, 0]: # orange cone not visible
                height_of_visible_cone = closest_cones[0][4] # pixels
                distance_to_visible_cone = (CONE_HEIGHT/height_of_visible_cone)*CAMERA_VGA_FOCAL_LENGTH # meters
                cone_offset_pixels = (CAMERA_VGA_FOCAL_LENGTH/distance_to_visible_cone)*NON_VISIBLE_CONE_OFFSET
                center_point = [left_cone_center[0] + cone_offset_pixels, left_cone_center[1]]
                angle_to_center_point = math.atan2(center_point[0] - IMAGE_WIDTH/2, CAMERA_VGA_FOCAL_LENGTH)
            else: # both cones visible
                angle_to_left_cone = math.atan2(left_cone_center[0] - IMAGE_WIDTH/2, CAMERA_VGA_FOCAL_LENGTH)
                angle_to_right_cone = math.atan2(right_cone_center[0] - IMAGE_WIDTH/2, CAMERA_VGA_FOCAL_LENGTH)
                angle_to_center_point = angle_to_left_cone + (angle_to_right_cone - angle_to_left_cone)/2
                center_point = [(left_cone_center[0] + right_cone_center[0])/2, (left_cone_center[1] + right_cone_center[1])/2]
                cv2.line(frame_as_array, (int(left_cone_center[0]), int(left_cone_center[1])), (int(right_cone_center[0]), int(right_cone_center[1])), (0, 0, 255), 2)

            #angle_to_center_point = math.atan2(center_point[0] - IMAGE_WIDTH/2, CAMERA_VGA_FOCAL_LENGTH)

            # Calculate steering and send it to steering servo
            steering = steering_pi_controller.compute_control(angle_to_center_point)
            steering = clamp(steering + STEERING_TRIM, -1.0 + STEERING_TRIM, 1.0 + STEERING_TRIM)*-1.0
            
            speed = speed_pi_controller.compute_control(angle_to_speed(abs(angle_to_center_point/(MAX_ANGLE_TO_CONE*math.pi/180.0))))
            speed = clamp(speed, SPEED_MIN, SPEED_MAX)/SPEED_MULTIPLYER

            car_controller.set_speed_and_steering(speed, steering, translate_values=True)

            cv2.rectangle(frame_as_array, (closest_cones[0][1], closest_cones[0][2]), (closest_cones[0][1] + closest_cones[0][3], closest_cones[0][2] + closest_cones[0][4]), (0, 0, 255), 2)
            cv2.rectangle(frame_as_array, (closest_cones[1][1], closest_cones[1][2]), (closest_cones[1][1] + closest_cones[1][3], closest_cones[1][2] + closest_cones[1][4]), (0, 0, 255), 2)
            cv2.line(frame_as_array, (int(IMAGE_WIDTH/2), 0), (int(IMAGE_WIDTH/2), 375), (0, 255, 0), 2)
            cv2.circle(frame_as_array, (int(center_point[0]), int(center_point[1])), 2, (0, 255, 0), 3)

        _, buffer = cv2.imencode('.jpg', frame_as_array)
        frame_bytes = buffer.tobytes()
        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
@streaming_app.route('/')
def index():
    return render_template("index.html")

@streaming_app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    streaming_app.run(host='0.0.0.0', port=5001)
