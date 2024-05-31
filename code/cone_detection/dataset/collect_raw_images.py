import sys
import pyzed.sl as sl
import cv2
import keyboard
import os

def main():
    PATH_TO_IMAGES = "raw_images"
    SESSION = 5
    os.makedirs(f"{PATH_TO_IMAGES}/session_{SESSION}")

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.camera_fps = 60

    init_params.sdk_verbose = True # Enable verbose logging
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Set the depth mode to performance (fastest)
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera failed to open")
        sys.exit()

    print("Opened Camera")

    image = sl.Mat()
    depth = sl.Mat()
    normalized_depth = sl.Mat()

    image_counter = 0
    while keyboard.is_pressed("a") == False:
        if zed.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            image_as_array = image.get_data()

            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            depth_as_array = depth.get_data()

            zed.retrieve_image(normalized_depth, sl.VIEW.DEPTH)
            normalized_depth_as_array = normalized_depth.get_data()

            #cv2.imshow("Depth", normalized_depth_as_array)
            cv2.imshow("Video", image_as_array)
            cv2.imwrite(f"{PATH_TO_IMAGES}/session_{SESSION}/raw_image_{image_counter}.png", image_as_array)
            cv2.waitKey(100)

            image_counter += 1

    zed.close()
    print("Closed Camera")

if __name__ == "__main__":
    main()