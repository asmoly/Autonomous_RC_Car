import sys
import pyzed.sl as sl
import cv2
import keyboard
import numpy as np

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.camera_fps = 60

    #init_params.sdk_verbose = True # Enable verbose logging
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

    while keyboard.is_pressed("a") == False:
        if zed.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)

            image_as_array = image.get_data()

            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            depth_as_array = depth.get_data()
            #print(depth_as_array[300, 300])
            print(np.nanmax(depth_as_array))

            zed.retrieve_image(normalized_depth, sl.VIEW.DEPTH)
            normalized_depth_as_array = normalized_depth.get_data()
            #print(np.max(normalized_depth_as_array))

            cv2.imshow("Video", normalized_depth_as_array)

            cv2.waitKey(1)

    zed.close()
    print("Closed Camera")

if __name__ == "__main__":
    main()