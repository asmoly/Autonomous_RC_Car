import pyzed.sl as sl
import cv2
import keyboard

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30

    init_params.sdk_verbose = True # Enable verbose logging
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Set the depth mode to performance (fastest)
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    print("Opened Camera")

    image = sl.Mat()
    depth = sl.Mat()
    depth_image = sl.Mat()

    while keyboard.is_pressed("a") == False:
        if zed.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)

            image_as_array = image.get_data()
            image_as_array = cv2.putText(image_as_array, f"{image.get_width()}x{image.get_height()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            depth_as_array = depth.get_data()

            zed.retrieve_image(depth_image, sl.VIEW.DEPTH)
            depth_image_as_array = depth_image.get_data()

            cv2.imshow("Video", depth_image_as_array)

            cv2.waitKey(1)

    zed.close()
    print("Closed Camera")

if __name__ == "__main__":
    main()