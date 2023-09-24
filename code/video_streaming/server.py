import cv2
from keyboard import is_pressed
from vidgear.gears import VideoGear, NetGear

server = NetGear()

camera_video = cv2.VideoCapture(0)

while True:
    ret, frame = camera_video.read()

    server.send(frame)

    cv2.imshow("Video Feed", frame)
    cv2.waitKey(1)

    if is_pressed("c"):
        break

cv2.destroyAllWindows()
server.close()