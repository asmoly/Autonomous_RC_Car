import cv2
from keyboard import is_pressed
from vidgear.gears import NetGear

client = NetGear(receive_mode = True)

while True:
    frame = client.recv()

    if frame is None:
        break

    cv2.imshow("Video Stream", frame)

    if is_pressed("c"):
        break

cv2.destroyAllWindows()
client.close()