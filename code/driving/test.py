from Car_Controller import *

controller = Car_Controller()
while True:
    controller.set_speed_and_steering(1, 0, translate_values=True)