import keyboard

from Car_Controller import Car_Controller
from Gamepad import Gamepad

car_controller = Car_Controller(33, 32)
gamepad = Gamepad()

while gamepad.a == 0:
    car_controller.set_speed(gamepad.right_trigger)
    car_controller.set_steering(gamepad.left_joystick[0])