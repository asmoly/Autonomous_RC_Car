import Gamepad
import Car_Controller

gamepad = Gamepad.Gamepad()
car_controller = Car_Controller.Car_Controller()

while True:
    car_controller.set_speed_and_steering(gamepad.right_trigger, gamepad.left_joystick[0])