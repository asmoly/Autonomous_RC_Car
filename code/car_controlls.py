import RPi.GPIO as GPIO

SERVO_FREQUENCY = 50
MOTOR_FREQUENCY = 1000

# Duty Cycle
STEERING_RANGE = (5, 10)

# Duty Cycle to center wheels
STEERING_CENTER = (STEERING_RANGE[1] - STEERING_RANGE[0])/2 + STEERING_RANGE[0]

class Car_Controller:
    def __init__(self, steering_pin, motor_pin) -> None:
        GPIO.setmode(GPIO.BOARD)

        GPIO.setup(steering_pin, GPIO.OUT)
        GPIO.setup(motor_pin, GPIO.OUT)

        self.servo = GPIO.PWM(steering_pin, SERVO_FREQUENCY)
        self.motor = GPIO.PWM(motor_pin, MOTOR_FREQUENCY)

        self.servo.start(STEERING_CENTER)
        self.motor.start(0)

    # Speed is from 0 to 1
    def set_speed(self, speed):
        self.motor.ChangeDutyCycle(speed*100)

    # Steering is from -1 to 1
    def set_steering(self, steering):
        steering_value = STEERING_CENTER + ((STEERING_RANGE[1] - STEERING_RANGE[0])/2)*steering
        self.servo.ChangeDutyCycle(steering_value)