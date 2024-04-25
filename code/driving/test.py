from PI_Controller import *

pid = PI(0.001, 0.001)

target = 45
current_value = 0

for i in range (0, 1000):
    current_value += pid.calculate_error(target - current_value)
    print(current_value)