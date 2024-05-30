# Autonomous_RC_Car
This is an autonomous car which I built and programed that can drive around a track made of cones. It uses a CNN for perception and a PID controller for the car's controls.

## The platform
These are the components and steps I used to build the platform:
* The base I used for the RC car was the [traxass 4-Tec 2.0 VXL AWD chassis](https://traxxas.com/products/models/electric/4-tec-2-vxl-chassis)
* First I removed the reciever and speed controller
* I replaced the motor with a rock crawler motor to achieve lower speeds [motor](fefe)
* I also replaced the servo with one of the same size and specifications but it shouldn't matter
* The battery I use to power the entire car is a [11.1v 5200mah 80c lipo battery](https://www.amazon.com/HOOVO-Battery-Softcase-Airplane-Helicopter/dp/B08YR15KQ8/ref=sr_1_26?crid=3T1CQWZ9E0ZQR&dib=eyJ2IjoiMSJ9.6ZjyZbfEiWJ7tdVpfAVvuvjt8XudmCtCPzSYpTleNc05mh2yo6mfVAMZlAF3hvNmckC2ep7nfSi4SJ74hudXTMdZZkmM9k-76faLNpIbUFBnAWACijy51pD85f4hKgeTCn34ava9G3eAQqr2qH7FEA7Cg1qoIk1BzBP1DYS2NZkXySz4zo3nr5_EIHcrQxVgAoznscEDhCsp7-KeBFDmaeiJQlVJl3_DbH6vAjIuQZyXx4YL04KSaSCYucGg5N-02sER1Vz146WEheqc_nmL9LXG3RgVCxjw9MLpna-Cvyk.6MwmgP6LzbD1DifoUhehMecCohP2Bj-yqL6HqxLd3QU&dib_tag=se&keywords=hoovo+battery+5200+mah+80c&qid=1717099192&sprefix=hoovo+battery+5200+mah+80%2Caps%2C218&sr=8-26)
* The battery should be soldered straight to the motor power and also to a switch which will provide power to the computer
* After I created a platform that sits on top of the RC car using a large pcb board
* Towards the back of the platform I put a [Jetson Orin Nano](https://www.amazon.com/NVIDIA-Jetson-Orin-Nano-Developer/dp/B0BZJTQ5YP/ref=sr_1_3?crid=10A6DHVZOMHYS&dib=eyJ2IjoiMSJ9.EY0iLDd0M9dkGkWsLUJY8GyM5_RC274wWAuZvuHJK3awsYoCOS7ApMqgCFxnREWCAg487swCnHqQlOT3l6tR1k6h4vm1iKd-lu0q7GZf3BnHLHl2MwSn6SY8o_craQPrrWwd4YmGHAbw27qIeQZoBN3mC43_aI9_udkkIgCGUnczk9KcKDmB5AG7x9ctKI_02A6jp643ird4vcV8CU88zz55-YsViHnQ7JmzxYWkXCk.ZX_zvPC0fUd2RQG6fz2tD6Ggt27hwGcSPr760RlVMLA&dib_tag=se&keywords=jetson+orin+nano&qid=1717099447&sprefix=jetson+orin+nano%2Caps%2C177&sr=8-3) this is the on board computer used for all the proccessing. I also soldered a connection from the switch to the barrel jack of the jetson.
* I also put an Arduino Uno on the platform. All the wire connections will be specified later.
* I created another small raised platform that is put on top of the main platform. I then glued a [ZED Mini Stereo Camera](https://store.stereolabs.com/products/zed-mini) ont he small platform.
* Both the camera and the Arduino should be connected to the Jetson via USB
### Wiring
* Both the servo and motor should be connected to 5v and ground on the Arduino, the third wire on the motor should be connected to pin 9, and the third pin on the servo should be connected to pin 10.
* Finally for the arduino to work upload the file 'code/controller/arduino_ESC_servo_controller/arduino_ESC_servo_controller.ino' to the arduino.
