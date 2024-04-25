#include <Servo.h>

Servo ESC;
Servo steering_servo;

int start_motor_value = 50;
int min_motor_value = 90;
int max_motor_value = 110;

int motor_value;
float accelerator = 0.0;
int steering = 0;

int no_signal_count = 0;
int no_signal_threshhold = 10000;

void set_target_speed(float target_speed)
{
  int target_motor_value = int(min_motor_value + (max_motor_value - min_motor_value)*target_speed);
  
  while (motor_value != target_motor_value)
  {
     if (motor_value < target_motor_value)
     {
        motor_value++;
     }
     else if (motor_value > target_motor_value)
     {
        motor_value--;
     } 

     ESC.write(motor_value);
  }
}

void setup() 
{
  pinMode(7, OUTPUT);
  digitalWrite(7, HIGH);
  steering_servo.attach(10);
  
  motor_value = start_motor_value;
  
  Serial.begin(115200);
  while (!Serial.available())
  {
    // wait for signal
  }
  
  ESC.attach(9, 1000, 2000);
  set_target_speed(0);
  delay(100);
}

void loop() 
{
   if (Serial.available())
   {
      String input = Serial.readStringUntil('\n'); // input has acceleration from 0-100 and steering from 0-100 with 50 being the center
      accelerator = float(input.substring(0, input.indexOf(',')).toInt()/100.0);
      steering = input.substring(input.indexOf(',') + 1).toInt() - 50;

      no_signal_count = 0;
   }
   else
   {
      no_signal_count++;
   }

   if (no_signal_count >= no_signal_threshhold)
   {
      accelerator = 0.0;
   }

   if (accelerator > 1.0)
   {
      accelerator = 1.0;
   }
   else if (accelerator < 0.0)
   {
      accelerator = 0.0;
   }

   set_target_speed(accelerator);
   steering_servo.write(90 + steering);
}
