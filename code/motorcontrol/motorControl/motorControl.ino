// Author: James Crossley, 21480395
/* Using adapted code from Justin's code for his initial motor control. Added a scaled functionality to allow for a range of speeds to be sent to the motor.
   This is done by sending a string with the first character being 'S' and the second being the speed. The speed is then mapped to a range of 0-180,
   which is then mapped to a pulse width of 1000-2000 microseconds. The ESC is then set to this pulse width.
*/

#include <Servo.h>

Servo esc;
int throttlePin = 9;

int mappedSpeed;
int motorSpeed;
int throttleSpeed;

String lastCommand = "F";

int currentPulseWidth = 1500;
int targetPulseWidth = 1500;

void setup() {
  esc.attach(throttlePin);
  initializeESC();
  Serial.begin(115200);
  adjustThrottle(1500);
}

void adjustThrottle(int newTarget) {
  targetPulseWidth = newTarget;
  while (currentPulseWidth != targetPulseWidth) {
    if (currentPulseWidth < targetPulseWidth) {
      currentPulseWidth++;
    } else {
      currentPulseWidth--;
    }
    esc.writeMicroseconds(currentPulseWidth);
    delay(10);
  }
}

void loop() {
  lastCommand = "Nothing";
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    if (command.length() > 0) {
      lastCommand = command;
    }
    else
    {
      lastCommand = "No command";
    }
  }

  //Serial.print("Median Distance: ");
  //Serial.println(distance);

  if (lastCommand[0] == 'S'){ // Speed command received. A non speed command will be ignored with CAV speed preserved.
    if (lastCommand.length() > 1) {
      int motorSpeed = lastCommand.substring(1).toInt();
      //Serial.println("Motor speed: " + String(motorSpeed));
      if (motorSpeed >= -10000 && motorSpeed <= 10000) {
        mappedSpeed = map(motorSpeed, -10000, 10000, 0, 180);
        //Serial.println("Mapped speed: " + String(mappedSpeed));
        throttleSpeed = map(mappedSpeed, 0, 180, 1000, 2000);
        //Serial.println("Throttle Speed: " + String(throttleSpeed));
        adjustThrottle(throttleSpeed);
      }
    }
    /*else {
      adjustThrottle(1500); // Stop command
      /* TODO: Possibly have a message sent back to the client to confirm an error. Should be handled in the client code tho.
      This also is entered when no command has been received too.
    }*/
  }
}

void initializeESC() {
  //Serial.println("Initializing ESC.");
  esc.writeMicroseconds(1500);
  delay(7000);
  //Serial.println("ESC is ready!");
}

void sort(unsigned int arr[], int n) {
  for (int i = 0; i < n-1; i++) {
    for (int j = 0; j < n-i-1; j++) {
      if (arr[j] > arr[j+1]) {
        unsigned int temp = arr[j];
        arr[j] = arr[j+1];
        arr[j+1] = temp;
      }
    }
  }
}