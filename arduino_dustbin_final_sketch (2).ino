#include <Servo.h>

Servo doorServo;
Servo moistureServo;

int doorServoPin = 9;   // Pin for automatic door servo
int moistureServoPin = 10; // Pin for moisture sensor servo

int moisturePin = A0; // Moisture sensor pin
int moistureThreshold = 500; // Moisture threshold for waste segregation

const int trigPin = 11;    // Trigger pin of ultrasonic sensor
const int echoPin = 12;    // Echo pin of ultrasonic sensor

const int thresholdDistance = 15;  // Threshold distance for triggering the door

void setup() {
  doorServo.attach(doorServoPin);
  moistureServo.attach(moistureServoPin);
  
  pinMode(moisturePin, INPUT);
  
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  Serial.begin(9600);
}

void loop() {
  int moistureValue = analogRead(moisturePin);
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  long duration = pulseIn(echoPin, HIGH);
  float distance = duration * 0.034 / 2;
  Serial.print(distance);

  
  if (moistureValue < moistureThreshold) {
    moistureServo.write(0);
    delay(3000);
    moistureServo.write(45);
    
    
  }else{
    moistureServo.write(90);
    delay(2000);
    moistureServo.write(45);
    delay(1000);// Open position for moisture segregation
    
  }

  if (distance < thresholdDistance) {
    openDoor();
  } else {
    closeDoor();
  }
}



void openDoor() {
  doorServo.write(50); // Open the door
   // Delay to keep the door open (adjust as needed)
}

void closeDoor() {
  doorServo.write(130); // Close the door
   // Delay to keep the door closed (adjust as needed)
}
