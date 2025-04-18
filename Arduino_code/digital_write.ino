#include <TimerOne.h>

int outputPin = 9;
long frequency = 1000;

void setup() {
    pinMode(outputPin, OUTPUT);
    Serial.begin(9600);
    Serial.println("Enter frequency in Hz:");
    
    Timer1.initialize();
    Timer1.attachInterrupt(generateSquareWave);
}

void loop() {
    if (Serial.available() > 0) {
        frequency = Serial.parseInt();
        if (frequency > 0) {
            // Calculate period for half the input frequency since we toggle twice per cycle
            long period = 500000 / frequency;  // Changed from 1000000 to 500000
            Timer1.setPeriod(period);
            Serial.print("Frequency set to: ");
            Serial.print(frequency);
            Serial.println(" Hz");
        }
    }
}

void generateSquareWave() {
    digitalWrite(outputPin, !digitalRead(outputPin));
}