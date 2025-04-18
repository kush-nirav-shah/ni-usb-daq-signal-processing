const int digitalPin = 2;

volatile unsigned long lastChangeTime = 0;  // Store the last time the pin state changed (µs)
volatile unsigned long period = 0;         // Store the time between two state changes (µs)
volatile int lastState = HIGH;             // Store the last state of the pin
volatile bool stateChanged = false;        // Flag to indicate state change
float frequency = 0;                       // Store the calculated frequency

void setup() {
  Serial.begin(230400);
  pinMode(digitalPin, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(digitalPin), stateChange, CHANGE);
}

void loop() {
  if (stateChanged) {
    stateChanged = false;  // Reset flag
    if (period > 0) {
      frequency = (1000000.0 / period) / 2;  // Convert period to Hz (period is in µs)
      Serial.print(frequency);
      Serial.print(", ");
      Serial.println(digitalRead(digitalPin));  // Print the current state
    }
  }
}

void stateChange() {
  unsigned long currentTime = micros();  // Use micros() for higher accuracy
  int currentState = digitalRead(digitalPin);  

  if (currentState != lastState) {  
    period = currentTime - lastChangeTime;  // Calculate period in µs
    lastChangeTime = currentTime;  
    lastState = currentState;  
    stateChanged = true;  // Set flag for main loop to process
  }
}