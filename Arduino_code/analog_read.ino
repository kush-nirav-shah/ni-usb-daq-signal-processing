const int analogPin = A0;  // Analog input pin
const unsigned long SAMPLE_INTERVAL = 100;  // microseconds between samples (10kHz sampling rate)
unsigned long lastSampleTime = 0;
const int BUFFER_SIZE = 64;  // Number of samples to buffer before sending
int buffer[BUFFER_SIZE];
int bufferIndex = 0;

void setup() {
  Serial.begin(500000);  // Higher baud rate for faster data transmission
  analogReference(DEFAULT);  // Using default 5V reference
  pinMode(analogPin, INPUT);
}

void loop() {
  unsigned long currentTime = micros();
  
  // Check if it's time for a new sample
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = currentTime;
    
    // Read analog value and store in buffer
    buffer[bufferIndex] = analogRead(analogPin);
    bufferIndex++;
    
    // When buffer is full, send data
    if (bufferIndex >= BUFFER_SIZE) {
      // Send timestamp and buffer data
      Serial.print(millis());
      Serial.print(",");
      
      // Send all samples
      for (int i = 0; i < BUFFER_SIZE; i++) {
        Serial.print(buffer[i]);
        if (i < BUFFER_SIZE - 1) {
          Serial.print(",");
        }
      }
      Serial.println();
      
      bufferIndex = 0;  // Reset buffer index
    }
  }
}
