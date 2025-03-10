/*
  Analog input, wave output, serial output with rate of change monitoring

  Reads an analog input pin and uses it to control the wave frequency
  while outputting a sine wave to the PWM pin.
  Also prints the results and rate of change to the Serial Monitor.

  The circuit:
  - potentiometer connected to analog pin 0.
    Center pin of the potentiometer goes to the analog pin.
    side pins of the potentiometer go to +5V and ground
  - LED connected from digital pin 9 to ground through 220 ohm resistor

  modified from original code by Tom Igoe
*/

// These constants won't change. They're used to give names to the pins used:
const int analogInPin = A0;  // Analog input pin that the potentiometer is attached to
const int analogOutPin = 9;  // Analog output pin that the LED is attached to

int sensorValue = 0;         // value read from the pot
int outputValue = 0;         // value output to the PWM (analog out)
int previousOutputValue = 0; // previous output value to calculate rate of change
float angle = 0.0;           // angle for sine wave generation
float frequency = 0.1;       // base frequency of the wave
float frequencyFactor;       // adjust frequency based on potentiometer
int rateOfChange = 0;        // stores the rate of change of the output value
unsigned long lastDisplayTime = 0; // timestamp for status updates

// Define sine wave lookup table (values mapped to 0-255)
const int WAVE_TABLE_SIZE = 256;
byte sineWave[WAVE_TABLE_SIZE];

// Status messages based on rate of change
const char* getChangeStatus(int rate) {
  if (rate == 0) return "0";
  if (abs(rate) < 4) return "1";
  if (abs(rate) < 8) return "2";
  if (abs(rate) < 15) return "3";
  return "RAPID";
}

void setup() {
  // initialize serial communications at 9600 bps:
  Serial.begin(9600);
  
  // Pre-calculate sine wave values mapped to 0-255 range
  for (int i = 0; i < WAVE_TABLE_SIZE; i++) {
    // Calculate sine value (0 to 2π) and map from -1.0...1.0 to 0...255
    float sinVal = sin(i * (2.0 * PI / WAVE_TABLE_SIZE));
    sineWave[i] = byte(127.5 + 127.5 * sinVal);
  }
  
  Serial.println("Wave Generator with Rate of Change Monitor");
  Serial.println("----------------------------------------");
}

void loop() {
  // Read the analog input value:
  sensorValue = analogRead(analogInPin);
  
  // Map sensor value to adjust frequency (higher value = faster wave)
  frequencyFactor = map(sensorValue, 0, 1023, 1, 20) / 10.0;
  frequency = 0.1 * frequencyFactor;
  
  // Save the previous output value for rate of change calculation
  previousOutputValue = outputValue;
  
  // Calculate the index into our sine lookup table
  int waveIndex = int(angle) % WAVE_TABLE_SIZE;
  
  // Get the output value from our lookup table
  outputValue = sineWave[waveIndex];
  
  // Output the value to the PWM pin
  analogWrite(analogOutPin, outputValue);
  
  // Calculate rate of change
  rateOfChange = outputValue - previousOutputValue;
  
  // Display status update every 250ms to avoid flooding the serial monitor
  unsigned long currentTime = millis();
  if (currentTime - lastDisplayTime >= 250) {
    lastDisplayTime = currentTime;
    

    Serial.println(getChangeStatus(rateOfChange));
    
  }
  
  // Increment the angle for the next cycle
  angle += frequency;
  if (angle >= WAVE_TABLE_SIZE) {
    angle -= WAVE_TABLE_SIZE;
  }
  
  // Small delay
  delay(2);
}
