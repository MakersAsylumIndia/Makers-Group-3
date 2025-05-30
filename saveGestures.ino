// ESP32 Flex Sensor Reader - Updated for Specific Finger Pins

// Define the analog input pins for the 5 flex sensors
// NEW PIN ASSIGNMENTS:
const int thumbPin  = 27; // Thumb on GPIO27 (ADC2_CH7) - NOTE: ADC2 conflicts with Wi-Fi
const int indexPin  = 32; // Index on GPIO32 (ADC1_CH4)
const int middlePin = 33; // Middle on GPIO33 (ADC1_CH5)
const int ringPin   = 34; // Ring on GPIO34 (ADC1_CH6) - Input only
const int pinkyPin  = 35; // Pinky on GPIO35 (ADC1_CH7) - Input only

// Array to hold sensor pin numbers in the desired order for sending
// This order will be: Thumb, Index, Middle, Ring, Pinky
const int sensorPins[] = {thumbPin, indexPin, middlePin, ringPin, pinkyPin};
const int numSensors = sizeof(sensorPins) / sizeof(sensorPins[0]); // This will correctly be 5

// Array to store sensor readings
int sensorValues[numSensors];

void setup() {
  Serial.begin(115200); // Initialize serial communication at 115200 baud
  while (!Serial && millis() < 5000); // Wait for serial port to connect (timeout after 5s)

  // ESP32 ADC is 12-bit by default (0-4095)
  // analogReadResolution(12); // This is default, so not strictly necessary to set

  // No pinMode() needed for analogRead on ESP32 for ADC1 pins (GPIO32-39)
  // For ADC2 pins (like GPIO27), analogRead usually handles setting it as input,
  // but explicitly setting pinMode(pin, INPUT) sometimes helps if issues arise.
  // However, it's often not required.

  Serial.println("ESP32: 5 Flex Sensor Reader (Finger-Specific) Ready.");
  Serial.println("Sending data as: Thumb,Index,Middle,Ring,Pinky\\n");
  Serial.print("Pins used (Thumb, Index, Middle, Ring, Pinky): ");
  for(int i=0; i<numSensors; i++) {
    Serial.print(sensorPins[i]);
    if(i < numSensors -1) Serial.print(", ");
  }
  Serial.println("");
  Serial.println("NOTE: Thumb (GPIO27) is an ADC2 pin. ADC2 may have issues if Wi-Fi is active.");
}

void loop() {
  // Read all 5 sensors based on the sensorPins array
  for (int i = 0; i < numSensors; i++) {
    sensorValues[i] = analogRead(sensorPins[i]);
  }

  // Send data over serial, comma-separated, followed by a newline
  // The order will match the sensorPins array: Thumb, Index, Middle, Ring, Pinky
  for (int i = 0; i < numSensors; i++) {
    Serial.print(sensorValues[i]);
    if (i < numSensors - 1) { // Add comma for all but the last value
      Serial.print(",");
    }
  }
  Serial.println(); // Newline character indicates end of one complete data packet

  delay(100); // Send data approximately every 100ms (10Hz). Adjust as needed.
}