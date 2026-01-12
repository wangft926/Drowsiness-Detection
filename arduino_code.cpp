#include <MD_MAX72XX.h>
#include <SPI.h>

#define HARDWARE_TYPE MD_MAX72XX::PAROLA
#define MAX_DEVICES 4 // Number of devices in the LED matrix
#define DATA_IN 4
#define CLK_PIN 5
#define CS_PIN 6

MD_MAX72XX mx = MD_MAX72XX(HARDWARE_TYPE, DATA_IN, CLK_PIN, CS_PIN);

const int buzzerPin = 9; // Pin for the buzzer
String receivedText;

void setup() {
    Serial.begin(9600); // Initialize serial communication
    mx.begin(); // Initialize the LED matrix
    pinMode(buzzerPin, OUTPUT); // Set buzzer pin as output
}

void loop() {
    // Check if data is available to read
    if (Serial.available() > 0) {
        receivedText = Serial.readStringUntil('\n'); // Read until newline
        handleCommands(receivedText);
    }
}

void handleCommands(String command) {
    if (command == "DROWSY") {
        mx.print("DROWSY!"); // Display "DROWSY!" on the LED matrix
        digitalWrite(buzzerPin, HIGH); // Activate the buzzer
    } else if (command == "ATTENTIVE") {
        mx.print("ATTENTIVE"); // Display "ATTENTIVE" on the LED matrix
        digitalWrite(buzzerPin, LOW); // Deactivate the buzzer
    } else {
        mx.print("NORMAL"); // Default state
        digitalWrite(buzzerPin, LOW); // Deactivate the buzzer
    }
}
