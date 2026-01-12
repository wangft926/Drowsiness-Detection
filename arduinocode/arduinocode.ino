#define BUZZER_PIN 9    // Buzzer pin

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);  // Set buzzer pin as output
  Serial.begin(9600);           // Initialize serial communication
}

void loop() {
  // Check if data is available from the serial port
  if (Serial.available() > 0) {
    char command = Serial.read(); // Read the incoming byte

    // Check the command and control the buzzer accordingly
    if (command == 'Y' || command == 'E') {
      soundBuzzer();  // Sound the buzzer for drowsiness
    } else if (command == 'N') {
      noTone(BUZZER_PIN);  // Turn off the buzzer when normal
    }
  }
}

void soundBuzzer() {
  for (int i = 0; i < 5; i++) {  // Repeat the sound for a few times
    digitalWrite(BUZZER_PIN, HIGH);  // Turn on the buzzer
    delay(500);                      // Buzzer on for 500 milliseconds
    digitalWrite(BUZZER_PIN, LOW);   // Turn off the buzzer
    delay(500);                      // Buzzer off for 500 milliseconds
  }
}
