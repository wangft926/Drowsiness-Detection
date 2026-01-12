import time

import serial

# Replace 'COM3' with the appropriate port for your Arduino
arduino = serial.Serial('COM3', 9600)  
time.sleep(2)  # Give time for Arduino to initialize

# Function to turn the buzzer on
def buzzer_on():
    arduino.write(b'1')  # Send '1' to Arduino to turn on the buzzer
    print("Buzzer ON")

# Function to turn the buzzer off
def buzzer_off():
    arduino.write(b'0')  # Send '0' to Arduino to turn off the buzzer
    print("Buzzer OFF")

# Example: Toggle buzzer on and off with delays
while True:
    buzzer_on()
    time.sleep(2)  # Buzzer on for 2 seconds
    buzzer_off()
    time.sleep(2)  # Buzzer off for 2 seconds
