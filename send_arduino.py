import time

import serial

# Open the serial connection to the Arduino
arduino = serial.Serial('COM3', 9600)  # Replace 'COM3' with your correct port
time.sleep(2)  # Wait for the connection to establish

# Test sending data to Arduino
arduino.write(b'0')  # Send 0 for normal state
time.sleep(5)  # Wait and observe the behavior on the LED matrix

arduino.write(b'1')  # Send 1 for alert (drowsiness detected)
time.sleep(5)  # Observe the blinking behavior

arduino.close()  # Close the connection
