import time

import serial

# Set up serial communication with Arduino
# Update the 'COM3' or '/dev/ttyACM0' based on your system
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)

# Function to send signal to Arduino
def send_signal(state):
    if state == 'on':
        arduino.write(b'on\n')  # Send 'on' signal
    else:
        arduino.write(b'off\n')  # Send 'off' signal

# Example usage
while True:
    # Simulate signal input (can be replaced by any logic)
    signal = input("Enter 'on' or 'off': ")
    
    if signal == 'on':
        send_signal('on')
    else:
        send_signal('off')

    time.sleep(2)  # Wait before sending the next signal
