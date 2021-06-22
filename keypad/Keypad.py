""" keypad """

import time
import RPi.GPIO as GPIO


class Keypad:
    """ keypad class """
    r_pins = [
        18,
        23,
        24,
        25]   # pins 18, 23, 24 and 25 for key rows 0, 1, 2 and 3
    c_pins = [17, 27, 22]       # pins 17, 27 and 22 for key columns 0, 1 and 2

    keypad = [['1', '2', '3'], ['4', '5', '6'],
              ['7', '8', '9'], ['*', '0', '#']]

    def __init__(self):
        """ Set the mode and the row pins as outputs and the column pins as inputs """
        GPIO.setmode(GPIO.BCM)
        for rp in range(0, 4):
            GPIO.setup(self.r_pins[rp], GPIO.OUT)
        for cp in range(0, 3):
            GPIO.setup(self.c_pins[cp], GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    def do_polling(self):
        """ Use nested loops (discussed above) to determine the
        key currently being pressed on the keypad. """
        for rp in range(0, 4):
            GPIO.output(self.r_pins[rp], GPIO.HIGH)
            for cp in range(0, 3):
                if GPIO.input(self.c_pins[cp]) == GPIO.HIGH:
                    time.sleep(1)
                    if GPIO.input(self.c_pins[cp]) == GPIO.HIGH:
                        print("Keypad ret: ", self.keypad[rp][cp])
                        return self.keypad[rp][cp]
            GPIO.output(self.r_pins[rp], GPIO.LOW)

    def get_next_signal(self):
        """ Calls do_polling until a key press is detected"""
        signal = ''
        while not signal:
            signal = self.do_polling()
        return signal
