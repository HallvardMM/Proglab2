'''Led Board class, with functions'''

import time
import RPi.GPIO as GPIO


class Led_board:

    pins = [13, 20, 21]

    pin_led_states = [
        [1, 0, -1],  # A
        [0, 1, -1],  # B
        [-1, 1, 0],  # C
        [-1, 0, 1],  # D
        [1, -1, 0],  # E
        [0, -1, 1],  # F
        [0, 0, 0]    # All leds off
    ]

    def setup(self):
        """Set the proper mode via: GPIO.setmode(GPIO.BCM)"""
        GPIO.setmode(GPIO.BCM)

    def set_pin(self, pin_index, pin_state):
        if pin_state == -1:
            GPIO.setup(self.pins[pin_index], GPIO.IN)
        else:
            GPIO.setup(self.pins[pin_index], GPIO.OUT)
            GPIO.output(self.pins[pin_index], pin_state)

    def light_led(self, led_number):
        """Turn on one of the 6 LEDs by making the appropriate combination of input and output declarations,
         and then making the appropriate HIGH / LOW settings on the output pins"""
        for pin_index, pin_state in enumerate(self.pin_led_states[led_number]):
            self.set_pin(pin_index, pin_state)

    def light_one_led(self, led_number, k):
        t_end = time.time() + k
        while time.time() < t_end:
            self.light_led(led_number)
        self.light_led(6)

    def flash_all_leds(self, k):
        """Flash all 6 LEDs on and off for k seconds, where k is an argument of the method"""
        t_end = time.time() + k
        while time.time() < t_end:
            for i in range(300):
                self.light_led(0)
                self.light_led(1)
                self.light_led(2)
                self.light_led(3)
                self.light_led(4)
                self.light_led(5)
            self.light_led(6)
            time.sleep(0.5)
        self.light_led(6)

    def twinkle_all_leds(self, k):
        """Turn all LEDs on and off in sequence for k seconds, where k is an argument of the method"""
        t_end = time.time() + k
        while time.time() < t_end:
            self.light_led(0)
            time.sleep(0.2)
            self.light_led(1)
            time.sleep(0.2)
            self.light_led(2)
            time.sleep(0.2)
            self.light_led(3)
            time.sleep(0.2)
            self.light_led(4)
            time.sleep(0.2)
            self.light_led(5)
            time.sleep(0.2)
        self.light_led(6)

    def power_up(self):
        """Led flash when turning on system"""
        t_end = time.time() + 3
        while time.time() < t_end:
            self.light_led(5)
        self.light_led(6)

    def power_down(self):
        """Led flash when turning off system"""
        t_end = time.time() + 3
        while time.time() < t_end:
            self.light_led(4)
        self.light_led(6)
