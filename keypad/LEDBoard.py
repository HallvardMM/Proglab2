""" led board """
import RPi.GPIO as GPIO
import time
import random as rand


class LEDBoard:
    """ LED Board class """
    mode = None
    # Helper-array to setup pin setting to light a particular LED (LED i at
    # index i)
    LED_pin_settings = []
    pins = [16, 20, 21]  # this is subject to change, pins[i] refers to pin i

    def __init__(self):
        """ init """
        self.setup()

    def setup(self):
        """ Set the proper mode via: GPIO.setmode(GPIO.BCM)  """
        self.mode = GPIO.BCM
        # Might use GPIO.BOARD instead, depends on the input pins
        GPIO.setmode(GPIO.BCM)
        self.LED_pin_settings = [  # 0=LOW, 1=HIGH and -1=INPUT
            [1, 0, -1],     # to light up LED 0
            [0, 1, -1],     # to light up LED 1
            [0, -1, 1],     # to light up LED 2
            [-1, 0, 1],     # to light up LED 3
            [-1, 1, 0],     # to light up LED 4
            [1, -1, 0]      # to light up LED 5
        ]

    def set_pin(self, pin_index, pin_state):
        """Sets the pin"""
        if pin_state == -1:
            GPIO.setup(self.pins[pin_index], GPIO.IN)
        else:
            GPIO.setup(self.pins[pin_index], GPIO.OUT)
            # pin_state should be GPIO.HIGH or GPIO.LOW
            GPIO.output(self.pins[pin_index], pin_state)

    def light_led(self, led, k):
        """ Turn on one of the 6 LEDs for k seconds """
        for pin_index, pin_state in enumerate(self.LED_pin_settings[led]):
            self.set_pin(pin_index, pin_state)
        time.sleep(k)
        for pin_index in self.LED_pin_settings[led]:
            self.set_pin(pin_index, -1)  # reset the pins to turn off the LED

    def flash_all_leds(self):
        """ Flash all 6 LEDs on and off, used when the user enters the wrong passcode."""
        for times in range(3):  # flash all the LEDs three times on and off
            for rounds in range(100):
                for led in range(6):
                    self.light_led(led, 0.001)
            time.sleep(0.4)

    def twinkle_all_leds(self):
        """  The lightshow to run when the user successfully authenticates """
        for rounds in range(50):
            self.light_led(rand.randint(0, 5), 0.05)

    def startup_lightshow(self):
        """ The lightshow sequence to run when the keypad is started """
        for rounds in range(3):  # 3 rounds, clockwise
            for led in range(6):
                self.light_led(led, 0.15)

    def shutdown_lightshow(self):
        """The lightshow sequence to run when the keypad is shut down"""
        for rounds in range(3):
            for led in range(5, -1, -1):  # 3 rounds,  counterclockwise
                self.light_led(led, 0.15)
