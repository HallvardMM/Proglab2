'''Keypad class, with functions'''

import time
import RPi.GPIO as GPIO

ROWS = [18, 23, 24, 25]
COLUMNS = [17, 27, 22]
SIGNALS = {
    (18, 17): 1,
    (18, 27): 2,
    (18, 22): 3,
    (23, 17): 4,
    (23, 27): 5,
    (23, 22): 6,
    (24, 17): 7,
    (24, 27): 8,
    (24, 22): 9,
    (25, 17): '*',
    (25, 27): 0,
    (25, 22): '#',
}


class Keypad:

    """Keypad class"""

    def __init__(self):
        pass

    def setup(self):
        """ Set the proper mode via: GPIO.setmode(GPIO.BCM). Also, use GPIO functions to set the row
         pins as outputs and the column pins as inputs"""

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(18, GPIO.OUT)
        GPIO.setup(23, GPIO.OUT)
        GPIO.setup(24, GPIO.OUT)
        GPIO.setup(25, GPIO.OUT)

        GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    def do_polling(self):
        ''' Use nested loops (discussed above) to determine the key currently being pressed on the keypad'''

        row_col = ()    # Tuple holding the row, col which was pressed

        for row in ROWS:
            GPIO.output(row, GPIO.HIGH)

            for col in COLUMNS:

                if GPIO.input(col) == GPIO.HIGH:
                    time.sleep(0.01)    # Sleep to verify that it's still HIGH
                    if GPIO.input(col) == GPIO.HIGH:
                        row_col = (row, col)

            GPIO.output(row, GPIO.LOW)

        return row_col

    def get_next_signal(self, debug=False):
        '''This is the main interface between the agent and the keypad.
         It should initiate repeated calls to do polling until a key press is detected'''

        pressed = False

        while not pressed:

            row_col = self.do_polling()
            if row_col:
                if debug:   # Check the input
                    print(row_col)
                    print(SIGNALS[row_col])
                return SIGNALS[row_col]
