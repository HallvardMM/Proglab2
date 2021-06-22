""" KPC Agent """

from Keypad import Keypad
from LEDBoard import LEDBoard
from FSM import FiniteStateMachine


class KPCAgent:
    """ KPC Agent class """

    keypad_pointer = None
    led_board_pointer = None
    input_buffer = None
    change_passcode_buffer = None
    end_buffer = ""
    password_path = "password.txt"
    override_signal = None
    led_id = None
    led_duration = None

    def __init__(self):
        """ init """
        self.keypad_pointer = Keypad()
        self.led_board_pointer = LEDBoard()
        FiniteStateMachine(self)

    def init_passcode_entry(self):
        """ Clear the passcode buffer and initiate a power up lighting sequence on the LED Board """
        self.input_buffer = ""
        self.led_board_pointer.startup_lightshow()

    def get_next_signal(self):
        """ Return query keypad for next signal """
        return self.keypad_pointer.get_next_signal()

    def set_led_id(self, Lid):
        """ set led id """
        self.led_id = int(Lid) - 1

    def add_to_buffer(self, signal):
        """ Add signal from FSM to self.input_buffet """
        self.input_buffer += signal
        print("BUFFER UPDATE: ", self.input_buffer)

    def input_buffer_to_led_duration(self):
        """input buffer for the led duration"""
        self.led_duration = int(self.input_buffer)
        self.input_buffer = ""

    def test_end(self, signal):
        """ end session """
        if(len(self.end_buffer) > 0):
            self.shutdown_lightshow()
            return True
        else:
            self.end_buffer += signal
            return False

    def clear_end_buffer(self):
        """ clear self.end_buffer """
        self.end_buffer = ""

    def verify_login(self):
        """ Check that the password just entered via the keypad matches that in the password file. """
        f = open(self.password_path, "r")
        passcode = ""
        if f.mode == "r":
            passcode = f.readline()
        f.close()
        print("PASSCODE: ", passcode)
        if self.input_buffer == passcode:  # the input buffer should contain the current passcode input
            self.twinkle_all_leds()
            self.input_buffer = ""
            return True
        else:
            self.flash_leds()
            self.input_buffer = ""
            return False

    def set_passcode_change(self):
        """  Hold the new password for confirmation. """
        valid_passcode = self.is_legal_passcode(self.input_buffer)
        if valid_passcode:
            self.change_passcode_buffer = self.input_buffer
            self.led_id = 3
            self.led_duration = 2.5
            self.light_one_led()
            self.input_buffer = ""
            return True
        else:
            self.flash_leds()
            self.input_buffer = ""
            return False

    def validate_passcode_change(self):
        """  Check that the new password is legal. If so, write the new password in the password file."""
        validate_passcode = (self.change_passcode_buffer == self.input_buffer)
        if validate_passcode:
            f = open(self.password_path, "w")  # open file for writing
            if f.mode == "w":
                f.write(self.input_buffer)
            f.close()
            self.led_id = 4
            self.led_duration = 2.5
            self.light_one_led()
            self.change_passcode_buffer = ""
            self.input_buffer = ""
            return True
        else:
            self.flash_leds()
            self.input_buffer = ""
            return False

    def is_legal_passcode(self, passcode):
        """ Returns true if passcode is legal, false if not.
         A legal password should be at least 4 digits long."""
        return (len(passcode) >= 4) and passcode.isdigit()

    def light_one_led(self):
        """  Using values stored in the Lid and Ldur slots,
        call the LED Board and request that LED # Lid be turned on for Ldur seconds. """
        self.led_board_pointer.light_led(self.led_id, self.led_duration)

    def flash_leds(self):
        """ Call the LED Board and request the flashing of all LEDs. """
        self.led_board_pointer.flash_all_leds()

    def twinkle_all_leds(self):
        """  Call the LED Board and request the twinkling of all LEDs. """
        self.led_board_pointer.twinkle_all_leds()

    def shutdown_lightshow(self):
        """ Call the LED Board to initiate the power down lighting sequence. """
        self.led_board_pointer.shutdown_lightshow()
