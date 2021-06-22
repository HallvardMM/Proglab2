'''Keypad Controller class, with functions'''


class KPC:
    """Class for keypad_controller"""

    def __init__(self, p_keypad, p_ledboard):
        self.pointer_keypad = p_keypad
        self.pointer_ledboard = p_ledboard
        self.password_buffer_old = ""
        self.password_buffer = ""
        self.current_password = self.load_password()
        self.override_signal = None
        self.lid = 0
        self.ldur = ""

    def init_passcode_entry(self):
        """Clear the passcode-buffer and initiate a ”power up” lighting sequence on the LED Board. This should be done when the user first presses the keypad"""
        self.password_buffer = ""
        self.pointer_ledboard.power_up()
        print("POWER UP")

    def get_next_signal(self):
        """Return the override-signal, if it is non-blank; otherwise query the keypad for the next pressed key"""
        if self.override_signal:
            return self.override_signal
        return self.pointer_keypad.get_next_signal()

    def begin_logout(self):
        """Begins logout"""
        print("Begin logout: Press # if you want to logout")

    def reset_password_accumulator(self):
        """Reset password"""
        self.password_buffer = ""
        print("Enter new password: ")

    def append_next_password_digit_old(self, digit):
        """Append digit to password buffer old"""
        self.password_buffer_old += str(digit)

    def append_next_password_digit(self, digit):
        """Append digit to password_buffer"""
        self.password_buffer += str(digit)

    def cache_new_password(self):
        """State change"""
        print("Write new password again")

    def refresh_agent(self):
        """Refresh agent"""
        self.password_buffer = ""
        self.password_buffer_old = ""

    def fully_activate_agent(self):
        """Print message"""
        self.override_signal = None
        print("Right password, you are now active.")

    def reset_agent(self):
        """Print error message"""
        self.password_buffer = ""
        self.override_signal = None
        print("Wrong password, try again.")

    def verify_password(self):
        """Check that the password just entered via the keypad matches that in the pass- word file.
        Store the result (Y or N) in the override-signal. Also, this should call the LED Board to initiate the appropriate
         lighting pattern for login success or failure"""
        print("Checking password...")
        if self.password_buffer == self.current_password:
            self.twinkle_leds(3)
            self.override_signal = "Y"
            print("Login accepted")
        else:
            self.password_buffer = ""
            self.flash_leds(3)
            self.override_signal = "N"
            print("Wrong password")

    def compare_new_passwords(self):

        """Check that the new password is legal. If so, write the new pass- word in the password file.
        A legal password should be at least 4 digits long and should contain no symbols other than the digits 0-9.
        As in verify login, this should use the LED Board to signal success or failure in changing the password"""
        if len(self.password_buffer) >= 4 and (self.password_buffer.isdigit()) and (self.password_buffer == self.password_buffer_old):

            self.save_password(self.password_buffer)
            self.current_password = self.load_password()

            # self.current_password = self.password_buffer
            self.twinkle_leds(3)

            print("Your password has been changed")
            self.password_buffer = ""
            self.password_buffer_old = ""
        else:
            self.flash_leds(3)
            print("The passwords does not correspond with each other")
            self.password_buffer = ""
            self.password_buffer_old = ""

    def choose_led(self, digit):
        """Choose led to light"""
        self.lid = digit
        print("Press * to continue")

    def choose_duration(self, digit):
        """Choose duration"""
        self.ldur += str(digit)

    def begin_duration_entry(self):
        """Duration"""
        print("Write the duration")

    def complete_duration(self):
        """Using values stored in the Lid and Ldur slots, call the LED Board and request that LED # Lid be turned on for Ldur seconds"""
        self.pointer_ledboard.light_one_led(self.lid, int(self.ldur))

    def flash_leds(self, k):
        """Call the LED Board and request the flashing of all LEDs"""
        self.pointer_ledboard.flash_all_leds(k)

    def twinkle_leds(self, k):
        """Call the LED Board and request the twinkling of all LEDs"""
        self.pointer_ledboard.twinkle_all_leds(k)

    def exit_action(self):
        """Call the LED Board to initiate the ”power down” lighting sequence"""
        self.pointer_ledboard.power_down()
        print("Exiting")

    def abort_exit(self):
        """Abort exit"""
        print("Abort logout")

    def save_password(self, password):
        file = open("password.txt", "w")
        file.write(password)
        file.close()

    def load_password(self):
        file = open("password.txt", "r")
        password = file.read().strip()
        file.close()
        return password
