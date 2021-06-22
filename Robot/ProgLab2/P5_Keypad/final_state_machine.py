"""Finite State Machine class, with functions"""

from keypad_controller import KPC
from keypad import Keypad
from led_board import Led_board
import time


class FSM:

    ALL_SYMBOLS = [1, 2, 3, 4, 5, 6, 7, 8, 9, '*', '#']
    ALL_DIGITS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __init__(self, k_pad_controller, keypad, l_board):
        self.agent = k_pad_controller

        self.keypad = keypad
        self.keypad.setup()

        self.led_board = l_board
        self.led_board.setup()

        self.curr_state = "s_init"
        self.rule_list = []  # list of rules
        self.curr_signal = None

        self.rules_need_input = []

    def add_rules(self):
        """Add rule to rule list"""

        """Password rules"""
        rule_a1 = Rule('s_init', 's_read', 'all_symbols', self.agent.init_passcode_entry)
        rule_a2 = Rule('s_read', 's_read', 'all_digits', self.agent.append_next_password_digit)  # ta inn input
        rule_a3 = Rule('s_read', 's_verify', '*', self.agent.verify_password)
        rule_a4 = Rule('s_read', 's_init', 'all_symbols', self.agent.reset_agent)
        rule_a5 = Rule('s_verify', 's_active', 'Y', self.agent.fully_activate_agent)
        rule_a6 = Rule('s_verify', 's_init', 'N', self.agent.reset_agent)
        rule_a11 = Rule('s_active', 's_read_2', '*', self.agent.init_passcode_entry)
        rule_a21 = Rule('s_read_2', 's_read_2', 'all_digits', self.agent.append_next_password_digit)  # ta inn input
        rule_a7 = Rule('s_read_2', 's_read_3', '*', self.agent.cache_new_password)
        rule_a61 = Rule('s_read_2', 's_active', 'all_symbols', self.agent.refresh_agent)
        rule_a22 = Rule('s_read_3', 's_read_3', 'all_digits', self.agent.append_next_password_digit_old)  # ta inn input
        rule_a8 = Rule('s_read_3', 's_active', '*', self.agent.compare_new_passwords)
        rule_a62 = Rule('s_read_3', 's_active', 'all_symbols', self.agent.refresh_agent)

        """Light rules"""
        rule_s1 = Rule('s_active', 's_led', '0-5_digits', self.agent.choose_led)  # ta inn input
        rule_s2 = Rule('s_led', 's_time', '*', self.agent.begin_duration_entry)
        rule_s3 = Rule('s_time', 's_time', 'all_digits', self.agent.choose_duration)  # ta inn input
        rule_s4 = Rule('s_time', 's_active', '*', self.agent.complete_duration)
        rule_s5 = Rule('s_active', 's_logout', '#', self.agent.begin_logout)
        rule_s61 = Rule('s_logout', 's_init', '#', self.agent.exit_action)
        rule_s62 = Rule('s_logout', 's_active', 'all_digits', self.agent.abort_exit)

        """Add rules to rule_list"""
        self.rule_list = [rule_a1, rule_a2, rule_a3, rule_a4, rule_a5, rule_a6, rule_a11, rule_a21, rule_a7,
                               rule_a61, rule_a22, rule_a8, rule_a62, rule_s1, rule_s2, rule_s3, rule_s4, rule_s5,
                               rule_s61, rule_s62]

        """Add to need input list"""
        self.rules_need_input = [rule_a2, rule_a21, rule_a22, rule_s1, rule_s3]

    def get_next_signal(self):
        """Query the agent for the next signal"""
        self.curr_signal = self.agent.get_next_signal()

    def run_rules(self):
        """Go through the rule set, in order, applying each rule until one of the rules is fired"""
        for i in self.rule_list:
            if i.state1 == self.curr_state and self.curr_signal in i.legal_signals:
                print("curr_state:", self.curr_state, "\ti.state1:", i.state1, "\ti.legal_signals:", i.legal_signals, "\tcurr_signal:", self.curr_signal)
                print("FOUND RULE")
                self.curr_state = i.state2
                if i in self.rules_need_input:
                    i.action(self.curr_signal)
                else:
                    i.action()
                break


    def main_loop(self):
        """Use the consequent of a rule to a) set the next
        state of the FSM, and b) call the appropriate agent action method"""
        while self.curr_state != 's_done':
            print('i')
            self.get_next_signal()
            self.run_rules()

            print("curr_state:", self.curr_state)

            time.sleep(0.5)


class Rule:
    """Rule class"""

    def __init__(self, state1, state2, signal, action):
        self.state1 = state1
        if signal == 'all_symbols':
            self.legal_signals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, '*', '#']
        elif signal == 'all_digits':
            self.legal_signals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif signal == '0-5_digits':
            self.legal_signals = [0, 1, 2, 3, 4, 5]
        else:
            self.legal_signals = [signal]
        self.state2 = state2
        self.action = action


def main():
    """Main function"""

    print("START")
    keypad = Keypad()
    led_board = Led_board()
    kpc = KPC(keypad, led_board)
    fsm = FSM(kpc, keypad, led_board)
    fsm.add_rules()
    fsm.main_loop()


if __name__ == "__main__":
    main()
