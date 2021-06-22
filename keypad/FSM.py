""" finite state machine """

from FSMrules import FSMrules


class FiniteStateMachine:
    """ Finite state machine class """

    states = ["s0", "s1", "s2", "s3", "s4", "s5"]
    # s0: login
    # s1: [1-6] led id. 8 change password.
    # s2: led dur
    # s3: new passcode
    # s4: confirm passcode change
    # s5: logged out
    state = "s"
    signal = None
    kpc_pointer = None
    FSM_rule_list = []

    def __init__(self, agent):
        """ init """
        self.kpc_pointer = agent
        FSMrules(self)
        self.main_loop()

    def add_rule(self, new_rule):
        """ Adds a new rule to end of FSM_rule_list """
        self.FSM_rule_list.append(new_rule)

    def get_next_signal(self):
        """ Query the agent for next signal """
        return self.kpc_pointer.get_next_signal()

    def run_rules(self):
        """ Try each rule until one of the rules is fired """
        for rule in self.FSM_rule_list:
            if self.apply_rule(rule):
                self.fire_rule(rule)
                break

    def apply_rule(self, rule):
        """ Check whether the conditions of a rule are met """
        return rule[0](self.state, self.signal)

    def fire_rule(self, rule):
        """ Use the consequent of a rule to set the next state of the FSM
        and call the appropriate agent action method """
        if rule[1][1] == -1:
            self.kpc_pointer.init_passcode_entry()
            self.kpc_pointer.clear_end_buffer()
            self.state = rule[1][0]
        elif rule[1][1] == 0:
            self.kpc_pointer.add_to_buffer(self.signal)
            self.kpc_pointer.clear_end_buffer()
            self.state = rule[1][0]
        elif rule[1][1] == 1:
            login = self.kpc_pointer.verify_login()
            self.kpc_pointer.clear_end_buffer()
            if login:
                self.state = rule[1][0]
        elif rule[1][1] == 2:
            self.kpc_pointer.set_led_id(self.signal)
            self.kpc_pointer.clear_end_buffer()
            self.state = rule[1][0]
        elif rule[1][1] == 3:
            self.kpc_pointer.add_to_buffer(self.signal)
            self.kpc_pointer.clear_end_buffer()
            self.state = rule[1][0]
        elif rule[1][1] == 4:
            self.kpc_pointer.input_buffer_to_led_duration()
            self.kpc_pointer.clear_end_buffer()
            self.kpc_pointer.light_one_led()
            self.state = rule[1][0]
        elif rule[1][1] == 5:
            self.kpc_pointer.init_passcode_entry()
            self.kpc_pointer.clear_end_buffer()
            self.state = rule[1][0]
        elif rule[1][1] == 6:
            self.kpc_pointer.clear_end_buffer()
            valid_passcode = self.kpc_pointer.set_passcode_change()
            if valid_passcode:
                self.state = rule[1][0]
        elif rule[1][1] == 7:
            self.kpc_pointer.clear_end_buffer()
            validated_passcode = self.kpc_pointer.validate_passcode_change()
            if validated_passcode:
                self.state = rule[1][0]
            else:
                self.state = "s3"
        elif rule[1][1] == 8:
            end_session = self.kpc_pointer.test_end(self.signal)
            if end_session:
                self.state = rule[1][0]

    def main_loop(self):
        """ The loop running the state machine until final state """
        self.state = "s"
        while True:
            print("CURRENT STATE: ", self.state)
            self.signal = self.get_next_signal()
            if self.signal:
                self.run_rules()
            print("\n")
