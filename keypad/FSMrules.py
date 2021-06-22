""" the rules for the final state machine """

def init_rule_condition(state, signal):
    """ type password"""
    return state == "s"


init_rule_consequence = ["s0", -1]


def state_0_rule_condition(state, signal):
    """type password"""
    return state == "s0" and signal != '*'


state_0_rule_consequence = ["s0", 0]


def state_0_rule_condition_2(state, signal):
    """commit password"""
    return state == 's0' and signal == '*'


state_0_rule_consequence_2 = ["s1", 1]


def state_1_rule_condition(state, signal):
    """set led id"""
    return state == 's1' and signal in ['1', '2', '3', '4', '5', '6']


state_1_rule_consequence = ["s2", 2]


def state_1_rule_condition_2(state, signal):
    """initiate new password"""
    return state == 's1' and signal == '8'


state_1_rule_consequence_2 = ["s3", 5]


def state_2_rule_condition(state, signal):
    """set led dur"""
    return state == 's2' and signal != '*'


state_2_rule_consequence = ["s2", 3]


def state_2_rule_condition_2(state, signal):
    """commit led dur"""
    return state == 's2' and signal == '*'


state_2_rule_consequence_2 = ["s1", 4]


def state_3_rule_condition(state, signal):
    """type new password"""
    return state == 's3' and signal != '*'


state_3_rule_consequence = ["s3", 0]


def state_3_rule_condition_2(state, signal):
    """commit first new password"""
    return state == 's3' and signal == '*'


state_3_rule_consequence_2 = ["s4", 6]


def state_4_rule_condition(state, signal):
    """type confirmed new password"""
    return state == 's4' and signal != '*'


state_4_rule_consequence = ["s4", 0]


def state_4_rule_condition_2(state, signal):
    """commit confirmed new password"""
    return state == 's4' and signal == '*'


state_4_rule_consequence_2 = ["s1", 7]


def end_rule_condition(state, signal):
    """test end"""
    return signal == '#'


end_rule_consequence = ["s", 8]


class FSMrules:
    """ Class to add rules to FSM """

    def __init__(self, FSM):
        """ init """
        FSM.add_rule([init_rule_condition, init_rule_consequence])
        FSM.add_rule([end_rule_condition, end_rule_consequence])
        FSM.add_rule([state_0_rule_condition, state_0_rule_consequence])
        FSM.add_rule([state_0_rule_condition_2, state_0_rule_consequence_2])
        FSM.add_rule([state_1_rule_condition, state_1_rule_consequence])
        FSM.add_rule([state_1_rule_condition_2, state_1_rule_consequence_2])
        FSM.add_rule([state_2_rule_condition, state_2_rule_consequence])
        FSM.add_rule([state_2_rule_condition_2, state_2_rule_consequence_2])
        FSM.add_rule([state_3_rule_condition, state_3_rule_consequence])
        FSM.add_rule([state_3_rule_condition_2, state_3_rule_consequence_2])
        FSM.add_rule([state_4_rule_condition, state_4_rule_consequence])
        FSM.add_rule([state_4_rule_condition_2, state_4_rule_consequence_2])
