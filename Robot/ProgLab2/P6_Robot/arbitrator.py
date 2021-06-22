""" Finds the most important behavior by weigth"""


class Arbitrator:
    """Decides at each timestep: which behavior wins
    and thus gets its motor recommendations transferred to the agent's motobs"""

    def choose_action(self, active_behaviors):  # stochastic solution
        """check all of the active behaviors and return a winner with highest weight
            returns motor_recommendations and halt request for winner behavior"""
        highest_weight = 0
        highest_behavior = None
        for behavior in active_behaviors:
            if behavior.priority > highest_weight:
                highest_behavior = behavior
        return highest_behavior
