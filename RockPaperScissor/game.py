"""single RSP game
pylint score 9.53"""


class Game():
    """Single game class"""
    player1 = None
    player2 = None
    player1_action = ''
    player2_action = ''
    result = ''

    def __init__(self, player1, player2):
        """This is where you set up the game and state the involved players"""
        self.player1 = player1
        self.player2 = player2

    def run_game(self):
        """This simulates the game and trigger the printing of result """
        self.player1_action = self.player1.action(self.player2)
        self.player2_action = self.player2.action(self.player1)
        self.result = self.winner(self.player1_action, self.player2_action)
        if (self.player1.get_class_name() == "Historian") or (
                self.player1.get_class_name() == "Most Common"):
            self.player1.save_result(self.player2, self.player2_action)
        if (self.player2.get_class_name() == "Historian") or (
                self.player2.get_class_name() == "Most Common"):
            self.player2.save_result(self.player1, self.player1_action)
        self.__str__()

    def winner(self, player1_action, player2_action):
        """ This is where the winner is decided """
        if player1_action == player2_action:
            self.player1.add_points(0.5)
            self.player2.add_points(0.5)
            return "No one"
        if player1_action == "rock":
            if player2_action == "scissors":
                self.player1.add_points(1)
                return self.player1.get_name()
            self.player2.add_points(1)
            return self.player2.get_name()
        if player1_action == "scissors":
            if player2_action == "paper":
                self.player1.add_points(1)
                return self.player1.get_name()
            self.player2.add_points(1)
            return self.player2.get_name()
        if player1_action == "paper":
            if player2_action == "rock":
                self.player1.add_points(1)
                return self.player1.get_name()
            self.player2.add_points(1)
            return self.player2.get_name()

    def __str__(self):
        """This print out the result to the interface"""
        print(
            self.player1.get_class_name() +
            " " +
            self.player1.get_name() +
            ": " +
            self.player1_action +
            ".  " +
            self.player2.get_class_name() +
            " " +
            self.player2.get_name() +
            ": " +
            self.player2_action +
            " -> " +
            self.result +
            " wins!")
