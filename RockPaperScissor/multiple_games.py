"""Multiple games file - arrange multiple RSP games between two players
pylint score 8.08 beacuse it doesn't recognise matplotlib import"""
from game import Game
from matplotlib.pyplot import plot, show


class MultipleGames():
    """Multiple games class - involved players and number of games"""

    player1 = ''
    player2 = ''
    nr_of_games = 0

    def __init__(self, player1, player2, nr_of_games):
        """This is where you set up the game and state the involved players"""
        self.player1 = player1
        self.player2 = player2
        self.nr_of_games = nr_of_games

    def run_games(self):
        """This simulates the game and trigger the printing of result"""
        x_values = []
        player1_y_values = []
        player2_y_values = []
        for times in range(0, self.nr_of_games):
            single_game = Game(self.player1, self.player2)
            single_game.run_game()
            x_values.append(times)
            player1_y_values.append(
                float(self.player1.get_points()) / int(times + 1))
            player2_y_values.append(
                float(self.player2.get_points()) / int(times + 1))
            self.__str__()
        plot(x_values, player1_y_values, color='Blue')
        plot(x_values, player2_y_values, color='Red')
        show()

    def __str__(self):
        """This print out the result to the interface"""
        print("Score:\n" +
              self.player1.get_class_name() +
              " " +
              self.player1.get_name() +
              ": " +
              str(self.player1.get_points()) +
              "\n" +
              self.player2.get_class_name() +
              " " +
              self.player2.get_name() +
              ": " +
              str(self.player2.get_points()))
