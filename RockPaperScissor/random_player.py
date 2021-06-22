"""Class random player"""
from random import choice
from player import Player


class RandomPlayer(Player):
    """Class random player inherits from abstract class player"""

    name = ''

    def __init__(self, name):
        self.name = name

    def get_class_name(self):
        return "Random"

    def action(self, opponent):
        return choice(["rock", "scissors", "paper"])

    def save_result(self, opponent, opponent_action):
        pass

    def add_points(self, value):
        self.points += value

    def get_points(self):
        return self.points

    def get_name(self):
        """get name"""
        return self.name

    def reset_points(self):
        self.points = 0
