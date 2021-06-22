"""Class serial player"""
from random import choice
from player import Player


class SerialPlayer(Player):
    """Class serial player inherits from abstract class player"""

    last_play = ""
    plays = ["rock", "scissors", "paper"]
    name = ''

    def __init__(self, name):
        self.name = name

    def get_class_name(self):
        return "Serial"

    def action(self, opponent):
        if self.last_play == "":
            play = choice(["rock", "scissors", "paper"])
            self.last_play = play
            return play
        play = self.plays[((self.plays.index(self.last_play)) + 1) % 3]
        self.last_play = play
        return play

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
