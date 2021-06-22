"""Class most common player"""
from random import choice
from player import Player


class MostCommonPlayer(Player):
    """Class most common player inherits from abstract class player"""

    opponents_statistics = {}
    name = ''

    def __init__(self, name):
        self.name = name

    def get_class_name(self):
        return "Most Common"

    def action(self, opponent):
        registered_opponent = False
        for previous_opponents in self.opponents_statistics:
            if previous_opponents == opponent:
                registered_opponent = True
                break
        if registered_opponent and self.opponents_statistics[opponent]:
            decision = {
                "rock": 0,
                "scissors": 0,
                "paper": 0
            }
            for previous_action in self.opponents_statistics[opponent]:
                decision[previous_action] += 1
            if (decision["rock"] >= decision["scissors"]) and (
                    decision["rock"] >= decision["paper"]):
                return "paper"
            if (decision["scissors"] >= decision["rock"]) and (
                    decision["scissors"] >= decision["paper"]):
                return "rock"
            return "scissors"
        return choice(['rock', 'scissors', 'paper'])

    def save_result(self, opponent, opponent_action):
        opponent_record = []
        for previous_opponent in self.opponents_statistics:
            if opponent == previous_opponent:
                opponent_record = self.opponents_statistics[opponent]
                break
        opponent_record.append(opponent_action)
        self.opponents_statistics[opponent] = opponent_record

    def add_points(self, value):
        self.points += value

    def get_points(self):
        return self.points

    def get_name(self):
        """get name"""
        return self.name

    def reset_points(self):
        self.points = 0
