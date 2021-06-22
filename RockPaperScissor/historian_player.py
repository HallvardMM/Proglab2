"""Class historian player"""
from random import choice
from player import Player


class HistorianPlayer(Player):
    """Class random player innherets from abstract class player"""

    # previously saved results and grade of memory
    opponents_statistics = {}
    memory = None

    def __init__(self, name, memory):
        self.memory = memory
        self.name = name

    def get_class_name(self):
        return "Historian"

    def action(self, opponent):
        decision = {
            "rock": 0,
            "scissors": 0,
            "paper": 0
        }
        opponent_record = []
        for previous_opponent in self.opponents_statistics:
            if opponent == previous_opponent:
                opponent_record = self.opponents_statistics[opponent]
                break
        if not opponent_record:
            return choice(["rock", "scissors", "paper"])
        record_length = len(opponent_record)
        memory_lane = opponent_record[(
            record_length - self.memory):record_length]
        for play in range(self.memory, record_length):
            if opponent_record[(play - self.memory):play] == memory_lane:
                decision[opponent_record[play]] += 1
        if (decision["rock"] >= decision["scissors"]) and (
                decision["rock"] >= decision["paper"]):
            return "paper"
        if (decision["scissors"] >= decision["rock"]) and (
                decision["scissors"] >= decision["paper"]):
            return "rock"
        return "scissors"

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
