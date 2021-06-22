"""
player abstract class
"""

from abc import ABC, abstractmethod


class Player(ABC):
    """Abstract Player class"""
    points = 0.0

    @abstractmethod
    def get_class_name(self):
        """Returns the Class name"""

    @abstractmethod
    def action(self, opponent):
        """Returns the action"""

    @abstractmethod
    def save_result(self, opponent, opponent_action):
        """Saves the result from a match"""

    @abstractmethod
    def add_points(self, value):
        """add points"""
        self.points += value

    @abstractmethod
    def get_points(self):
        """get points"""
        return self.points

    @abstractmethod
    def reset_points(self):
        """set points to null"""
        self.points = 0
