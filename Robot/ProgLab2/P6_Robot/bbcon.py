"""The highest-level class, BBCON (Behavior-Based Controller)"""

import time

from arbitrator import Arbitrator
from sensob import Sensob
from behavior import Behavior
from motob import Motob

from Help_Classes.imager2 import Imager
from Help_Classes.camera import Camera
from Help_Classes.reflectance_sensors import ReflectanceSensors 
from Help_Classes.ultrasonic import Ultrasonic
from Help_Classes.zumo_button import ZumoButton

class BBCON:

    def __init__(self):
        """ init """
        self.sensobs = []
        self.add_sensob(Sensob(Ultrasonic()))
        self.add_sensob(Sensob(ReflectanceSensors()))
        self.add_sensob(Sensob(Camera()))

        self.motob = Motob()

        self.behaviors = []
        self.add_behavior(Behavior(self, [10000, 10000, [10000, 10000, 10000]], "drive", 1))
        self.add_behavior(Behavior(self, [30, 10000, [10000, 10000, 10000]], "stop", 3))
        self.add_behavior(Behavior(self, [10000, 0.3, [10000, 10000, 10000]], "turnaround", 2))
        self.add_behavior(Behavior(self, [10000, 10000, [210, 10000, 10000]], "turn_left", 5))
        #self.add_behavior(Behavior(self, [10000, 10000, [10000, 200, 10000]], "turn_right", 4))
        self.active_behaviors = []

        self.arbitrator = Arbitrator()

    def add_behavior(self, behavior):
        """append a newly-created behavior onto the behaviors list"""
        self.behaviors.append(behavior)

    def add_sensob(self, sensor):
        """append a newly-created sensob onto the sensobs list"""
        self.sensobs.append(sensor)

    def activate_behavior(self, behavior):
        """add an existing behavior onto the active-behaviors list"""
        self.active_behaviors.append(behavior)

    def deactivate_behavior(self, behavior):
        """remove an existing behavior from the active behaviors list"""
        self.active_behaviors.remove(behavior)

    def run_one_timestep(self):
        """constitutes the core BBCON activity"""
        prod_count = 0
        for sensob in self.sensobs:         #Updates all sensobs
            sensob.update()
            if(prod_count == 2):
                image = Imager(False, sensob.get_value())
                print("Camera pixel", image.get_pixel(20, 30))
            prod_count += 1

        for behavior in self.behaviors:     #Update all behaviors
            behavior.update(self.sensobs)

        fav_behavior = self.arbitrator.choose_action(self.active_behaviors)
        self.motob.update(fav_behavior.sense_and_act())

def main():
    ZumoButton().wait_for_press()
    bbcon = BBCON()
    i = 0
    while i < 15:
            bbcon.run_one_timestep()
            i+=1

if __name__ == "__main__":
    main()
