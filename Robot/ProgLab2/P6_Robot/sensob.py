"""Sensob class"""



class Sensob:
    """Sensor object class"""

    sensor = None

    def __init__(self, sensor):
        self.sensor = sensor  # Sensor(s)

    def update(self):
        """force the sensob to fetch the relevant sensor value(s) and convert them into the"""
        self.value = self.sensor.update()

    def get_value(self):
        """ Get value from sensor wrapper """
        return self.sensor.get_value()
