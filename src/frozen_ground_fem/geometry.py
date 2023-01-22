import numpy as np


class Point1D:

    def __init__(self, value=0.):
        self._coords = np.zeros((1,))
        self.z = value

    @property
    def coords(self):
        return self._coords

    @property
    def z(self):
        return self.coords[0]
        
    @z.setter
    def z(self, value):
        self.coords[0] = value
        
    def __str__(self):
        return self.coords.__str__()