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
        
        
class Node1D(Point1D):

    def __init__(self, coord=0., temp=0.):
        super().__init__(coord)
        self._temp = np.zeros((1,))
        self.temp = temp

    @property
    def temp(self):
        return self._temp[0]

    @temp.setter
    def temp(self, value):
        self._temp[0] = value

    def __str__(self):
        return super().__str__() + f", temp={self.temp}"
