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
        

class IntegrationPoint1D(Point1D):

    def __init__(self, coord=0., porosity=0., vol_ice_cont=0.):
        super().__init__(coord)
        self._porosity = np.zeros((1,))
        self._vol_ice_cont = np.zeros((1,))
        self.porosity = porosity
        self.vol_ice_cont = vol_ice_cont

    @property
    def porosity(self):
        return self._porosity[0]

    @porosity.setter
    def porosity(self, value):
        value = float(value)
        if value < 0. or value > 1.:
            raise ValueError(f"porosity value {value} not between 0.0 and 1.0")
        self._porosity[0] = value
        
    @property
    def vol_ice_cont(self):
        return self._vol_ice_cont[0]

    @vol_ice_cont.setter
    def vol_ice_cont(self, value):
        value = float(value)
        if value < 0. or value > self.porosity:
            raise ValueError(f"vol_ice_cont value {value} "
                             + f"not between 0.0 and porosity={self.porosity}")
        self._vol_ice_cont[0] = value

    def __str__(self):
        return (super().__str__()
                + f", porosity={self.porosity}"
                + f", vol_ice_cont={self.vol_ice_cont}")
