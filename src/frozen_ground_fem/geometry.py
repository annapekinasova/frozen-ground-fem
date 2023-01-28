import numpy as np

from frozen_ground_fem.materials import (
    Material,
    NULL_MATERIAL,
    thrm_cond_ice as lam_i,
    thrm_cond_water as lam_w,
    vol_heat_cap_ice as C_i,
    vol_heat_cap_water as C_w,
    )


def shape_matrix(s):
    s = float(s)
    return np.array([[(1. - s), s]])


def gradient_matrix(s, dz):
    s = float(s)
    dz = float(dz)
    return np.array([[-1., 1.]]) / dz


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

    def __init__(self, index, coord=0., temp=0.):
        super().__init__(coord)
        self._temp = np.zeros((1,))
        self.temp = temp
        self._index = None
        self.index = index

    @property
    def temp(self):
        return self._temp[0]
    
    @temp.setter
    def temp(self, value):
        self._temp[0] = value

    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self, value):
        self._index = int(value)

    def __str__(self):
        return super().__str__() + f", temp={self.temp}"
    

class IntegrationPoint1D(Point1D):

    def __init__(self, coord=0., porosity=0., vol_ice_cont=0.,
                 material=NULL_MATERIAL):
        super().__init__(coord)
        self._porosity = np.zeros((1,))
        self._vol_ice_cont = np.zeros((1,))
        self.porosity = porosity
        self.vol_ice_cont = vol_ice_cont
        self.material = material

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

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, value):
        if not isinstance(value, Material):
            raise TypeError(f'{value} is not a Material object')
        self._material = value

    @property
    def thrm_cond(self):
        lam_s = self.material.thrm_cond_solids
        por = self.porosity
        th_i = self.vol_ice_cont
        th_w = por - th_i
        return (lam_s ** (1 - por)) * (lam_i ** th_i) * (lam_w ** th_w)

    @property
    def vol_heat_cap(self):
        C_s = self.material.vol_heat_cap_solids
        por = self.porosity
        th_i = self.vol_ice_cont
        th_w = por - th_i
        return ((1 - por) * C_s) + (th_i * C_i) + (th_w * C_w)

    def __str__(self):
        return (super().__str__()
                + f", porosity={self.porosity}"
                + f", vol_ice_cont={self.vol_ice_cont}")
    

class Element1D:

    def __init__(self, nodes):
        # check for valid node list and assign to self
        if (nnod := len(nodes)) != 2:
            raise ValueError(f'len(nodes) is {nnod} not equal to 2')
        for nd in nodes:
            if not isinstance(nd, Node1D):
                raise TypeError('nodes contains invalid objects, not Node1D')
        self._nodes = tuple(nodes)

    @property
    def nodes(self):
        return self._nodes