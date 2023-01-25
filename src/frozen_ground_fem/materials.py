vol_heat_cap_water = 4.204e6  # J.m^{-3}.K^{-1}
vol_heat_cap_ice = 1.881e6  # J.m^{-3}.K^{-1}
thrm_cond_water = 0.563  # W.m^{-1}.K^{-1}
thrm_cond_ice = 2.22  # W.m^{-1}.K^{-1}


class Material:
    
    def __init__(self, 
                 thrm_cond_solids=0.,
                 dens_solids=0.,
                 spec_heat_cap_solids=0.):
        self._thrm_cond_solids = 0.
        self._dens_solids = 0.
        self._spec_heat_cap_solids = 0.
        self.thrm_cond_solids = thrm_cond_solids
        self.dens_solids = dens_solids
        self.spec_heat_cap_solids = spec_heat_cap_solids

    @property
    def thrm_cond_solids(self):
        return self._thrm_cond_solids

    @thrm_cond_solids.setter
    def thrm_cond_solids(self, value):
        value = float(value)
        if value < 0.:
            raise ValueError(f'thrm_cond_solids {value} is not positive')
        self._thrm_cond_solids = value
        
    @property
    def dens_solids(self):
        return self._dens_solids

    @dens_solids.setter
    def dens_solids(self, value):
        value = float(value)
        if value < 0.:
            raise ValueError(f'dens_solids {value} is not positive')
        self._dens_solids = value
        self._update_vol_heat_cap_solids()
        
    @property
    def spec_heat_cap_solids(self):
        return self._spec_heat_cap_solids

    @spec_heat_cap_solids.setter
    def spec_heat_cap_solids(self, value):
        value = float(value)
        if value < 0.:
            raise ValueError(f'spec_heat_cap_solids {value} is not positive')
        self._spec_heat_cap_solids = value
        self._update_vol_heat_cap_solids()
        
    @property
    def vol_heat_cap_solids(self):
        return self._vol_heat_cap_solids
                                     
    def _update_vol_heat_cap_solids(self):
        self._vol_heat_cap_solids = (self.dens_solids
                                     * self.spec_heat_cap_solids)