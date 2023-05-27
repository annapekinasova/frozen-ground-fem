"""frozen_ground_fem.geometry.py - A module for classes
for finite element model geometry.

"""

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
    """Calculates the shape (interpolation) function matrix.

    Parameters
    ----------
    s : float
        The local coordinate. Should be between 0.0 and 1.0.

    Returns
    -------
    numpy.ndarray
        The shape function matrix.

    Raises
    ------
    ValueError
        If s cannot be converted to float.

    Notes
    -----
    Assumes linear interpolation of a single variable between two nodes.
    The resulting shape matrix N is:

        N = [[(1 - s), s]]
    """
    s = float(s)
    return np.array([[(1.0 - s), s]])


def gradient_matrix(s, dz):
    """Calculates the gradient of the shape (interpolation) function matrix.

    Parameters
    ----------
    s : float
        The local coordinate. Should be between 0.0 and 1.0.
    dz : float
        The element scale parameter (Jacobian).

    Returns
    -------
    numpy.ndarray
        The gradient of the shape function matrix.

    Raises
    ------
    ValueError
        If s cannot be converted to float.
        If dz cannot be converted to float.

    Notes
    -----
    Assumes linear interpolation of a single variable between two nodes.
    The resulting gradient matrix B is:

        B = [[-1 , 1]] / dz
    """
    s = float(s)
    dz = float(dz)
    return np.array([[-1.0, 1.0]]) / dz


class Point1D:
    """Class for storing the coordinates of a point.

    Attributes
    ----------
    coords
    z
    """

    def __init__(self, value=0.0):
        self._coords = np.zeros((1,))
        self.z = value

    @property
    def coords(self):
        """Coordinates of the point as an array.

        Returns
        -------
        (1, ) numpy.ndarray
        """
        return self._coords

    @property
    def z(self):
        """Coordinate of the point.

        Parameters
        ----------
        float or int or str
            The value to assign to the coordinate.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign cannot be converted to float.
        """
        return self.coords[0]

    @z.setter
    def z(self, value):
        self.coords[0] = value


class Node1D(Point1D):
    """Class for storing the properties of a node.
    Inherits from :c:`Point1D`.

    Attributes
    ----------
    index
    coords
    z
    temp
    void_ratio
    """

    def __init__(self, index, coord=0.0, temp=0.0, void_ratio=0.0):
        super().__init__(coord)
        self._temp = np.zeros((1,))
        self.temp = temp
        self._index = None
        self.index = index
        self._void_ratio = 0.0
        self.void_ratio = void_ratio

    @property
    def temp(self):
        """Temperature of the node.

        Parameters
        ----------
        float
            Value to assign to the temperature of the :c:`Node1D`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
        """
        return self._temp[0]

    @temp.setter
    def temp(self, value):
        self._temp[0] = value

    @property
    def index(self):
        """Index of the node.

        Parameters
        ----------
        int
            Value to assign to the index of the :c:`Node1D`.

        Returns
        -------
        int

        Raises
        ------
        TypeError
            If value to assign is a float.
        ValueError
            If value to assign is a str not convertible to int.
            If value to assign is negative.
        """
        return self._index

    @index.setter
    def index(self, value):
        if isinstance(value, float):
            raise TypeError(f"{value} is a float, must be int")
        _value = int(value)
        if _value < 0:
            raise ValueError(f"{_value} is negative")
        self._index = _value

    @property
    def void_ratio(self):
        """Void ratio of the node.

        Parameters
        ----------
        float
            Value to assign to the void ratio of the :c:`Node1D`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
        """
        return self._void_ratio

    @void_ratio.setter
    def void_ratio(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio {value} is not positive")
        self._void_ratio = value


class IntegrationPoint1D(Point1D):
    """Class for storing the properties of an integration point.
    Inherits from :c:`Point1D`.

    Attributes
    ----------
    coords
    z
    local_coord
    weight
    porosity
    vol_ice_cont
    material
    thrm_cond
    vol_heat_cap
    """

    def __init__(
        self,
        coord=0.0,
        local_coord=0.0,
        weight=0.0,
        void_ratio=0.0,
        deg_sat_water=1.0,
        material=NULL_MATERIAL,
        void_ratio_0=0.0,
        temp=0.0,
        temp_rate=0.0,
        temp_gradient=0.0,
    ):
        super().__init__(coord)
        self._local_coord = 0.0
        self._weight = 0.0
        self._void_ratio = 0.0
        self._porosity = 0.0
        self._void_ratio_0 = 0.0
        self._temp = 0.0
        self._temp_rate = 0.0
        self._temp_gradient = 0.0
        self._deg_sat_water = 1.0
        self._deg_sat_ice = 0.0
        self._vol_ice_cont = 0.0
        self.local_coord = local_coord
        self.weight = weight
        self.void_ratio = void_ratio
        self.void_ratio_0 = void_ratio_0
        self.temp = temp
        self.temp_rate = temp_rate
        self.temp_gradient = temp_gradient
        self.deg_sat_water = deg_sat_water
        self.material = material

    @property
    def local_coord(self):
        """Local coordinate of the integration point.

        Parameters
        ----------
        float
            Value to assign to the local coordinate of the
            :c:`IntegrationPoint1D`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
        """
        return self._local_coord

    @local_coord.setter
    def local_coord(self, value):
        value = float(value)
        self._local_coord = value

    @property
    def weight(self):
        """Quadrature weight of the integration point.

        Parameters
        ----------
        float
            Value to assign to the weight of the :c:`IntegrationPoint1D`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
        """
        return self._weight

    @weight.setter
    def weight(self, value):
        value = float(value)
        self._weight = value

    @property
    def void_ratio(self):
        """Void ratio of the integration point.

        Parameters
        ----------
        float
            Value to assign to the void ratio of the :c:`IntegrationPoint1D`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value to assign is negative.

        Notes
        -----
        Also updates porosity.
        """
        return self._void_ratio

    @void_ratio.setter
    def void_ratio(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio {value} is not positive")
        self._void_ratio = value
        self._porosity = value / (1.0 + value)
        self._vol_ice_cont = self.porosity * self.deg_sat_ice

    @property
    def void_ratio_0(self):
        """Initial void ratio of the integration point.

        Parameters
        ----------
        float
            Value to assign to the initial void ratio of the :c:`IntegrationPoint1D`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value to assign is negative.

        Notes
        -----
        """
        return self._void_ratio_0

    @void_ratio_0.setter
    def void_ratio_0(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"void_ratio_0 {value} is not positive")
        self._void_ratio_0 = value

    @property
    def temp(self):
        """Temperature at the integration point.

        Parameters
        ----------
        float
            Value to assign to the temperature of the :c:`IntegrationPoint1D`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.

        Notes
        -----
        """
        return self._temp

    @temp.setter
    def temp(self, value):
        self._temp = float(value)

    @property
    def temp_rate(self):
        """Temperature rate at the integration point.

        Parameters
        ----------
        float
            Value to assign to the temperature rate of the :c:`IntegrationPoint1D`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.

        Notes
        -----
        """
        return self._temp_rate

    @temp_rate.setter
    def temp_rate(self, value):
        self._temp_rate = float(value)

    @property
    def temp_gradient(self):
        """Temperature gradient at the integration point.

        Parameters
        ----------
        float
            Value to assign to the temperature gradient of the :c:`IntegrationPoint1D`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.

        Notes
        -----
        """
        return self._temp_gradient

    @temp_gradient.setter
    def temp_gradient(self, value):
        self._temp_gradient = float(value)

    @property
    def porosity(self):
        """Porosity of the integration point.

        Returns
        -------
        float

        Notes
        -----
        Porosity is not intended to be updated directly.
        It is updated each time void ratio is set.
        """
        return self._porosity

    @property
    def vol_ice_cont(self):
        """Volumetric ice content of the integration point.

        Returns
        -------
        float

        Notes
        ------
        Volumetric ice content is not intended to be set directly.
        It is updated when degree of saturation of water is updated.
        """
        return self._vol_ice_cont

    @property
    def deg_sat_water(self):
        """Degree of saturation of water of the integration point.

        Parameters
        ----------
        float
            Value to assign to the degree of saturation of water of the
            :c:`IntegrationPoint1D`.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If value to assign is not convertible to float.
            If value < 0.0 or value > 1.0

        Notes
        -----
        Also updates degree of saturation of ice (assuming full saturation)
        and volumetric ice content.
        """
        return self._deg_sat_water

    @deg_sat_water.setter
    def deg_sat_water(self, value):
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"deg_sat_water value {value} " + f"not between 0.0 and 1.0"
            )
        self._deg_sat_water = value
        self._deg_sat_ice = 1.0 - value
        self._vol_ice_cont = self.porosity * self._deg_sat_ice

    @property
    def deg_sat_ice(self):
        """Degree of saturation of ice of the integration point.

        Returns
        -------
        float

        Notes
        ------
        Degree of saturation of ice is not intended to be set directly.
        It is updated when degree of saturation of water is updated,
        assuming fully saturated conditions, i.e.
            deg_sat_water + deg_sat_ice = 1.0
        """
        return self._deg_sat_ice

    @property
    def material(self):
        """Contains the properties of the solids.

        Parameters
        ----------
        frozen_ground_fem.materials.Material

        Returns
        -------
        frozen_ground_fem.materials.Material

        Raises
        ------
        TypeError
            If value to assign is not an instance of
            :c:`frozen_ground_fem.materials.Material`.
        """
        return self._material

    @material.setter
    def material(self, value):
        if not isinstance(value, Material):
            raise TypeError(f"{value} is not a Material object")
        self._material = value

    @property
    def thrm_cond(self):
        """Contains the bulk thermal conductivity of the integration point.

        Returns
        ------
        float

        Notes
        -----
        Calculated according to the geometric mean formula [1]_::

            lam = (lam_s ** (1 - por)) * (lam_i ** th_i) * (lam_w ** th_w)

        References
        ----------
        .. [1] Côté, J. and Konrad, J.-M. 2005. A generalized thermal
           conductivity model for soils and construction materials. Canadian
           Geotechnical Journal 42(2): 443-458, doi: 10.1139/t04-106.
        """
        lam_s = self.material.thrm_cond_solids
        por = self.porosity
        th_i = self.vol_ice_cont
        th_w = por - th_i
        return (lam_s ** (1 - por)) * (lam_i**th_i) * (lam_w**th_w)

    @property
    def vol_heat_cap(self):
        """Contains the volumetric heat capacity of the integration point.

        Returns
        ------
        float

        Notes
        -----
        Calculated according to the volume averaging formula [1]_::

            C = (1 - por) * C_s + th_i * C_i + th_w * C_w

        References
        ----------
        .. [1] Andersland, O. and Ladanyi, B. 2004. Frozen Ground Engineering,
           2nd ed. Wiley: Hoboken, N.J.
        """
        C_s = self.material.vol_heat_cap_solids
        por = self.porosity
        th_i = self.vol_ice_cont
        th_w = por - th_i
        return ((1 - por) * C_s) + (th_i * C_i) + (th_w * C_w)


class Element1D:
    """Class for organizing element level information.

    Attributes
    ----------
    nodes
    int_pts
    jacobian

    Raises
    ------
    TypeError
        If nodes initializer contains non-:c:`Node1D` objects.
    ValueError
        If len(nodes) != 2.
    """

    _int_pt_coords = (0.211324865405187, 0.788675134594813)
    _int_pt_weights = (0.5, 0.5)

    def __init__(self, nodes):
        # check for valid node list and assign to self
        if (nnod := len(nodes)) != 2:
            raise ValueError(f"len(nodes) is {nnod} not equal to 2")
        for nd in nodes:
            if not isinstance(nd, Node1D):
                raise TypeError("nodes contains invalid objects, not Node1D")
        self._nodes = tuple(nodes)
        # initialize integration points
        self._int_pts = tuple(
            IntegrationPoint1D(local_coord=xi, weight=wt)
            for (xi, wt) in zip(Element1D._int_pt_coords, Element1D._int_pt_weights)
        )
        z_e = np.array([[nd.z for nd in self.nodes]]).T
        for ip in self.int_pts:
            N = shape_matrix(ip.local_coord)
            ip.z = float(N @ z_e)

    @property
    def nodes(self):
        """The tuple of :c:`Node1D` contained in the element.

        Returns
        ------
        tuple of :c:`Node1D`
        """
        return self._nodes

    @property
    def jacobian(self):
        """The length scale of the element (in Lagrangian coordinates).

        Returns
        -------
        float
        """
        return self.nodes[1].z - self.nodes[0].z

    @property
    def int_pts(self):
        """The tuple of :c:`IntegrationPoint1D` contained in the element.

        Returns
        ------
        tuple of :c:`IntegrationPoint1D`
        """
        return self._int_pts


class Boundary1D:
    """Class for storing boundary condition geometry information.

    Attributes
    ----------
    nodes

    Raises
    ------
    TypeError:
        If nodes initializer contains non-:c:`Node1D` objects.
    ValueError
        If len(nodes) != 1.
    """

    def __init__(self, nodes, int_pts=None):
        # check for valid node list and assign to self
        if (nnod := len(nodes)) != 1:
            raise ValueError(f"len(nodes) is {nnod} not equal to 1")
        for nd in nodes:
            if not isinstance(nd, Node1D):
                raise TypeError("nodes contains invalid objects, not Node1D")
        self._nodes = tuple(nodes)
        if int_pts is None:
            self._int_pts = None
        else:
            if len(int_pts) != 1:
                raise ValueError(f"len(int_pts) not equal to 1")
            for ip in int_pts:
                if not isinstance(ip, IntegrationPoint1D):
                    raise TypeError(
                        "int_pts contains invalid objects, not IntegrationPoint1D"
                    )
            self._int_pts = tuple(int_pts)

    @property
    def nodes(self):
        """The tuple of :c:`Node1D` contained in the element.

        Returns
        ------
        tuple of :c:`Node1D`
        """
        return self._nodes

    @property
    def int_pts(self):
        """The tuple of :c:`IntegrationPoint1D` contained in the element.

        Returns
        ------
        tuple of :c:`IntegrationPoint1D`
        """
        return self._int_pts


class Mesh1D:
    """Class for generating, storing, and organizing global geometry
    information about the analysis mesh.

    Attributes
    ----------
    nodes

    Raises
    ------
    """

    def __init__(self, z_range=None, grid_size=0.0, num_nodes=10, generate=False):
        self._boundaries = set()
        self.mesh_valid = False
        self._z_min = -np.inf
        self._z_max = np.inf
        if z_range is not None:
            self.z_min = np.min(z_range)
            self.z_max = np.max(z_range)
        self.grid_size = grid_size
        if generate:
            self.generate_mesh(num_nodes)

    @property
    def z_min(self):
        """The minimum z value of the mesh.

        Parameters
        ----------
        float
            Value to assign to z_min.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign cannot be cast to float.
            If the value to assign is >= z_max.
        """
        return self._z_min

    @z_min.setter
    def z_min(self, value):
        value = float(value)
        if value >= self.z_max:
            raise ValueError(f"{value} >= z_max := {self.z_max}")
        self._z_min = value
        self.mesh_valid = False

    @property
    def z_max(self):
        """The maximum z value of the mesh.

        Parameters
        ----------
        float
            Value to assign to z_max.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign cannot be cast to float.
            If the value to assign is <= z_min.
        """
        return self._z_max

    @z_max.setter
    def z_max(self, value):
        value = float(value)
        if value <= self.z_min:
            raise ValueError(f"{value} <= z_min := {self.z_min}")
        self._z_max = value
        self.mesh_valid = False

    @property
    def grid_size(self):
        """The specified grid size of the mesh.

        Parameters
        ----------
        float
            Value to assign to grid size.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If the value to assign cannot be cast to float.
            If the value to assign is < 0.0.

        Notes
        -----
        This parameter is not the actual size of any element of the mesh.
        It is a suggested target value that will be recalculated so that
        an integer number of nodes between z_min and z_max is achieved.
        The actual element size will typically be smaller.
        If grid_size is set to 0.0, its value is ignored
        and the element size is calculated based on a specified (or default)
        number of nodes.
        """
        return self._grid_size

    @grid_size.setter
    def grid_size(self, value):
        value = float(value)
        if value < 0.0:
            raise ValueError(f"{value} is negative")
        self._grid_size = value
        self.mesh_valid = False

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def nodes(self):
        """The tuple of :c:`Node1D` contained in the mesh.

        Returns
        ------
        tuple of :c:`Node1D`
        """
        return self._nodes

    @property
    def num_elements(self):
        return len(self.elements)

    @property
    def elements(self):
        """The tuple of :c:`Element1D` contained in the mesh.

        Returns
        ------
        tuple of :c:`Element1D`
        """
        return self._elements

    @property
    def num_boundaries(self):
        return len(self.boundaries)

    @property
    def boundaries(self):
        """The tuple of :c:`Boundary1D` contained in the mesh.

        Returns
        ------
        tuple of :c:`Boundary1D`
        """
        return self._boundaries

    @property
    def mesh_valid(self):
        """Flag for valid mesh.

        Parameters
        ----------
        bool

        Returns
        -------
        bool

        Raises
        ------
        ValueError
            If the value to assign cannot be cast to bool.

        Notes
        -----
        When assigning to False also clears mesh information
        (e.g. nodes, elements).
        """
        return self._mesh_valid

    @mesh_valid.setter
    def mesh_valid(self, value):
        value = bool(value)
        if value:
            # TODO: check for mesh validity
            self._mesh_valid = True
        else:
            self._nodes = ()
            self._elements = ()
            self.clear_boundaries()
            self._mesh_valid = False

    def generate_mesh(self, num_nodes=10):
        """Generates a mesh using assigned mesh properties.

        Parameters
        ----------
        num_nodes : int, optional, default=10
            Number of nodes to be created in the generated mesh

        Raises
        ------
        ValueError
            If z_min or z_max are invalid (e.g. left as default +/-inf)
            If grid_size is invalid (e.g. set to inf)

        Notes
        -----
        If the grid_size paramater is set,
        the argument num_nodes will be ignored
        and the number of nodes will be calculated
        as the nearest integer number of nodes:
            (z_max - z_min) // grid_size + 1
        """
        self.mesh_valid = False
        self._generate_nodes(num_nodes)
        self._generate_elements()
        self.mesh_valid = True

    def _generate_nodes(self, num_nodes=10):
        if np.isinf(self.z_min) or np.isinf(self.z_max):
            raise ValueError("cannot generate mesh, non-finite limits")
        if np.isinf(self.grid_size):
            raise ValueError("cannot generate mesh, non-finite grid size")
        if self.grid_size > 0.0:
            num_nodes = int((self.z_max - self.z_min) // self.grid_size) + 1
        z_nodes = np.linspace(self.z_min, self.z_max, num_nodes)
        self._nodes = tuple(Node1D(k, zk) for k, zk in enumerate(z_nodes))

    def _generate_elements(self):
        self._elements = tuple(
            Element1D((self.nodes[k], self.nodes[k + 1]))
            for k in range(self.num_nodes - 1)
        )

    def add_boundary(self, new_boundary: Boundary1D) -> None:
        if not isinstance(new_boundary, Boundary1D):
            raise TypeError(
                f"type(new_boundary) {type(new_boundary)} invalid, must be Boundary1D"
            )
        for nd in new_boundary.nodes:
            if nd not in self.nodes:
                raise ValueError(f"new_boundary contains node {nd} not in mesh")
        if new_boundary.int_pts is not None:
            int_pts = tuple(ip for e in self.elements for ip in e.int_pts)
            for ip in new_boundary.int_pts:
                if ip not in int_pts:
                    raise ValueError(f"new_boundary contains int_pt {ip} not in mesh")
        self._boundaries.add(new_boundary)

    def remove_boundary(self, boundary: Boundary1D) -> None:
        self._boundaries.remove(boundary)

    def clear_boundaries(self):
        self._boundaries.clear()
