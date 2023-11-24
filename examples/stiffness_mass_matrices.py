import numpy as np

from frozen_ground_fem import (
    Material,
    Node1D,
    Element1D,
)
from frozen_ground_fem.materials import (
    unit_weight_water as gam_w,
)
from frozen_ground_fem.consolidation import (
    ConsolidationElement1D,
)


def main():
    m = Material(
        spec_grav_solids=2.60,
        hyd_cond_index=0.305,
        hyd_cond_0=4.05e-4,
        void_ratio_0_hyd_cond=2.60,
        void_ratio_min=0.30,
        void_ratio_tr=2.60,
        void_ratio_0_comp=2.60,
        comp_index_unfrozen=0.421,
        rebound_index_unfrozen=0.08,
        eff_stress_0_comp=2.80e00,
    )

    nd_lin = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(2))
    el_lin = ConsolidationElement1D(Element1D(nd_lin, order=1))
    for ip in el_lin.int_pts:
        ip.material = m
        ip.void_ratio_0 = 0.9
        ip.void_ratio = 0.3
        ip.deg_sat_water = 0.1
        ip.pre_consol_stress = 1.0
        k, dk_de = m.hyd_cond(ip.void_ratio, 1.0, False)
        ip.hyd_cond = k
        ip.hyd_cond_gradient = dk_de
        sig, dsig_de = m.eff_stress(ip.void_ratio, ip.pre_consol_stress)
        ip.eff_stress = sig
        ip.eff_stress_gradient = dsig_de
    jac = el_lin.jacobian
    Gs = el_lin.int_pts[0].material.spec_grav_solids
    e0 = el_lin.int_pts[0].void_ratio_0
    e = el_lin.int_pts[0].void_ratio
    ppc = el_lin.int_pts[0].pre_consol_stress
    k, dk_de = el_lin.int_pts[0].material.hyd_cond(e, 1.0, False)
    sig_p, dsig_de = el_lin.int_pts[0].material.eff_stress(e, ppc)
    e_ratio = (1.0 + e0) / (1.0 + e)
    coef_0 = k * e_ratio * dsig_de / gam_w / jac
    coef_1 = (dk_de * (Gs - 1.0) / (1.0 + e)
              - k * (Gs - 1.0) / (1.0 + e) ** 2)
    stiff_0 = coef_0 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    stiff_1 = coef_1 * np.array([[-0.5, 0.5], [-0.5, 0.5]])
    expected = stiff_0 + stiff_1
    print("linear:")
    print(f"expected:\n{expected}")
    print(f"actual:\n{el_lin.stiffness_matrix}")

    nd_cub = tuple(Node1D(k, 2.0 * k + 1.0) for k in range(4))
    el_cub = ConsolidationElement1D(Element1D(nd_cub, order=3))
    for ip in el_cub.int_pts:
        ip.material = m
        ip.void_ratio_0 = 0.9
        ip.void_ratio = 0.3
        ip.deg_sat_water = 0.1
        ip.pre_consol_stress = 1.0
        k, dk_de = m.hyd_cond(ip.void_ratio, 1.0, False)
        ip.hyd_cond = k
        ip.hyd_cond_gradient = dk_de
        sig, dsig_de = m.eff_stress(ip.void_ratio, ip.pre_consol_stress)
        ip.eff_stress = sig
        ip.eff_stress_gradient = dsig_de
    jac = el_cub.jacobian
    Gs = el_cub.int_pts[0].material.spec_grav_solids
    e0 = el_cub.int_pts[0].void_ratio_0
    e = el_cub.int_pts[0].void_ratio
    ppc = el_cub.int_pts[0].pre_consol_stress
    k, dk_de = el_cub.int_pts[0].material.hyd_cond(e, 1.0, False)
    sig_p, dsig_de = el_cub.int_pts[0].material.eff_stress(e, ppc)
    e_ratio = (1.0 + e0) / (1.0 + e)
    coef_0 = k * e_ratio * dsig_de / gam_w / jac
    coef_1 = (dk_de * (Gs - 1.0) / (1.0 + e)
              - k * (Gs - 1.0) / (1.0 + e) ** 2)
    stiff_0 = coef_0 / 40.0 * np.array([[148, -189, 54, -13],
                                        [-189, 432, -297, 54],
                                        [54, -297, 432, -189],
                                        [-13, 54, -189, 148]])
    stiff_1 = coef_1 / 1680.0 * np.array([[-840, 1197, -504, 147],
                                          [-1197, 0, 1701, -504],
                                          [504, -1701, 0, 1197],
                                          [-147, 504, -1197, 840]])
    expected = stiff_0 + stiff_1
    print("cubic:")
    print(f"expected:\n{expected}")
    print(f"actual:\n{el_cub.stiffness_matrix}")


if __name__ == "__main__":
    main()
