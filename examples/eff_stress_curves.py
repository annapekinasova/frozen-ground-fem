import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.materials import (
    Material,
)


def main():
    e_ncl = np.array(
        [
            2.60,
            2.50,
            2.40,
            2.30,
            2.20,
            2.10,
            2.00,
            1.90,
            1.80,
            1.70,
            1.60,
            1.50,
            1.40,
            1.30,
            1.20,
            1.10,
            1.00,
            0.90,
            0.80,
            0.70,
            0.60,
            0.50,
            0.40,
            0.30,
            0.20,
        ]
    )

    sig_p_ncl = np.array(
        [
            2.80e-03,
            4.84e-03,
            8.36e-03,
            1.44e-02,
            2.50e-02,
            4.31e-02,
            7.45e-02,
            1.29e-01,
            2.23e-01,
            3.85e-01,
            6.64e-01,
            1.15e00,
            1.98e00,
            3.43e00,
            5.92e00,
            1.02e01,
            1.77e01,
            3.06e01,
            5.28e01,
            9.12e01,
            1.58e02,
            2.72e02,
            4.71e02,
            8.13e02,
            1.41e03,
        ]
    )

    e_exp = np.array(
        [
            2.60,
            2.50,
            2.40,
            2.30,
            2.20,
            2.10,
            2.00,
            1.90,
            1.80,
            1.70,
            1.60,
            1.50,
            1.40,
            1.30,
            1.20,
            1.10,
            1.00,
            1.01,
            1.02,
            1.03,
            1.04,
            1.05,
            1.06,
            1.07,
            1.08,
            1.09,
            1.10,
            1.11,
            1.10,
            1.09,
            1.08,
            1.07,
            1.06,
            1.05,
            1.04,
            1.03,
            1.02,
            1.01,
            1.00,
            0.90,
            0.80,
            0.70,
            0.60,
            0.50,
            0.40,
            0.41,
            0.42,
            0.43,
            0.44,
            0.45,
            0.46,
            0.47,
            0.48,
            0.49,
            0.50,
            0.51,
            0.52,
            0.51,
            0.50,
            0.49,
            0.48,
            0.47,
            0.46,
            0.45,
            0.44,
            0.43,
            0.42,
            0.41,
            0.40,
            0.30,
            0.20,
        ]
    )

    sig_p_exp = np.array(
        [
            2.80e-03,
            4.84e-03,
            8.36e-03,
            1.44e-02,
            2.50e-02,
            4.31e-02,
            7.45e-02,
            1.29e-01,
            2.23e-01,
            3.85e-01,
            6.64e-01,
            1.15e00,
            1.98e00,
            3.43e00,
            5.92e00,
            1.02e01,
            1.77e01,
            1.33e01,
            9.95e00,
            7.46e00,
            5.59e00,
            4.19e00,
            3.15e00,
            2.36e00,
            1.77e00,
            1.33e00,
            9.95e-01,
            7.46e-01,
            9.95e-01,
            1.33e00,
            1.77e00,
            2.36e00,
            3.15e00,
            4.19e00,
            5.59e00,
            7.46e00,
            9.95e00,
            1.33e01,
            1.77e01,
            3.06e01,
            5.28e01,
            9.12e01,
            1.58e02,
            2.72e02,
            4.71e02,
            3.53e02,
            2.65e02,
            1.99e02,
            1.49e02,
            1.12e02,
            8.37e01,
            6.28e01,
            4.71e01,
            3.53e01,
            2.65e01,
            1.99e01,
            1.49e01,
            1.99e01,
            2.65e01,
            3.53e01,
            4.71e01,
            6.28e01,
            8.37e01,
            1.12e02,
            1.49e02,
            1.99e02,
            2.65e02,
            3.53e02,
            4.71e02,
            8.13e02,
            1.41e03,
        ]
    )

    dsig_de_exp = np.array(
        [
            -1.53e-02,
            -2.65e-02,
            -4.57e-02,
            -7.90e-02,
            -1.37e-01,
            -2.36e-01,
            -4.08e-01,
            -7.04e-01,
            -1.22e00,
            -2.10e00,
            -3.63e00,
            -6.28e00,
            -1.09e01,
            -1.87e01,
            -3.24e01,
            -5.60e01,
            -9.67e01,
            -3.82e02,
            -2.86e02,
            -2.15e02,
            -1.61e02,
            -1.21e02,
            -9.05e01,
            -6.79e01,
            -5.09e01,
            -3.82e01,
            -2.86e01,
            -2.15e01,
            -2.86e01,
            -3.82e01,
            -5.09e01,
            -6.79e01,
            -9.05e01,
            -1.21e02,
            -1.61e02,
            -2.15e02,
            -2.86e02,
            -3.82e02,
            -9.67e01,
            -1.67e02,
            -2.89e02,
            -4.99e02,
            -8.62e02,
            -1.49e03,
            -2.57e03,
            -1.02e04,
            -7.62e03,
            -5.71e03,
            -4.28e03,
            -3.21e03,
            -2.41e03,
            -1.81e03,
            -1.35e03,
            -1.02e03,
            -7.62e02,
            -5.71e02,
            -4.28e02,
            -5.71e02,
            -7.62e02,
            -1.02e03,
            -1.35e03,
            -1.81e03,
            -2.41e03,
            -3.21e03,
            -4.28e03,
            -5.71e03,
            -7.62e03,
            -1.02e04,
            -2.57e03,
            -4.45e03,
            -7.69e03,
        ]
    )

    # define material properties
    # Note: eff_stress_0_comp in Pa here
    m = Material(
        comp_index_unfrozen=0.421,
        rebound_index_unfrozen=0.08,
        void_ratio_sep=3.00,
        void_ratio_0_comp=2.60,
        eff_stress_0_comp=2.80,
    )

    e = m.void_ratio_0_comp
    ppc = m.eff_stress_0_comp
    de = -0.01
    e_lim = [1.0, 1.11, 0.4, 0.52, 0.2]
    j_lim = 0
    n_lim = len(e_lim)
    e_act = []
    dsig_de_act = []
    sig_p_act = []
    while j_lim < n_lim:
        sig_p, dsig_de = m.eff_stress(e, ppc)
        if sig_p > ppc:
            ppc = sig_p
        e_act.append(e)
        sig_p_act.append(sig_p)
        dsig_de_act.append(dsig_de)
        e += de
        if np.abs(e - e_lim[j_lim]) < np.abs(de):
            de *= -1
            j_lim += 1

    # store results as numpy arrays
    # convert stresses to kPa
    e_act = np.array(e_act)
    sig_p_act = np.array(sig_p_act) * 1.0e-03
    dsig_de_act = np.array(dsig_de_act) * 1.0e-03

    plt.rc("font", size=8)

    plt.figure(figsize=(5, 4))
    plt.semilogx(sig_p_ncl, e_ncl, "-b", label="NCL")
    plt.semilogx(sig_p_exp, e_exp, "--r", label="expected")
    plt.semilogx(sig_p_act, e_act, ":k", label="actual")
    plt.legend()
    plt.ylabel("void ratio, e")
    plt.xlabel("eff stress, sig' [kPa]")
    plt.title("void ratio - effective stress curve")
    plt.savefig("examples/eff_stress_curves_semilog.svg")

    plt.figure(figsize=(5, 4))
    plt.semilogx(-dsig_de_exp, e_exp, "--r", label="expected")
    plt.semilogx(-dsig_de_act, e_act, ":k", label="actual")
    plt.legend()
    plt.ylabel("void ratio, e")
    plt.xlabel("eff stress gradient, |dsig'/de| [kPa]")
    plt.savefig("examples/eff_stress_curves_gradient.svg")

    plt.figure(figsize=(5, 4))
    plt.plot(sig_p_ncl, e_ncl, "-b", label="NCL")
    plt.plot(sig_p_exp, e_exp, "--r", label="expected")
    plt.plot(sig_p_act, e_act, ":k", label="actual")
    plt.xlim((0, 30))
    plt.ylim((0.9, 1.2))
    plt.legend()
    plt.ylabel("void ratio, e")
    plt.xlabel("eff stress, sig' [kPa]")
    plt.title("unloading curve 1")
    plt.savefig("examples/eff_stress_curves_url1.svg")

    plt.figure(figsize=(5, 4))
    plt.plot(sig_p_ncl, e_ncl, "-b", label="NCL")
    plt.plot(sig_p_exp, e_exp, "--r", label="expected")
    plt.plot(sig_p_act, e_act, ":k", label="actual")
    plt.xlim((0, 800))
    plt.ylim((0.3, 0.6))
    plt.legend()
    plt.ylabel("void ratio, e")
    plt.xlabel("eff stress, sig' [kPa]")
    plt.title("unloading curve 2")
    plt.savefig("examples/eff_stress_curves_url2.svg")


if __name__ == "__main__":
    main()
