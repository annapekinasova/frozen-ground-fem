import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.materials import (
    Material,
)


def main():
    e_plot = np.linspace(0.3, 2.0, 35)

    k_unfrozen_exp = np.array(
        [
            1.17E-11,
            1.70E-11,
            2.48E-11,
            3.62E-11,
            5.27E-11,
            7.69E-11,
            1.12E-10,
            1.64E-10,
            2.39E-10,
            3.48E-10,
            5.08E-10,
            7.41E-10,
            1.08E-09,
            1.58E-09,
            2.30E-09,
            3.35E-09,
            4.89E-09,
            7.13E-09,
            1.04E-08,
            1.52E-08,
            2.21E-08,
            3.23E-08,
            4.71E-08,
            6.87E-08,
            1.00E-07,
            1.46E-07,
            2.13E-07,
            3.11E-07,
            4.54E-07,
            6.62E-07,
            9.65E-07,
            1.41E-06,
            2.05E-06,
            2.99E-06,
            4.37E-06,
        ]
    )

    dkde_unfrozen_exp = np.array(
        [
            8.80E-11,
            1.28E-10,
            1.87E-10,
            2.73E-10,
            3.98E-10,
            5.81E-10,
            8.47E-10,
            1.24E-09,
            1.80E-09,
            2.63E-09,
            3.83E-09,
            5.59E-09,
            8.16E-09,
            1.19E-08,
            1.74E-08,
            2.53E-08,
            3.69E-08,
            5.39E-08,
            7.86E-08,
            1.15E-07,
            1.67E-07,
            2.44E-07,
            3.56E-07,
            5.19E-07,
            7.57E-07,
            1.10E-06,
            1.61E-06,
            2.35E-06,
            3.42E-06,
            4.99E-06,
            7.28E-06,
            1.06E-05,
            1.55E-05,
            2.26E-05,
            3.30E-05,
        ]
    )

    k_thawed_exp = np.array(
        [
            1.17E-10,
            1.70E-10,
            2.48E-10,
            3.62E-10,
            5.27E-10,
            7.69E-10,
            1.12E-09,
            1.64E-09,
            2.39E-09,
            3.48E-09,
            5.08E-09,
            7.41E-09,
            1.08E-08,
            1.58E-08,
            2.30E-08,
            3.35E-08,
            4.89E-08,
            7.13E-08,
            1.04E-07,
            1.52E-07,
            2.21E-07,
            3.23E-07,
            4.71E-07,
            6.87E-07,
            1.00E-06,
            1.46E-06,
            2.13E-06,
            3.11E-07,
            4.54E-07,
            6.62E-07,
            9.65E-07,
            1.41E-06,
            2.05E-06,
            2.99E-06,
            4.37E-06,
        ]
    )

    dkde_thawed_exp = np.array(
        [
            8.80E-10,
            1.28E-09,
            1.87E-09,
            2.73E-09,
            3.98E-09,
            5.81E-09,
            8.47E-09,
            1.24E-08,
            1.80E-08,
            2.63E-08,
            3.83E-08,
            5.59E-08,
            8.16E-08,
            1.19E-07,
            1.74E-07,
            2.53E-07,
            3.69E-07,
            5.39E-07,
            7.86E-07,
            1.15E-06,
            1.67E-06,
            2.44E-06,
            3.56E-06,
            5.19E-06,
            7.57E-06,
            1.10E-05,
            1.61E-05,
            2.35E-06,
            3.42E-06,
            4.99E-06,
            7.28E-06,
            1.06E-05,
            1.55E-05,
            2.26E-05,
            3.30E-05,
        ]
    )

    m = Material(
        hyd_cond_0=4.05e-4,
        void_ratio_0_hyd_cond=2.6,
        hyd_cond_mult=10.0,
        hyd_cond_index=0.305,
        void_ratio_min=0.3,
        void_ratio_tr=1.6,
    )
    k_unfrozen_act = np.zeros_like(e_plot)
    dkde_unfrozen_act = np.zeros_like(e_plot)
    k_thawed_act = np.zeros_like(e_plot)
    dkde_thawed_act = np.zeros_like(e_plot)
    for j, e in enumerate(e_plot):
        k_u, dkde_u = m.hyd_cond(e, 1.0, False)
        k_t, dkde_t = m.hyd_cond(e, 1.0, True)
        k_unfrozen_act[j] = k_u
        dkde_unfrozen_act[j] = dkde_u
        k_thawed_act[j] = k_t
        dkde_thawed_act[j] = dkde_t

    plt.figure(figsize=(6, 8))

    plt.subplot(2, 2, 1)
    plt.plot(k_unfrozen_act, e_plot, "-k", label="actual")
    plt.plot(k_unfrozen_exp, e_plot, "--r", label="expected")
    plt.plot(k_thawed_act, e_plot, "-k")
    plt.plot(k_thawed_exp, e_plot, "--r")
    plt.ylim((0.0, 2.5))
    plt.legend()
    plt.ylabel("void ratio, e")
    plt.xlabel("hyd cond, k [m/s]")

    plt.subplot(2, 2, 3)
    plt.plot(dkde_unfrozen_act, e_plot, "-k", label="actual")
    plt.plot(dkde_unfrozen_exp, e_plot, "--r", label="expected")
    plt.plot(dkde_thawed_act, e_plot, "-k")
    plt.plot(dkde_thawed_exp, e_plot, "--r")
    plt.ylim((0.0, 2.5))
    plt.legend()
    plt.ylabel("void ratio, e")
    plt.xlabel("hyd cond grad, dk/de [m/s]")

    plt.subplot(2, 2, 2)
    plt.semilogx(k_unfrozen_act, e_plot, "-k", label="actual")
    plt.semilogx(k_unfrozen_exp, e_plot, "--r", label="expected")
    plt.semilogx(k_thawed_act, e_plot, "-k")
    plt.semilogx(k_thawed_exp, e_plot, "--r")
    plt.ylim((0.0, 2.5))
    plt.legend()
    plt.xlabel("hyd cond, k [m/s]")

    plt.subplot(2, 2, 4)
    plt.semilogx(dkde_unfrozen_act, e_plot, "-k", label="actual")
    plt.semilogx(dkde_unfrozen_exp, e_plot, "--r", label="expected")
    plt.semilogx(dkde_thawed_act, e_plot, "-k")
    plt.semilogx(dkde_thawed_exp, e_plot, "--r")
    plt.ylim((0.0, 2.5))
    plt.legend()
    plt.xlabel("hyd cond grad, dk/de [m/s]")

    plt.savefig("examples/hyd_cond_curves.png")


if __name__ == "__main__":
    main()
