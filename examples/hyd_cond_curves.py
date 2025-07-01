import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.materials import (
    Material,
)


def main():
    e_plot = np.linspace(0.3, 2.0, 35)

    k_unfrozen_exp = np.array(
        [
            1.17e-11,
            1.70e-11,
            2.48e-11,
            3.62e-11,
            5.27e-11,
            7.69e-11,
            1.12e-10,
            1.64e-10,
            2.39e-10,
            3.48e-10,
            5.08e-10,
            7.41e-10,
            1.08e-09,
            1.58e-09,
            2.30e-09,
            3.35e-09,
            4.89e-09,
            7.13e-09,
            1.04e-08,
            1.52e-08,
            2.21e-08,
            3.23e-08,
            4.71e-08,
            6.87e-08,
            1.00e-07,
            1.46e-07,
            2.13e-07,
            3.11e-07,
            4.54e-07,
            6.62e-07,
            9.65e-07,
            1.41e-06,
            2.05e-06,
            2.99e-06,
            4.37e-06,
        ]
    )

    dkde_unfrozen_exp = np.array(
        [
            8.80e-11,
            1.28e-10,
            1.87e-10,
            2.73e-10,
            3.98e-10,
            5.81e-10,
            8.47e-10,
            1.24e-09,
            1.80e-09,
            2.63e-09,
            3.83e-09,
            5.59e-09,
            8.16e-09,
            1.19e-08,
            1.74e-08,
            2.53e-08,
            3.69e-08,
            5.39e-08,
            7.86e-08,
            1.15e-07,
            1.67e-07,
            2.44e-07,
            3.56e-07,
            5.19e-07,
            7.57e-07,
            1.10e-06,
            1.61e-06,
            2.35e-06,
            3.42e-06,
            4.99e-06,
            7.28e-06,
            1.06e-05,
            1.55e-05,
            2.26e-05,
            3.30e-05,
        ]
    )

    k_thawed_exp = np.array(
        [
            1.17e-10,
            1.70e-10,
            2.48e-10,
            3.62e-10,
            5.27e-10,
            7.69e-10,
            1.12e-09,
            1.64e-09,
            2.39e-09,
            3.48e-09,
            5.08e-09,
            7.41e-09,
            1.08e-08,
            1.58e-08,
            2.30e-08,
            3.35e-08,
            4.89e-08,
            7.13e-08,
            1.04e-07,
            1.52e-07,
            2.21e-07,
            3.23e-07,
            4.71e-07,
            6.87e-07,
            1.00e-06,
            1.46e-06,
            2.13e-06,
            3.11e-07,
            4.54e-07,
            6.62e-07,
            9.65e-07,
            1.41e-06,
            2.05e-06,
            2.99e-06,
            4.37e-06,
        ]
    )

    dkde_thawed_exp = np.array(
        [
            8.80e-10,
            1.28e-09,
            1.87e-09,
            2.73e-09,
            3.98e-09,
            5.81e-09,
            8.47e-09,
            1.24e-08,
            1.80e-08,
            2.63e-08,
            3.83e-08,
            5.59e-08,
            8.16e-08,
            1.19e-07,
            1.74e-07,
            2.53e-07,
            3.69e-07,
            5.39e-07,
            7.86e-07,
            1.15e-06,
            1.67e-06,
            2.44e-06,
            3.56e-06,
            5.19e-06,
            7.57e-06,
            1.10e-05,
            1.61e-05,
            2.35e-06,
            3.42e-06,
            4.99e-06,
            7.28e-06,
            1.06e-05,
            1.55e-05,
            2.26e-05,
            3.30e-05,
        ]
    )

    m = Material(
        hyd_cond_0=4.05e-4,
        void_ratio_0_hyd_cond=2.6,
        hyd_cond_mult=10.0,
        hyd_cond_index=0.305,
        void_ratio_min=0.3,
        void_ratio_tr=1.6,
        void_ratio_sep=3.0,
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

    plt.savefig("examples/hyd_cond_curves.svg")


if __name__ == "__main__":
    main()
