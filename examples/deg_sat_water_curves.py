import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.materials import (
    Material,
)


def main():
    T_plot = np.hstack(
        [
            np.linspace(-40, -1, 16),
            np.linspace(-0.9, -0.1, 9),
            np.linspace(-0.09, -0.01, 9),
            np.linspace(-0.009, -0.001, 9),
            np.linspace(-0.0009, -0.0001, 9),
        ]
    )

    Sw_exp = np.array(
        [
            0.0115,
            0.0124,
            0.0136,
            0.0150,
            0.0170,
            0.0200,
            0.0250,
            0.0265,
            0.0283,
            0.0304,
            0.0331,
            0.0365,
            0.0412,
            0.0482,
            0.0600,
            0.0872,
            0.0923,
            0.0983,
            0.1057,
            0.1148,
            0.1266,
            0.1427,
            0.1666,
            0.2068,
            0.2983,
            0.3151,
            0.3348,
            0.3586,
            0.3877,
            0.4245,
            0.4729,
            0.5400,
            0.6399,
            0.8005,
            0.8212,
            0.8428,
            0.8650,
            0.8878,
            0.9108,
            0.9336,
            0.9554,
            0.9751,
            0.9912,
            0.9925,
            0.9937,
            0.9949,
            0.9959,
            0.9969,
            0.9978,
            0.9986,
            0.9992,
            0.9997,
        ]
    )

    dSwdT_exp = np.array(
        [
            1.6752e-04,
            2.0464e-04,
            2.5807e-04,
            3.3994e-04,
            4.7687e-04,
            7.3888e-04,
            1.3725e-03,
            1.6125e-03,
            1.9312e-03,
            2.3695e-03,
            3.0009e-03,
            3.9690e-03,
            5.5896e-03,
            8.6932e-03,
            1.6204e-02,
            4.6992e-02,
            5.5244e-02,
            6.6193e-02,
            8.1249e-02,
            1.0293e-01,
            1.3612e-01,
            1.9156e-01,
            2.9733e-01,
            5.5091e-01,
            1.5557e00,
            1.8158e00,
            2.1551e00,
            2.6112e00,
            3.2473e00,
            4.1764e00,
            5.6169e00,
            8.0260e00,
            1.2417e01,
            2.0280e01,
            2.1146e01,
            2.1931e01,
            2.2569e01,
            2.2963e01,
            2.2975e01,
            2.2408e01,
            2.0977e01,
            1.8247e01,
            1.3380e01,
            1.2706e01,
            1.1983e01,
            1.1202e01,
            1.0353e01,
            9.4206e00,
            8.3829e00,
            7.2016e00,
            5.8037e00,
            4.0035e00,
        ]
    )

    m = Material(
        deg_sat_water_alpha=1.2e4,
        deg_sat_water_beta=0.35,
    )
    Sw_act = np.zeros_like(T_plot)
    dSwdT_act = np.zeros_like(T_plot)
    for k, temp in enumerate(T_plot):
        Sw_dSw = m.deg_sat_water(temp)
        Sw_act[k] = Sw_dSw[0]
        dSwdT_act[k] = Sw_dSw[1]

    plt.figure(figsize=(6, 6))

    plt.subplot(2, 2, 1)
    plt.plot(T_plot, Sw_act, "-k", label="actual")
    plt.plot(T_plot, Sw_exp, "--r", label="expected")
    plt.xlim((-20, 0))
    plt.legend()
    plt.ylabel("Sw")

    plt.subplot(2, 2, 3)
    plt.plot(T_plot, dSwdT_act, "-k", label="actual")
    plt.plot(T_plot, dSwdT_exp, "--r", label="expected")
    plt.xlim((-20, 0))
    plt.legend()
    plt.ylabel("dSw/dT")
    plt.xlabel("Temperature, T [deg C]")

    plt.subplot(2, 2, 2)
    plt.plot(T_plot, Sw_act, "-k", label="actual")
    plt.plot(T_plot, Sw_exp, "--r", label="expected")
    plt.xlim((-1, 0))
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(T_plot, dSwdT_act, "-k", label="actual")
    plt.plot(T_plot, dSwdT_exp, "--r", label="expected")
    plt.xlim((-1, 0))
    plt.legend()
    plt.xlabel("Temperature, T [deg C]")

    plt.savefig("examples/deg_sat_water_curves.png")


if __name__ == "__main__":
    main()
