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
            1.1485e-02,
            1.2410e-02,
            1.3558e-02,
            1.5038e-02,
            1.7047e-02,
            2.0008e-02,
            2.5017e-02,
            2.6504e-02,
            2.8268e-02,
            3.0406e-02,
            3.3071e-02,
            3.6519e-02,
            4.1222e-02,
            4.8175e-02,
            5.9986e-02,
            8.7193e-02,
            9.2286e-02,
            9.8331e-02,
            1.0566e-01,
            1.1480e-01,
            1.2662e-01,
            1.4275e-01,
            1.6656e-01,
            2.0684e-01,
            2.9826e-01,
            3.1506e-01,
            3.3484e-01,
            3.5855e-01,
            3.8766e-01,
            4.2447e-01,
            4.7287e-01,
            5.3997e-01,
            6.3988e-01,
            8.0050e-01,
            8.2121e-01,
            8.4276e-01,
            8.6503e-01,
            8.8782e-01,
            9.1083e-01,
            9.3358e-01,
            9.5536e-01,
            9.7510e-01,
            9.9115e-01,
            9.9246e-01,
            9.9369e-01,
            9.9485e-01,
            9.9593e-01,
            9.9692e-01,
            9.9781e-01,
            9.9859e-01,
            9.9924e-01,
            9.9974e-01,
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
