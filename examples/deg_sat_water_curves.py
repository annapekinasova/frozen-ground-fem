import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.materials import (
    Material,
)


def main():
    T_plot = np.hstack([
        np.linspace(-40, -1, 16),
        np.linspace(-0.9, -0.1, 9),
        np.linspace(-0.09, -0.01, 9),
        np.linspace(-0.009, -0.001, 9),
        np.linspace(-0.0009, -0.0001, 9),
    ])

    Sw_exp = np.array([
        0.0115, 0.0124, 0.0136, 0.0150, 0.0170, 0.0200, 0.0250, 0.0265, 0.0283,
        0.0304, 0.0331, 0.0365, 0.0412, 0.0482, 0.0600, 0.0872, 0.0923, 0.0983,
        0.1057, 0.1148, 0.1266, 0.1427, 0.1666, 0.2068, 0.2983, 0.3151, 0.3348,
        0.3586, 0.3877, 0.4245, 0.4729, 0.5400, 0.6399, 0.8005, 0.8212, 0.8428,
        0.8650, 0.8878, 0.9108, 0.9336, 0.9554, 0.9751, 0.9912, 0.9925, 0.9937,
        0.9949, 0.9959, 0.9969, 0.9978, 0.9986, 0.9992, 0.9997,
    ])

    dSwdT_exp = np.array([
        1.6752E-04, 2.0464E-04, 2.5807E-04, 3.3994E-04, 4.7687E-04, 7.3888E-04,
        1.3725E-03, 1.6125E-03, 1.9312E-03, 2.3695E-03, 3.0009E-03, 3.9690E-03,
        5.5896E-03, 8.6932E-03, 1.6204E-02, 4.6992E-02, 5.5244E-02, 6.6193E-02,
        8.1249E-02, 1.0293E-01, 1.3612E-01, 1.9156E-01, 2.9733E-01, 5.5091E-01,
        1.5557E+00, 1.8158E+00, 2.1551E+00, 2.6112E+00, 3.2473E+00, 4.1764E+00,
        5.6169E+00, 8.0260E+00, 1.2417E+01, 2.0280E+01, 2.1146E+01, 2.1931E+01,
        2.2569E+01, 2.2963E+01, 2.2975E+01, 2.2408E+01, 2.0977E+01, 1.8247E+01,
        1.3380E+01, 1.2706E+01, 1.1983E+01, 1.1202E+01, 1.0353E+01, 9.4206E+00,
        8.3829E+00, 7.2016E+00, 5.8037E+00, 4.0035E+00,
    ])

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

    plt.figure(figsize=(6,6))

    plt.subplot(2,2,1)
    plt.plot(T_plot, Sw_act, '-k', label="actual")
    plt.plot(T_plot, Sw_exp, '--r', label="expected")
    plt.xlim((-20, 0))
    plt.legend()
    plt.ylabel("Sw")

    plt.subplot(2,2,3)
    plt.plot(T_plot, dSwdT_act, '-k', label="actual")
    plt.plot(T_plot, dSwdT_exp, '--r', label="expected")
    plt.xlim((-20, 0))
    plt.legend()
    plt.ylabel("dSw/dT")
    plt.xlabel("Temperature, T [deg C]")

    plt.subplot(2,2,2)
    plt.plot(T_plot, Sw_act, '-k', label="actual")
    plt.plot(T_plot, Sw_exp, '--r', label="expected")
    plt.xlim((-1, 0))
    plt.legend()

    plt.subplot(2,2,4)
    plt.plot(T_plot, dSwdT_act, '-k', label="actual")
    plt.plot(T_plot, dSwdT_exp, '--r', label="expected")
    plt.xlim((-1, 0))
    plt.legend()
    plt.xlabel("Temperature, T [deg C]")

    plt.savefig("examples/deg_sat_water_curves.png")


if __name__ == "__main__":
    main()
