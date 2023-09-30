import numpy as np
import matplotlib.pyplot as plt

from frozen_ground_fem.geometry import (
    shape_matrix_cubic,
    gradient_matrix_cubic,
)


def main():
    s_plot = np.linspace(0, 1)
    N_plot = np.zeros((len(s_plot), 4))
    B_plot = np.zeros_like(N_plot)
    for k, s in enumerate(s_plot):
        N = shape_matrix_cubic(s)
        B = gradient_matrix_cubic(s, 1.0)
        N_plot[k, :] = N[0, :]
        B_plot[k, :] = B[0, :]

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(N_plot[:, 0], s_plot, '-k', label=r"$N_0$")
    plt.plot(N_plot[:, 1], s_plot, '--b', label=r"$N_1$")
    plt.plot(N_plot[:, 2], s_plot, ':r', label=r"$N_2$")
    plt.plot(N_plot[:, 3], s_plot, '-.k', label=r"$N_3$")
    plt.xlabel(r"$N_k(s)$")
    plt.ylabel(r"$s$")
    plt.legend()
    plt.ylim((1.0, 0.0))

    plt.subplot(1, 2, 2)
    plt.plot(B_plot[:, 0], s_plot, '-k', label=r"$B_0$")
    plt.plot(B_plot[:, 1], s_plot, '--b', label=r"$B_1$")
    plt.plot(B_plot[:, 2], s_plot, ':r', label=r"$B_2$")
    plt.plot(B_plot[:, 3], s_plot, '-.k', label=r"$B_3$")
    plt.xlabel(r"$B_k(s)$")
    plt.ylabel(r"$s$")
    plt.legend()
    plt.ylim((1.0, 0.0))

    plt.savefig("examples/cubic_shape_functions.svg")


if __name__ == "__main__":
    main()
