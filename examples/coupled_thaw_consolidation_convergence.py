"""convergence plots for thaw consolidation lab benchmark

Notes
-----

This script generates plots only.

It is meant to be run after completion of the script
coupled_thaw_consolidation_lab_benchmark.py
which generates the data to be plotted.

If you run this before running that script,
it will just generate empty plots.

Make sure that the file names for
plot_root
mach_fold
are correct for your system,
otherwise you will get file does not exist errors.
"""

import os

import numpy as np
import matplotlib.pyplot as plt


def main():
    # s_per_min = 60.0
    cm_per_m = 100.0

    plot_root = "/home/karcheba/Dropbox/Anna work_school/PhD/Research/Numerical/thaw consolidation lab/"
    mach_fold = "core_i9/"
    fnames2 = [
        plot_root + mach_fold + "thaw_consol_lab_10_1p0e-02_t_s_Z.out",
        plot_root + mach_fold + "thaw_consol_lab_25_1p0e-02_t_s_Z.out",
        plot_root + mach_fold + "thaw_consol_lab_50_1p0e-02_t_s_Z.out",
        plot_root + mach_fold + "thaw_consol_lab_100_1p0e-02_t_s_Z.out",
        plot_root + mach_fold + "thaw_consol_lab_200_1p0e-02_t_s_Z.out",
    ]
    fnames3 = [
        plot_root + mach_fold + "thaw_consol_lab_10_1p0e-03_t_s_Z.out",
        plot_root + mach_fold + "thaw_consol_lab_25_1p0e-03_t_s_Z.out",
        plot_root + mach_fold + "thaw_consol_lab_50_1p0e-03_t_s_Z.out",
        plot_root + mach_fold + "thaw_consol_lab_100_1p0e-03_t_s_Z.out",
        plot_root + mach_fold + "thaw_consol_lab_200_1p0e-03_t_s_Z.out",
    ]
    fnames4 = [
        plot_root + mach_fold + "thaw_consol_lab_10_1p0e-04_t_s_Z.out",
        plot_root + mach_fold + "thaw_consol_lab_25_1p0e-04_t_s_Z.out",
        plot_root + mach_fold + "thaw_consol_lab_50_1p0e-04_t_s_Z.out",
        plot_root + mach_fold + "thaw_consol_lab_100_1p0e-04_t_s_Z.out",
        plot_root + mach_fold + "thaw_consol_lab_200_1p0e-04_t_s_Z.out",
    ]
    fnames5 = [
        plot_root + mach_fold + "thaw_consol_lab_10_1p0e-05_t_s_Z.out",
        plot_root + mach_fold + "thaw_consol_lab_25_1p0e-05_t_s_Z.out",
        plot_root + mach_fold + "thaw_consol_lab_50_1p0e-05_t_s_Z.out",
        plot_root + mach_fold + "thaw_consol_lab_100_1p0e-05_t_s_Z.out",
        plot_root + mach_fold + "thaw_consol_lab_200_1p0e-05_t_s_Z.out",
    ]

    ind_t = np.array([23, 41, 68, 76], dtype=int)
    nel = np.array([10, 25, 50, 100, 200], dtype=int)
    # t = np.loadtxt(fnames2[0], unpack=True)[0][ind_t] / s_per_min

    s_YuEtAl20 = np.array([0.218, 0.910, 1.718])
    z_YuEtAl20 = np.array([0.503, 2.105, 3.973])
    s_DK18 = np.array([0.230, 1.065, 2.081])
    z_DK18 = np.array([0.612, 2.594, 4.980])

    s2 = np.ones((len(ind_t), len(nel))) * np.nan
    s3 = np.ones_like(s2) * np.nan
    s4 = np.ones_like(s2) * np.nan
    s5 = np.ones_like(s2) * np.nan
    z2 = np.ones_like(s2) * np.nan
    z3 = np.ones_like(s2) * np.nan
    z4 = np.ones_like(s2) * np.nan
    z5 = np.ones_like(s2) * np.nan

    for k, f in enumerate(fnames2):
        if os.path.isfile(f):
            s, z = np.loadtxt(f, unpack=True, usecols=[1, 2])
            s2[:, k] = s[ind_t] * cm_per_m
            z2[:, k] = z[ind_t] * cm_per_m
    for k, f in enumerate(fnames3):
        if os.path.isfile(f):
            s, z = np.loadtxt(f, unpack=True, usecols=[1, 2])
            s3[:, k] = s[ind_t] * cm_per_m
            z3[:, k] = z[ind_t] * cm_per_m
    for k, f in enumerate(fnames4):
        if os.path.isfile(f):
            s, z = np.loadtxt(f, unpack=True, usecols=[1, 2])
            s4[:, k] = s[ind_t] * cm_per_m
            z4[:, k] = z[ind_t] * cm_per_m
    for k, f in enumerate(fnames5):
        if os.path.isfile(f):
            s, z = np.loadtxt(f, unpack=True, usecols=[1, 2])
            s5[:, k] = s[ind_t] * cm_per_m
            z5[:, k] = z[ind_t] * cm_per_m

    # set plotting parameters
    plt.rc("font", size=9)
    plt.rc(
        "lines",
        linewidth=0.5,
        color="black",
        markeredgewidth=0.5,
        markerfacecolor="none",
        markersize=4,
    )

    # settlement and thaw depth (separate)
    # at all tolerances
    fig = plt.figure(figsize=(10.0, 10.0))

    plt.subplot(2, 2, 1)
    plt.semilogx(nel, s2[0, :], ":ok", label="5 min, tol=1e-2")
    plt.semilogx(nel, s3[0, :], "--sk", label="5, 1e-3")
    plt.semilogx(nel, s4[0, :], "-^k", label="5, 1e-4")
    plt.semilogx(nel, s5[0, :], "-dk", label="5, 1e-5")
    plt.semilogx(
        [nel[0], nel[-1]],
        [s_DK18[0], s_DK18[0]],
        ":k",
        linewidth=1.5,
        label="5, DK18",
    )
    plt.semilogx(
        [nel[0], nel[-1]],
        [s_YuEtAl20[0], s_YuEtAl20[0]],
        "--k",
        linewidth=1.5,
        label="5, YuEtAl20",
    )
    plt.semilogx(nel, s2[1, :], ":ob", label="95, 1e-2")
    plt.semilogx(nel, s3[1, :], "--sb", label="95, 1e-3")
    plt.semilogx(nel, s4[1, :], "-^b", label="95, 1e-4")
    plt.semilogx(nel, s5[1, :], "-db", label="95, 1e-5")
    plt.semilogx(
        [nel[0], nel[-1]],
        [s_DK18[1], s_DK18[1]],
        ":b",
        linewidth=1.5,
        label="95, DK18",
    )
    plt.semilogx(
        [nel[0], nel[-1]],
        [s_YuEtAl20[1], s_YuEtAl20[1]],
        "--b",
        linewidth=1.5,
        label="95, YuEtAl20",
    )
    plt.semilogx(nel, s2[2, :], ":or", label="348, 1e-2")
    plt.semilogx(nel, s3[2, :], "--sr", label="348, 1e-3")
    plt.semilogx(nel, s4[2, :], "-^r", label="348, 1e-4")
    plt.semilogx(nel, s5[2, :], "-dr", label="348, 1e-5")
    plt.semilogx(
        [nel[0], nel[-1]],
        [s_DK18[2], s_DK18[2]],
        ":r",
        linewidth=1.5,
        label="348, DK18",
    )
    plt.semilogx(
        [nel[0], nel[-1]],
        [s_YuEtAl20[2], s_YuEtAl20[2]],
        "--r",
        linewidth=1.5,
        label="348, YuEtAl20",
    )
    plt.xlabel("Number of elements, N")
    plt.ylabel("Settlement, s [cm]")
    plt.ylim([4.0, 0.0])

    plt.subplot(2, 2, 3)
    plt.semilogx(nel, z2[0, :], ":ok")
    plt.semilogx(nel, z3[0, :], "--sk")
    plt.semilogx(nel, z4[0, :], "-^k")
    plt.semilogx(nel, z5[0, :], "-dk")
    plt.semilogx(
        [nel[0], nel[-1]],
        [z_DK18[0], z_DK18[0]],
        ":k",
        linewidth=1.5,
    )
    plt.semilogx(
        [nel[0], nel[-1]],
        [z_YuEtAl20[0], z_YuEtAl20[0]],
        "--k",
        linewidth=1.5,
    )
    plt.semilogx(nel, z2[1, :], ":ob")
    plt.semilogx(nel, z3[1, :], "--sb")
    plt.semilogx(nel, z4[1, :], "-^b")
    plt.semilogx(nel, z5[1, :], "-db")
    plt.semilogx(
        [nel[0], nel[-1]],
        [z_DK18[1], z_DK18[1]],
        ":b",
        linewidth=1.5,
    )
    plt.semilogx(
        [nel[0], nel[-1]],
        [z_YuEtAl20[1], z_YuEtAl20[1]],
        "--b",
        linewidth=1.5,
    )
    plt.semilogx(nel, z2[2, :], ":or")
    plt.semilogx(nel, z3[2, :], "--sr")
    plt.semilogx(nel, z4[2, :], "-^r")
    plt.semilogx(nel, z5[2, :], "-dr")
    plt.semilogx(
        [nel[0], nel[-1]],
        [z_DK18[2], z_DK18[2]],
        ":r",
        linewidth=1.5,
    )
    plt.semilogx(
        [nel[0], nel[-1]],
        [z_YuEtAl20[2], z_YuEtAl20[2]],
        "--r",
        linewidth=1.5,
    )
    plt.xlabel("Number of elements, N")
    plt.ylabel("Thaw depth, z [cm]")
    plt.ylim([8.0, 0.0])

    fig.legend()
    plt.savefig(plot_root + mach_fold + "s_z_tol_all.svg")

    # settlement and thaw depth (together)
    # at tol=1e-4
    fig = plt.figure(figsize=(10.0, 4.0))

    plt.subplot(1, 2, 1)
    plt.semilogx(nel, s4[0, :], "-or", label="s, 5 min")
    plt.semilogx(
        [nel[0], nel[-1]],
        [s_YuEtAl20[0], s_YuEtAl20[0]],
        "--r",
        linewidth=1.5,
    )
    plt.semilogx(nel, z4[0, :], "--ob", label="z, 5")
    plt.semilogx(
        [nel[0], nel[-1]],
        [z_YuEtAl20[0], z_YuEtAl20[0]],
        ":b",
        linewidth=1.5,
    )
    plt.semilogx(nel, s4[1, :], "-sr", label="s, 95")
    plt.semilogx(
        [nel[0], nel[-1]],
        [s_YuEtAl20[1], s_YuEtAl20[1]],
        "--r",
        linewidth=1.5,
    )
    plt.semilogx(nel, z4[1, :], "--sb", label="z, 95")
    plt.semilogx(
        [nel[0], nel[-1]],
        [z_YuEtAl20[1], z_YuEtAl20[1]],
        ":b",
        linewidth=1.5,
    )
    plt.semilogx(nel, s4[2, :], "-^r", label="s, 348")
    plt.semilogx(nel, z4[2, :], "--^b", label="z, 348")
    plt.semilogx(
        [nel[0], nel[-1]],
        [s_YuEtAl20[2], s_YuEtAl20[2]],
        "--r",
        linewidth=1.5,
        label="s, YuEtAl20",
    )
    plt.semilogx(
        [nel[0], nel[-1]],
        [z_YuEtAl20[2], z_YuEtAl20[2]],
        ":b",
        linewidth=1.5,
        label="z, YuEtAl20",
    )
    plt.xlabel("Number of elements, N")
    plt.ylabel("Depth [cm]")
    plt.ylim([5.0, 0.0])

    fig.legend()
    plt.savefig(plot_root + mach_fold + "s_z_tol_1e-4.svg")

    # settlement and thaw depth (together)
    # at tol=1e-5
    fig = plt.figure(figsize=(10.0, 4.0))

    plt.subplot(1, 2, 1)
    plt.semilogx(nel, s5[0, :], "-or", label="s, 5 min")
    plt.semilogx(
        [nel[0], nel[-1]],
        [s_YuEtAl20[0], s_YuEtAl20[0]],
        "--r",
        linewidth=1.5,
    )
    plt.semilogx(nel, z5[0, :], "--ob", label="z, 5")
    plt.semilogx(
        [nel[0], nel[-1]],
        [z_YuEtAl20[0], z_YuEtAl20[0]],
        ":b",
        linewidth=1.5,
    )
    plt.semilogx(nel, s5[1, :], "-sr", label="s, 95")
    plt.semilogx(
        [nel[0], nel[-1]],
        [s_YuEtAl20[1], s_YuEtAl20[1]],
        "--r",
        linewidth=1.5,
    )
    plt.semilogx(nel, z5[1, :], "--sb", label="z, 95")
    plt.semilogx(
        [nel[0], nel[-1]],
        [z_YuEtAl20[1], z_YuEtAl20[1]],
        ":b",
        linewidth=1.5,
    )
    plt.semilogx(nel, s5[2, :], "-^r", label="s, 348")
    plt.semilogx(nel, z5[2, :], "--^b", label="z, 348")
    plt.semilogx(
        [nel[0], nel[-1]],
        [s_YuEtAl20[2], s_YuEtAl20[2]],
        "--r",
        linewidth=1.5,
        label="s, YuEtAl20",
    )
    plt.semilogx(
        [nel[0], nel[-1]],
        [z_YuEtAl20[2], z_YuEtAl20[2]],
        ":b",
        linewidth=1.5,
        label="z, YuEtAl20",
    )
    plt.xlabel("Number of elements, N")
    plt.ylabel("Depth [cm]")
    plt.ylim([5.0, 0.0])

    fig.legend()
    plt.savefig(plot_root + mach_fold + "s_z_tol_1e-5.svg")


if __name__ == "__main__":
    main()
