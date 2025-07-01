"""convergence plots for freezing front lab benchmark

Notes
-----

This script generates plots only.

It is meant to be run after completion of the script
coupled_freezing_front_lab_benchmark.py
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

    plot_root = "/home/karcheba/Dropbox/Anna work_school/PhD/Research/Numerical/freezing front benchmark/"
    mach_fold = "mbp_M3_8GB_seg_rapid/"
    fnames2 = [
        plot_root + mach_fold + "freeze_front_lab_10_1p0e-02_t_s_Z.out",
        plot_root + mach_fold + "freeze_front_lab_25_1p0e-02_t_s_Z.out",
        plot_root + mach_fold + "freeze_front_lab_50_1p0e-02_t_s_Z.out",
        plot_root + mach_fold + "freeze_front_lab_100_1p0e-02_t_s_Z.out",
        plot_root + mach_fold + "freeze_front_lab_200_1p0e-02_t_s_Z.out",
    ]
    fnames3 = [
        plot_root + mach_fold + "freeze_front_lab_10_1p0e-03_t_s_Z.out",
        plot_root + mach_fold + "freeze_front_lab_25_1p0e-03_t_s_Z.out",
        plot_root + mach_fold + "freeze_front_lab_50_1p0e-03_t_s_Z.out",
        plot_root + mach_fold + "freeze_front_lab_100_1p0e-03_t_s_Z.out",
        plot_root + mach_fold + "freeze_front_lab_200_1p0e-03_t_s_Z.out",
    ]
    fnames4 = [
        plot_root + mach_fold + "freeze_front_lab_10_1p0e-04_t_s_Z.out",
        plot_root + mach_fold + "freeze_front_lab_25_1p0e-04_t_s_Z.out",
        plot_root + mach_fold + "freeze_front_lab_50_1p0e-04_t_s_Z.out",
        plot_root + mach_fold + "freeze_front_lab_100_1p0e-04_t_s_Z.out",
        plot_root + mach_fold + "freeze_front_lab_200_1p0e-04_t_s_Z.out",
    ]
    fnames5 = [
        plot_root + mach_fold + "freeze_front_lab_10_1p0e-05_t_s_Z.out",
        plot_root + mach_fold + "freeze_front_lab_25_1p0e-05_t_s_Z.out",
        plot_root + mach_fold + "freeze_front_lab_50_1p0e-05_t_s_Z.out",
        plot_root + mach_fold + "freeze_front_lab_100_1p0e-05_t_s_Z.out",
        plot_root + mach_fold + "freeze_front_lab_200_1p0e-05_t_s_Z.out",
    ]

    ind_t = np.array([22, 27, 42, 72, 82, 96, 114, 136], dtype=int)
    nel = np.array([10, 25, 50, 100, 200], dtype=int)
    # t = np.loadtxt(fnames2[0], unpack=True)[0][ind_t] / s_per_min

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

    # settlement and freeze depth (separate)
    # at all tolerances
    fig = plt.figure(figsize=(10.0, 10.0))

    plt.subplot(2, 2, 1)
    plt.semilogx(nel, s2[0, :], ":ok", label="4 min, tol=1e-2")
    plt.semilogx(nel, s3[0, :], "--sk", label="4, 1e-3")
    plt.semilogx(nel, s4[0, :], "-^k", label="4, 1e-4")
    plt.semilogx(nel, s5[0, :], "-dk", label="4, 1e-5")
    plt.semilogx(nel, s2[2, :], ":ob", label="100, 1e-2")
    plt.semilogx(nel, s3[2, :], "--sb", label="100, 1e-3")
    plt.semilogx(nel, s4[2, :], "-^b", label="100, 1e-4")
    plt.semilogx(nel, s5[2, :], "-db", label="100, 1e-5")
    plt.semilogx(nel, s2[3, :], ":or", label="400, 1e-2")
    plt.semilogx(nel, s3[3, :], "--sr", label="400, 1e-3")
    plt.semilogx(nel, s4[3, :], "-^r", label="400, 1e-4")
    plt.semilogx(nel, s5[3, :], "-dr", label="400, 1e-5")
    plt.semilogx(nel, s2[4, :], ":og", label="900, 1e-2")
    plt.semilogx(nel, s3[4, :], "--sg", label="900, 1e-3")
    plt.semilogx(nel, s4[4, :], "-^g", label="900, 1e-4")
    plt.semilogx(nel, s5[4, :], "-dg", label="900, 1e-5")
    plt.xlabel("Number of elements, N")
    plt.ylabel("Settlement, s [cm]")
    plt.ylim([0.0, -2.0])

    plt.subplot(2, 2, 3)
    plt.semilogx(nel, z2[0, :], ":ok")
    plt.semilogx(nel, z3[0, :], "--sk")
    plt.semilogx(nel, z4[0, :], "-^k")
    plt.semilogx(nel, z5[0, :], "-dk")
    plt.semilogx(nel, z2[2, :], ":ob")
    plt.semilogx(nel, z3[2, :], "--sb")
    plt.semilogx(nel, z4[2, :], "-^b")
    plt.semilogx(nel, z5[2, :], "-db")
    plt.semilogx(nel, z2[3, :], ":or")
    plt.semilogx(nel, z3[3, :], "--sr")
    plt.semilogx(nel, z4[3, :], "-^r")
    plt.semilogx(nel, z5[3, :], "-dr")
    plt.semilogx(nel, z2[4, :], ":og")
    plt.semilogx(nel, z3[4, :], "--sg")
    plt.semilogx(nel, z4[4, :], "-^g")
    plt.semilogx(nel, z5[4, :], "-dg")
    plt.xlabel("Number of elements, N")
    plt.ylabel("Freeze depth, z [cm]")
    plt.ylim([15.0, 0.0])

    fig.legend()
    plt.savefig(plot_root + mach_fold + "s_z_tol_all.svg")

    # freeze and thaw depth (together)
    # at tol=1e-4
    fig = plt.figure(figsize=(10.0, 4.0))

    plt.subplot(1, 2, 1)
    plt.semilogx(nel, s4[0, :], "-or", label="s, 4 min")
    plt.semilogx(nel, z4[0, :], "--ob", label="z, 4")
    plt.semilogx(nel, s4[2, :], "-sr", label="s, 100")
    plt.semilogx(nel, z4[2, :], "--sb", label="z, 100")
    plt.semilogx(nel, s4[3, :], "-^r", label="s, 400")
    plt.semilogx(nel, z4[3, :], "--^b", label="z, 400")
    plt.semilogx(nel, s4[4, :], "-dr", label="s, 900")
    plt.semilogx(nel, z4[4, :], "--db", label="z, 900")
    plt.xlabel("Number of elements, N")
    plt.ylabel("Depth [cm]")
    plt.ylim([15.0, -2.0])

    fig.legend()
    plt.savefig(plot_root + mach_fold + "s_z_tol_1e-4.svg")


if __name__ == "__main__":
    main()
