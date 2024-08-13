"""runtime performance plots for thaw consolidation lab benchmark"""

import os

import numpy as np
import matplotlib.pyplot as plt


def main():
    # s_per_min = 60.0
    # cm_per_m = 100.0

    plot_root = "/home/karcheba/Dropbox/Anna work_school/PhD/Research/Numerical/thaw consolidation lab/"
    fnames_mbp_M3_8GB = [
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_10_1p0e-02_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_10_1p0e-03_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_10_1p0e-04_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_10_1p0e-05_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_25_1p0e-02_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_25_1p0e-03_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_25_1p0e-04_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_25_1p0e-05_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_50_1p0e-02_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_50_1p0e-03_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_50_1p0e-04_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_50_1p0e-05_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_100_1p0e-02_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_100_1p0e-03_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_100_1p0e-04_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_100_1p0e-05_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_200_1p0e-02_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_200_1p0e-03_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_200_1p0e-04_t_rt.out",
        plot_root + "mbp_M3_8GB/" + "thaw_consol_lab_200_1p0e-05_t_rt.out",
    ]
    fnames_core_i9 = [
        plot_root + "core_i9/" + "thaw_consol_lab_10_1p0e-02_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_10_1p0e-03_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_10_1p0e-04_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_10_1p0e-05_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_25_1p0e-02_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_25_1p0e-03_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_25_1p0e-04_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_25_1p0e-05_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_50_1p0e-02_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_50_1p0e-03_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_50_1p0e-04_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_50_1p0e-05_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_100_1p0e-02_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_100_1p0e-03_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_100_1p0e-04_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_100_1p0e-05_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_200_1p3e-02_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_200_1p3e-03_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_200_1p0e-04_t_rt.out",
        plot_root + "core_i9/" + "thaw_consol_lab_200_1p0e-05_t_rt.out",
    ]
    fnames_core_i7_dell = [
        plot_root + "core_i7_dell/" + "thaw_consol_lab_10_1p0e-02_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_10_1p0e-03_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_10_1p0e-04_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_10_1p0e-05_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_25_1p0e-02_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_25_1p0e-03_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_25_1p0e-04_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_25_1p0e-05_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_50_1p0e-02_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_50_1p0e-03_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_50_1p0e-04_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_50_1p0e-05_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_100_1p0e-02_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_100_1p0e-03_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_100_1p0e-04_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_100_1p0e-05_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_200_1p0e-02_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_200_1p0e-03_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_200_1p0e-04_t_rt.out",
        plot_root + "core_i7_dell/" + "thaw_consol_lab_200_1p0e-05_t_rt.out",
    ]
    fnames_core_i5 = [
        plot_root + "core_i5/" + "thaw_consol_lab_10_1p0e-02_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_10_1p0e-03_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_10_1p0e-04_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_10_1p0e-05_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_25_1p0e-02_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_25_1p0e-03_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_25_1p0e-04_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_25_1p0e-05_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_50_1p0e-02_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_50_1p0e-03_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_50_1p0e-04_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_50_1p0e-05_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_100_1p0e-02_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_100_1p0e-03_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_100_1p0e-04_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_100_1p0e-05_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_200_1p0e-02_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_200_1p5e-03_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_200_1p0e-04_t_rt.out",
        plot_root + "core_i5/" + "thaw_consol_lab_200_1p0e-05_t_rt.out",
    ]

    ind_t = 68
    nel = np.array([10, 25, 50, 100, 200], dtype=int)
    tol = np.array([1e-2, 1e-3, 1e-4, 1e-5])

    rt_mbp_M3_8GB = np.ones((4, 5)) * np.nan
    rt_core_i9 = np.ones_like(rt_mbp_M3_8GB) * np.nan
    rt_core_i7_dell = np.ones_like(rt_mbp_M3_8GB) * np.nan
    rt_core_i5 = np.ones_like(rt_mbp_M3_8GB) * np.nan

    print("mbp_M3_8GB")
    for k, f in enumerate(fnames_mbp_M3_8GB):
        if os.path.isfile(f):
            rt = np.loadtxt(f, unpack=True, usecols=[1])[ind_t]
            i = k % 4
            j = k // 4
            rt_mbp_M3_8GB[i, j] = rt
            print(f"tol={tol[i]:0.2e}, nel={nel[j]}, rt={rt:0.3e}")
    print("\ncore_i9")
    for k, f in enumerate(fnames_core_i9):
        if os.path.isfile(f):
            rt = np.loadtxt(f, unpack=True, usecols=[1])[ind_t]
            i = k % 4
            j = k // 4
            rt_core_i9[i, j] = rt
            print(f"tol={tol[i]:0.2e}, nel={nel[j]}, rt={rt:0.3e}")
    print("\ncore_i7_dell")
    for k, f in enumerate(fnames_core_i7_dell):
        if os.path.isfile(f):
            rt = np.loadtxt(f, unpack=True, usecols=[1])[ind_t]
            i = k % 4
            j = k // 4
            rt_core_i7_dell[i, j] = rt
            print(f"tol={tol[i]:0.2e}, nel={nel[j]}, rt={rt:0.3e}")
    print("\ncore_i5")
    for k, f in enumerate(fnames_core_i5):
        if os.path.isfile(f):
            rt = np.loadtxt(f, unpack=True, usecols=[1])[ind_t]
            i = k % 4
            j = k // 4
            rt_core_i5[i, j] = rt
            print(f"tol={tol[i]:0.2e}, nel={nel[j]}, rt={rt:0.3e}")

    # set plotting parameters
    plt.rc("font", size=9)
    plt.rc(
        "lines",
        linewidth=1.0,
        color="black",
        markeredgewidth=1.0,
        markerfacecolor="none",
        markersize=5,
    )

    # runtime vs. error tolerance
    plt.figure(figsize=(6.0, 7.0))

    plt.loglog(tol, rt_mbp_M3_8GB[:, 2], ":ok", label="mbp_M3, nel=50")
    plt.loglog(tol, rt_mbp_M3_8GB[:, 3], "--ok", label="mbp_M3, 100")
    plt.loglog(tol, rt_mbp_M3_8GB[:, 4], "-ok", label="mbp_M3, 200")

    plt.loglog(tol, rt_core_i9[:, 2], ":sr", label="core_i9, 50")
    plt.loglog(tol, rt_core_i9[:, 3], "--sr", label="core_i9, 100")
    plt.loglog(tol, rt_core_i9[:, 4], "-sr", label="core_i9, 200")

    plt.loglog(tol, rt_core_i7_dell[:, 2], ":dg", label="core_i7_dell, 50")
    plt.loglog(tol, rt_core_i7_dell[:, 3], "--dg", label="core_i7_dell, 100")
    plt.loglog(tol, rt_core_i7_dell[:, 4], "-dg", label="core_i7_dell, 200")

    plt.loglog(tol, rt_core_i5[:, 2], ":^b", label="core_i5, 50")
    plt.loglog(tol, rt_core_i5[:, 3], "--^b", label="core_i5, 100")
    plt.loglog(tol, rt_core_i5[:, 4], "-^b", label="core_i5, 200")

    plt.xlabel("Implicit error tolerance")
    plt.ylabel("Run time [min]")
    plt.legend(frameon=False)
    plt.savefig(plot_root + "runtime_tol.svg")

    # runtime vs. number of elements
    plt.figure(figsize=(6.0, 7.0))

    plt.loglog(nel, rt_mbp_M3_8GB[1, :], ":ok", label="mbp_M3, tol=1e-3")
    plt.loglog(nel, rt_mbp_M3_8GB[2, :], "--ok", label="mbp_M3, 1e-4")
    plt.loglog(nel, rt_mbp_M3_8GB[3, :], "-ok", label="mbp_M3, 1e-5")

    plt.loglog(nel, rt_core_i9[1, :], ":sr", label="core_i9, 1e-3")
    plt.loglog(nel, rt_core_i9[2, :], "--sr", label="core_i9, 1e-4")
    plt.loglog(nel, rt_core_i9[3, :], "-sr", label="core_i9, 1e-5")

    plt.loglog(nel, rt_core_i7_dell[1, :], ":dg", label="core_i7_dell, 1e-3")
    plt.loglog(nel, rt_core_i7_dell[2, :], "--dg", label="core_i7_dell, 1e-4")
    plt.loglog(nel, rt_core_i7_dell[3, :], "-dg", label="core_i7_dell, 1e-5")

    plt.loglog(nel, rt_core_i5[1, :], ":^b", label="core_i5, 1e-3")
    plt.loglog(nel, rt_core_i5[2, :], "--^b", label="core_i5, 1e-4")
    plt.loglog(nel, rt_core_i5[3, :], "-^b", label="core_i5, 1e-5")

    plt.xlabel("Number of elements")
    plt.ylabel("Run time [min]")
    plt.legend(frameon=False)
    plt.savefig(plot_root + "runtime_nel.svg")


if __name__ == "__main__":
    main()
