import os
import random
import time
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer


def calGSI(ax, ay, az, t):
    """
    Calculate Gadd Severity Index (GSI)

    Args:
        ax (np.array): X-Component of acceleration at center of gravity of head {g}
        ay (np.array): Y-Component of acceleration at center of gravity of head {g}
        az (np.array): Z-Component of acceleration at center of gravity of head {g}
        t  (np.array): Time vector {s}

    Returns:
        gsi (np.float): GSI Value
    """

    t_idx = np.where(t == 0)

    t = t[t_idx[0][0] :]
    ax = ax[t_idx[0][0] :]
    ay = ay[t_idx[0][0] :]
    az = az[t_idx[0][0] :]

    ar = np.sqrt(ax**2 + ay**2 + az**2)

    try:
        # Calculate GSI
        gsi = integrate.trapezoid(ar ** (2.5), t)

        return gsi

    except Exception as e:
        print(f"Exception has Occurred: ''{e}''")
        return None


def calHIC(ax, ay, az, t, opt=None):
    """
    Calculate Head Injury Criteria (HIC)

    Args:
        ax (np.array): X-Component of acceleration at center of gravity of head {g}
        ay (np.array): Y-Component of acceleration at center of gravity of head {g}
        az (np.array): Z-Component of acceleration at center of gravity of head {g}
        t  (np.array): Time vector {s}

    Returns:
        hic  (np.float): HIC value
        prob (np.float): Skull fracture probability value
    """

    t_idx = np.where(t == 0)

    t = t[t_idx[0][0] :]
    ax = ax[t_idx[0][0] :]
    ay = ay[t_idx[0][0] :]
    az = az[t_idx[0][0] :]

    ar = np.sqrt(ax**2 + ay**2 + az**2)
    mu = 6.96352
    sigma = 0.84664

    if opt is not None:
        ar_max = np.max(ar)
        ar_max_idx = np.where(ar == ar_max)[0][0]
        td = abs(t[0] - t[1])
        if opt == "HIC36":
            try:
                t = t[
                    int(ar_max_idx - np.ceil(0.018 / td)) : int(
                        ar_max_idx + np.ceil(0.018 / td)
                    )
                    + 1
                ]
                ar = ar[
                    int(ar_max_idx - np.ceil(0.018 / td)) : int(
                        ar_max_idx + np.ceil(0.018 / td)
                    )
                    + 1
                ]
            except:
                if ar_max_idx > ar.shape[0] / 2:
                    idx_r = int(ar.shape[0] - ar_max_idx)
                    idx_l = int(np.ceil(0.036 / td) - idx_r)
                    t = t[ar_max_idx - idx_l : ar_max_idx + idx_r + 1]
                    ar = ar[ar_max_idx - idx_l : ar_max_idx + idx_r + 1]
                elif ar_max_idx < ar.shape[0] / 2:
                    idx_l = int(ar_max_idx)
                    idx_r = int(np.ceil(0.036 / td) - idx_l)
                    t = t[ar_max_idx - idx_l : ar_max_idx + idx_r + 1]
                    ar = ar[ar_max_idx - idx_l : ar_max_idx + idx_r + 1]
        elif opt == "HIC15":
            try:
                t = t[
                    int(ar_max_idx - np.ceil(0.0075 / td)) : int(
                        ar_max_idx + np.ceil(0.0075 / td)
                    )
                    + 1
                ]
                ar = ar[
                    int(ar_max_idx - np.ceil(0.0075 / td)) : int(
                        ar_max_idx + np.ceil(0.0075 / td)
                    )
                    + 1
                ]
            except:
                if ar_max_idx > ar.shape[0] / 2:
                    idx_r = int(ar.shape[0] - ar_max_idx)
                    idx_l = int(np.ceil(0.015 / td) - idx_r)
                    t = t[ar_max_idx - idx_l : ar_max_idx + idx_r + 1]
                    ar = ar[ar_max_idx - idx_l : ar_max_idx + idx_r + 1]
                elif ar_max_idx < ar.shape[0] / 2:
                    idx_l = int(ar_max_idx)
                    idx_r = int(np.ceil(0.015 / td) - idx_l)
                    t = t[ar_max_idx - idx_l : ar_max_idx + idx_r + 1]
                    ar = ar[ar_max_idx - idx_l : ar_max_idx + idx_r + 1]

    try:
        # Calculate HIC
        hic = (t[-1] - t[0]) * np.max(
            (1 / (t[-1] - t[0])) * integrate.trapezoid(ar, t)
        ) ** (2.5)

        # Calculate probability of skulls fracture
        prob = stats.norm.cdf((np.log(hic) - mu) / sigma)

        return hic, prob

    except Exception as e:
        print(f"Exception has Occurred: ''{e}''")
        return None


def calc3ms(ax, ay, az, t, tol=60):
    """
    Calculate 3ms Criterion

    Args:
        ax  (np.array): X-Component of acceleration at chest {g}
        ay  (np.array): Y-Component of acceleration at chest {g}
        az  (np.array): Z-Component of acceleration at chest {g}
        t   (np.array): Time vector {s}
        tol (np.float): 3ms Tolerance {g}

    Returns:
        ms3_val (np.float): 3ms Criterion value
    """

    t_idx = np.where(t == 0)

    t = t[t_idx[0][0] :]
    ax = ax[t_idx[0][0] :]
    ay = ay[t_idx[0][0] :]
    az = az[t_idx[0][0] :]

    td = np.abs(t[2] - t[1])
    ms3_val = 0

    try:
        ar = np.sqrt(ax**2 + ay**2 + az**2)
        for i, value in enumerate(ar):
            if value < tol and value > ms3_val:
                ms3_val = value

        # for i, value in enumerate(ar):
        #     if value <= tol:
        #         ms3_count += 1
        #     else:
        #         ms3_count = 0

        #     if ms3_count >= (3e-3/td) and value < ms3_val:
        #         ms3_val = value

        return ms3_val
    except Exception as e:
        print(f"Exception has Occurred: ''{e}''")
        return None


def calcCTI(ms3, D, A_int=85, D_int=102):
    """
    Calculate Combined Thoracic Index (CTI)

    Args:
        ms3   (np.float): 3ms Value {g}
        D     (np.array): Chest deflection vector {mm}
        A_int (np.float): Intercept value for the acceleration (85g for H3 50th percentile dummy) [optional]
        D_int (np.float): Intercept value for the deflection (102mm for H3 50th percentile dummy) [optional]

    Returns:
        cti (np.float): CTI value
    """

    cti = (ms3 / A_int) + (np.max(D) / D_int)
    return cti


def calcCSI(ax, ay, az, t):
    """
    Calculate Chest Severity Index (CSI)

    Args:
        ax  (np.array): X-Component of acceleration at chest {g}
        ay  (np.array): Y-Component of acceleration at chest {g}
        az  (np.array): Z-Component of acceleration at chest {g}
        t   (np.array): Time vector {s}

    Returns:
        csi (np.float): CSI value
    """

    t_idx = np.where(t == 0)

    t = t[t_idx[0][0] :]
    ax = ax[t_idx[0][0] :]
    ay = ay[t_idx[0][0] :]
    az = az[t_idx[0][0] :]

    try:
        ar = np.sqrt(ax**2 + ay**2 + az**2)
        csi = (integrate.trapezoid(ar, t)) ** (2.5)

        return csi
    except Exception as e:
        print(f"Exception has Occurred: ''{e}''")
        return None


def calcFemurLoad(f, tol=10):
    """
    Calculate Femur Load

    Args:
        f   (np.array): Femur force {KN}
        tol (np.float): Femur load tolerance {g}

    Returns:
        fl_val  (np.float): Femur load value
        fl_flag (bool): Tolerance flag
        prob (np.array): Probability of injury
    """

    fl_val = np.max(f)

    if fl_val > tol:
        fl_flag = True
    else:
        fl_flag = False

    prob = 1 / (1 + np.exp(5.795 - (0.5196 * (fl_val / 1000))))

    return fl_val, fl_flag, prob


def calcNkm(fx, my, t):
    """
    Calculate Neck Protection Criterion (Nkm)

    Args:
        fx   (np.array): Neck Force in x-axis {KN}
        my   (np.array): Neck Moment in y-axis {g}

    Returns:
        nfa  (np.float): Nkm (Flexion-Positive)
        nea  (np.float): Nkm (Extension-Positive)
        nfp  (np.float): Nkm (Flexion-Negative)
        nep  (np.float): Nkm (Extension-Negative)
    """

    def Nkm(f, m, fint, mint, time):
        nkm = (f / fint) + (m / mint)
        # return integrate.trapezoid(nkm, time)
        return nkm

    t_idx = np.where(t == 0)

    t = t[t_idx[0][0] :]
    fx = fx[t_idx[0][0] :]
    my = my[t_idx[0][0] :]

    nfa = Nkm(max(abs(fx)), max(abs(my)), 845, 88.1, t)
    nea = Nkm(max(abs(fx)), max(abs(my)), 845, 47.5, t)
    nfp = Nkm(max(abs(fx)), max(abs(my)), 845, 88.1, t)
    nep = Nkm(max(abs(fx)), max(abs(my)), 845, 47.5, t)

    return nfa, nea, nfp, nep


def plotRegression(x, y):
    """
    Plot regression line using linear regression

    Args:
        x  (np.array): x-axis data
        y  (np.array): y-axis data
    Returns:
        m  (np.float): Regression line slope
        c  (np.float): Regression line intercept
    """
    regres = LinearRegression()
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    y_imputed = imputer.fit_transform(np.array(y).reshape(-1, 1))
    regres.fit(np.array(x).reshape(-1, 1), y_imputed)
    m = regres.coef_.flatten()[-1]
    c = regres.intercept_.flatten()[-1]
    return m, c


def plotData(
    basePath,
    ddata,
    name,
    regres=True,
    lim=True,
    pltType="scatter",
    regColor="red",
    limColor="black",
):
    """
    Plot and save data in graphs

    Args:
        basePath  (str) : Base path of project
        ddata     (dict): Data dictionary
        name      (str) : Name for the results
        regres    (bool): Regression line flag      [Optional]
        lim       (bool): Limit line flag           [Optional]
        pltType   (str) : Plot type (scatter, line) [Optional]
        regColor  (str) : Regression line color     [Optional]
    """

    year = []
    gsi_1 = []
    hic_1 = []
    prob_head_1 = []
    hic36_1 = []
    prob_head36_1 = []
    hic15_1 = []
    prob_head15_1 = []
    ms3h_1 = []
    ms3_1 = []
    cti_1 = []
    csi_1 = []
    f_val_1 = []
    f_prob_1 = []
    f_flag_1 = []
    nfa_1 = []
    nea_1 = []
    nfp_1 = []
    nep_1 = []
    nkmm_1 = []

    gsi_2 = []
    hic_2 = []
    prob_head_2 = []
    hic36_2 = []
    prob_head36_2 = []
    hic15_2 = []
    prob_head15_2 = []
    ms3h_2 = []
    ms3_2 = []
    cti_2 = []
    csi_2 = []
    f_val_2 = []
    f_prob_2 = []
    f_flag_2 = []
    nfa_2 = []
    nea_2 = []
    nfp_2 = []
    nep_2 = []
    nkmm_2 = []

    os.makedirs(os.path.join(basePath, "Results", name))

    if pltType == "scatter":
        colors = [
            f"#{random.randint(0, 255):02X}{random.randint(0, 255):02X}{random.randint(0, 255):02X}"
            for _ in ddata
        ]

        for itr, myDict in enumerate(ddata):
            year.append(int(myDict["date"]["year"]))
            gsi_1.append(myDict["proc_01"]["gsi"])
            hic_1.append(myDict["proc_01"]["hic"])
            prob_head_1.append(myDict["proc_01"]["prob_head"])
            hic36_1.append(myDict["proc_01"]["hic36"])
            prob_head36_1.append(myDict["proc_01"]["prob_head36"])
            hic15_1.append(myDict["proc_01"]["hic15"])
            prob_head15_1.append(myDict["proc_01"]["prob_head15"])
            ms3h_1.append(myDict["proc_01"]["ms3h"])
            ms3_1.append(myDict["proc_01"]["ms3"])
            cti_1.append(myDict["proc_01"]["cti"])
            csi_1.append(myDict["proc_01"]["csi"])
            f_val_1.append(myDict["proc_01"]["fval"])
            f_prob_1.append(myDict["proc_01"]["f_prob"])
            f_flag_1.append(myDict["proc_01"]["f_flag"])
            nfa_1.append(myDict["proc_01"]["nfa"])
            nea_1.append(myDict["proc_01"]["nea"])
            nfp_1.append(myDict["proc_01"]["nfp"])
            nep_1.append(myDict["proc_01"]["nep"])
            try:
                nkmm_1.append(
                    max(
                        [
                            myDict["proc_01"]["nfa"],
                            myDict["proc_01"]["nea"],
                            myDict["proc_01"]["nfp"],
                            myDict["proc_01"]["nep"],
                        ]
                    )
                )
            except:
                nkmm_1.append(None)

            gsi_2.append(myDict["proc_02"]["gsi"])
            hic_2.append(myDict["proc_02"]["hic"])
            prob_head_2.append(myDict["proc_02"]["prob_head"])
            hic36_2.append(myDict["proc_02"]["hic36"])
            prob_head36_2.append(myDict["proc_02"]["prob_head36"])
            hic15_2.append(myDict["proc_02"]["hic15"])
            prob_head15_2.append(myDict["proc_02"]["prob_head15"])
            ms3h_2.append(myDict["proc_02"]["ms3h"])
            ms3_2.append(myDict["proc_02"]["ms3"])
            cti_2.append(myDict["proc_02"]["cti"])
            csi_2.append(myDict["proc_02"]["csi"])
            f_val_2.append(myDict["proc_02"]["fval"])
            f_prob_2.append(myDict["proc_02"]["f_prob"])
            f_flag_2.append(myDict["proc_02"]["f_flag"])
            nfa_2.append(myDict["proc_02"]["nfa"])
            nea_2.append(myDict["proc_02"]["nea"])
            nfp_2.append(myDict["proc_02"]["nfp"])
            nep_2.append(myDict["proc_02"]["nep"])
            try:
                nkmm_2.append(
                    max(
                        [
                            myDict["proc_02"]["nfa"],
                            myDict["proc_02"]["nea"],
                            myDict["proc_02"]["nfp"],
                            myDict["proc_02"]["nep"],
                        ]
                    )
                )
            except:
                nkmm_2.append(None)

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, gsi_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, gsi_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax1.set_ylabel("GSI (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, gsi_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, gsi_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax2.set_ylabel("GSI (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"GSI ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, hic_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, hic_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax1.set_ylabel("HIC (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, hic_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, hic_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax2.set_ylabel("HIC (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"HIC ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, prob_head_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, prob_head_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax1.set_ylabel("" r"$\rho$ (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, prob_head_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, prob_head_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax2.set_ylabel("" r"$\rho$ (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"Head Fracture Probability ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, hic36_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, hic36_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        if lim:
            ax1.axhline(1000, color=limColor, linestyle="--")
        ax1.set_ylabel("HIC36 (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, hic36_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, hic36_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        if lim:
            ax2.axhline(1000, color=limColor, linestyle="--")
        ax2.set_ylabel("HIC36 (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"HIC36 ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, prob_head36_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, prob_head36_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax1.set_ylabel("" r"$\rho$ (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, prob_head36_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, prob_head36_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax2.set_ylabel("" r"$\rho$ (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"Head Fracture Probability (HIC36) ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, hic15_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, hic15_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        if lim:
            ax1.axhline(700, color=limColor, linestyle="--")
        ax1.set_ylabel("HIC15 (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, hic15_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, hic15_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        if lim:
            ax2.axhline(700, color=limColor, linestyle="--")
        ax2.set_ylabel("HIC15 (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"HIC15 ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, prob_head15_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, prob_head15_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax1.set_ylabel("" r"$\rho$ (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, prob_head15_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, prob_head15_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax2.set_ylabel("" r"$\rho$ (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"Head Fracture Probability (HIC15) ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, ms3h_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, ms3h_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        if lim:
            ax1.axhline(80, color=limColor, linestyle="--")
        ax1.set_ylabel("3ms (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, ms3h_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, ms3h_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        if lim:
            ax2.axhline(80, color=limColor, linestyle="--")
        ax2.set_ylabel("3ms (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"3ms (Head) Criterion ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, ms3_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, ms3_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        if lim:
            ax1.axhline(60, color=limColor, linestyle="--")
        ax1.set_ylabel("3ms (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, ms3_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, ms3_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        if lim:
            ax2.axhline(60, color=limColor, linestyle="--")
        ax2.set_ylabel("3ms (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"3ms (Chest) Criterion ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, cti_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, cti_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax1.set_ylabel("CTI (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, cti_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, cti_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax2.set_ylabel("CTI (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"CTI ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, csi_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, csi_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax1.set_ylabel("CSI (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, csi_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, csi_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax2.set_ylabel("CSI (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"CSI ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, f_val_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, f_val_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax1.set_ylabel("Femur Load (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, f_val_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, f_val_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax2.set_ylabel("Femur Load (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"Femur Load Value ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, f_prob_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, f_prob_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax1.set_ylabel("" r"$\rho$ (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, f_prob_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, f_prob_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax2.set_ylabel("" r"$\rho$ (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"Femur Injury Probability ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, nfa_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, nfa_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax1.set_ylabel("N$_f$$_a$ (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, nfa_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, nfa_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax2.set_ylabel("N$_f$$_a$ (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"N$_f$$_a$ ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, nea_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, nea_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax1.set_ylabel("N$_e$$_a$ (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, nea_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, nea_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax2.set_ylabel("N$_e$$_a$ (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"N$_e$$_a$ ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, nfp_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, nfp_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax1.set_ylabel("N$_f$$_p$ (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, nfp_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, nfp_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax2.set_ylabel("N$_f$$_p$ (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"N$_f$$_p$ ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, nep_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, nep_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax1.set_ylabel("N$_e$$_p$ (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, nep_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, nep_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax2.set_ylabel("N$_e$$_p$ (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"N$_e$$_p$ ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.scatter(year, nkmm_1, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, nkmm_1)
            ax1.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax1.set_ylabel("N$_k$$_m$ (Right Front Seat)")
        # ax1.grid(True)
        ax2.scatter(year, nkmm_2, s=10, c=colors)
        if regres:
            m, c = plotRegression(year, nkmm_2)
            ax2.plot(np.array(year), m * np.array(year) + c, color=regColor)
        ax2.set_ylabel("N$_k$$_m$ (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"N$_k$$_m$$^m$$^a$$^x$ ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

    elif pltType == "line":
        for item in ddata:
            year.append(int(item[0]))
            gsi_1.append(item[1]["gsi"] if not item[1]["gsi"] == None else 0)
            hic_1.append(item[1]["hic"] if not item[1]["hic"] == None else 0)
            prob_head_1.append(
                item[1]["prob_head"] if not item[1]["prob_head"] == None else 0
            )
            ms3_1.append(item[1]["ms3"] if not item[1]["ms3"] == None else 0)
            cti_1.append(item[1]["cti"] if not item[1]["cti"] == None else 0)
            csi_1.append(item[1]["csi"] if not item[1]["csi"] == None else 0)
            f_val_1.append(item[1]["fval"] if not item[1]["fval"] == None else 0)
            f_prob_1.append(item[1]["f_prob"] if not item[1]["f_prob"] == None else 0)
            f_flag_1.append(item[1]["f_flag"] if not item[1]["f_flag"] == None else 0)
            gsi_2.append(item[2]["gsi"] if not item[2]["gsi"] == None else 0)
            hic_2.append(item[2]["hic"] if not item[2]["hic"] == None else 0)
            prob_head_2.append(
                item[2]["prob_head"] if not item[2]["prob_head"] == None else 0
            )
            ms3_2.append(item[2]["ms3"] if not item[2]["ms3"] == None else 0)
            cti_2.append(item[2]["cti"] if not item[2]["cti"] == None else 0)
            csi_2.append(item[2]["csi"] if not item[2]["csi"] == None else 0)
            f_val_2.append(item[2]["fval"] if not item[2]["fval"] == None else 0)
            f_prob_2.append(item[2]["f_prob"] if not item[2]["f_prob"] == None else 0)
            f_flag_2.append(item[2]["f_flag"] if not item[2]["f_flag"] == None else 0)

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(year, gsi_1, marker="o")
        ax1.set_ylabel("GSI (Right Front Seat)")
        # ax1.grid(True)
        ax2.plot(year, gsi_2, marker="o")
        ax2.set_ylabel("GSI (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"GSI ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(year, hic_1, marker="o")
        ax1.set_ylabel("HIC (Right Front Seat)")
        # ax1.grid(True)
        ax2.plot(year, hic_2, marker="o")
        ax2.set_ylabel("HIC (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"HIC ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(year, prob_head_1, marker="o")
        ax1.set_ylabel("" r"$\rho$ (Right Front Seat)")
        # ax1.grid(True)
        ax2.plot(year, prob_head_2, marker="o")
        ax2.set_ylabel("" r"$\rho$ (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"Head Fracture Probability ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(year, ms3_1, marker="o")
        ax1.set_ylabel("3ms (Right Front Seat)")
        # ax1.grid(True)
        ax2.plot(year, ms3_2, marker="o")
        ax2.set_ylabel("3ms (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"3ms Criterion ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(year, cti_1, marker="o")
        ax1.set_ylabel("CTI (Right Front Seat)")
        # ax1.grid(True)
        ax2.plot(year, cti_2, marker="o")
        ax2.set_ylabel("CTI (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"CTI ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(year, csi_1, marker="o")
        ax1.set_ylabel("CSI (Right Front Seat)")
        # ax1.grid(True)
        ax2.plot(year, csi_2, marker="o")
        ax2.set_ylabel("CSI (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"CSI ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(year, f_val_1, marker="o")
        ax1.set_ylabel("Femur Load (Right Front Seat)")
        # ax1.grid(True)
        ax2.plot(year, f_val_2, marker="o")
        ax2.set_ylabel("Femur Load (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"Femur Load Value ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(year, f_prob_1, marker="o")
        ax1.set_ylabel("" r"$\rho$ (Right Front Seat)")
        # ax1.grid(True)
        ax2.plot(year, f_prob_2, marker="o")
        ax2.set_ylabel("" r"$\rho$ (Left Front Seat)")
        # ax2.grid(True)
        fig.suptitle(f"Femur Injury Probability ({name})")
        plt.xlabel("Year")
        fig.savefig(os.path.join("Results", name, str(time.time()) + ".png"))
        plt.close()
