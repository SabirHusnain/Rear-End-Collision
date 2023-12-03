# -*- coding: utf-8 -*-
"""
Created on Sat Dec 2 23:08:45 2023

@author: Sabir Husnain
"""


import os
import sys
import logging
import shutil
import numpy as np
from tqdm import tqdm
from RearEndUtils import analysisUtils as analysis
from RearEndUtils import fileUtils as procFile


base_path = os.path.dirname(os.path.realpath(sys.argv[0]))
dir_path = "NHTSA"

"""
*****************************************************************************
=========================== Extracting  Meta Data ===========================
*****************************************************************************
"""

data_files = os.listdir(dir_path)
data_files_new = []
for i in data_files:
    if os.path.isdir(os.path.join(dir_path, i)) and i.startswith("v"):
        data_files_new.append(i)
data_files = []
data_files = data_files_new.copy()

print(f"{len(data_files)} files Found in Total!\n")

data = []
for itr, file in enumerate(data_files):
    data.append(procFile.getFileInfo(dirPath=os.path.join(dir_path, file)))

"""
*****************************************************************************
============================== Filtering  Data ==============================
*****************************************************************************
"""

log_file = os.path.join(base_path, "exceptions.log")
logger = logging.getLogger("my_logger")
handler = logging.FileHandler(log_file, mode="w")
logger.addHandler(handler)
logger.setLevel(logging.ERROR)

for itr, myDict in enumerate(tqdm(data)):
    """============ Right Front Seat Data Filter ============"""
    myDict["data_01"] = {}
    myDict["info_01"] = {}

    # Head CG Acceleration Filer
    myDict["data_01"]["hdcg_acc"] = procFile.filterData(
        data=myDict["instrument"],
        sensorAttachment="HDCG",
        sensorLocation="01",
        yUnits="G'S",
    )
    try:
        myDict["info_01"]["hdcg_acc"] = {}
        for j, file in enumerate(myDict["data_01"]["hdcg_acc"]["fileName"]):
            myDict["info_01"]["hdcg_acc"].update(
                {
                    f"{myDict['data_01']['hdcg_acc']['axis'][j]}": procFile.getSensorData(
                        filePath=os.path.join(myDict["file"], file)
                    )
                }
            )
    except TypeError:
        logger.error(f"Data not found at {itr+1} --> {myDict['file']}/01/HDCG_ACC")
        myDict["info_01"]["hdcg_acc"] = None

    # Chest Acceleration Filer
    myDict["data_01"]["chst_acc"] = procFile.filterData(
        data=myDict["instrument"],
        sensorAttachment="CHST",
        sensorLocation="01",
        yUnits="G'S",
    )
    try:
        myDict["info_01"]["chst_acc"] = {}
        for j, file in enumerate(myDict["data_01"]["chst_acc"]["fileName"]):
            myDict["info_01"]["chst_acc"].update(
                {
                    f"{myDict['data_01']['chst_acc']['axis'][j]}": procFile.getSensorData(
                        filePath=os.path.join(myDict["file"], file)
                    )
                }
            )
    except TypeError:
        logger.error(f"Data not found at {itr+1} --> {myDict['file']}/01/CHST_ACC")
        myDict["info_01"]["chst_acc"] = None

    # Chest Acceleration Deflection
    myDict["data_01"]["chst_def"] = procFile.filterData(
        data=myDict["instrument"],
        sensorAttachment="CHST",
        sensorLocation="01",
        yUnits="MM",
    )
    try:
        myDict["info_01"]["chst_def"] = {}
        for j, file in enumerate(myDict["data_01"]["chst_def"]["fileName"]):
            myDict["info_01"]["chst_def"].update(
                {
                    f"{myDict['data_01']['chst_def']['axis'][j]}": procFile.getSensorData(
                        filePath=os.path.join(myDict["file"], file)
                    )
                }
            )
    except TypeError:
        logger.error(f"Data not found at {itr+1} --> {myDict['file']}/01/CHST_DEF")
        myDict["info_01"]["chst_def"] = None

    # Femur Force
    myDict["data_01"]["fmrr_f"] = procFile.filterData(
        data=myDict["instrument"],
        sensorAttachment="FMRR",
        sensorLocation="01",
        yUnits="NWT",
    )
    myDict["data_01"]["fmrl_f"] = procFile.filterData(
        data=myDict["instrument"],
        sensorAttachment="FMRL",
        sensorLocation="01",
        yUnits="NWT",
    )
    try:
        myDict["info_01"]["fmrr_f"] = {}
        for j, file in enumerate(myDict["data_01"]["fmrr_f"]["fileName"]):
            myDict["info_01"]["fmrr_f"].update(
                {
                    f"{myDict['data_01']['fmrr_f']['axis'][j]}": procFile.getSensorData(
                        filePath=os.path.join(myDict["file"], file)
                    )
                }
            )
    except TypeError:
        logger.error(f"Data not found at {itr+1} --> {myDict['file']}/01/FMRR_F")
        myDict["info_01"]["fmrr_f"] = None
    try:
        myDict["info_01"]["fmrl_f"] = {}
        for j, file in enumerate(myDict["data_01"]["fmrl_f"]["fileName"]):
            myDict["info_01"]["fmrl_f"].update(
                {
                    f"{myDict['data_01']['fmrl_f']['axis'][j]}": procFile.getSensorData(
                        filePath=os.path.join(myDict["file"], file)
                    )
                }
            )
    except TypeError:
        logger.error(f"Data not found at {itr+1} --> {myDict['file']}/01/FMRL_F")
        myDict["info_01"]["fmrl_f"] = None

    # Upper Neck Forces
    myDict["data_01"]["neku_f"] = procFile.filterData(
        data=myDict["instrument"],
        sensorAttachment="NEKU",
        sensorLocation="01",
        yUnits="NWT",
    )
    try:
        myDict["info_01"]["neku_f"] = {}
        for j, file in enumerate(myDict["data_01"]["neku_f"]["fileName"]):
            myDict["info_01"]["neku_f"].update(
                {
                    f"{myDict['data_01']['neku_f']['axis'][j]}": procFile.getSensorData(
                        filePath=os.path.join(myDict["file"], file)
                    )
                }
            )
    except TypeError:
        logger.error(f"Data not found at {itr+1} --> {myDict['file']}/01/neku_f")
        myDict["info_01"]["neku_f"] = None

    # Upper Neck Moments
    myDict["data_01"]["neku_m"] = procFile.filterData(
        data=myDict["instrument"],
        sensorAttachment="NEKU",
        sensorLocation="01",
        yUnits="NWM",
    )
    try:
        myDict["info_01"]["neku_m"] = {}
        for j, file in enumerate(myDict["data_01"]["neku_m"]["fileName"]):
            myDict["info_01"]["neku_m"].update(
                {
                    f"{myDict['data_01']['neku_m']['axis'][j]}": procFile.getSensorData(
                        filePath=os.path.join(myDict["file"], file)
                    )
                }
            )
    except TypeError:
        logger.error(f"Data not found at {itr+1} --> {myDict['file']}/01/neku_m")
        myDict["info_01"]["neku_m"] = None

    """============ Left Front Seat Data Filter ============"""
    myDict["data_02"] = {}
    myDict["info_02"] = {}

    # Head CG Acceleration Filer
    myDict["data_02"]["hdcg_acc"] = procFile.filterData(
        data=myDict["instrument"],
        sensorAttachment="HDCG",
        sensorLocation="02",
        yUnits="G'S",
    )
    try:
        myDict["info_02"]["hdcg_acc"] = {}
        for j, file in enumerate(myDict["data_02"]["hdcg_acc"]["fileName"]):
            myDict["info_02"]["hdcg_acc"].update(
                {
                    f"{myDict['data_02']['hdcg_acc']['axis'][j]}": procFile.getSensorData(
                        filePath=os.path.join(myDict["file"], file)
                    )
                }
            )
    except TypeError:
        logger.error(f"Data not found at {itr+1} --> {myDict['file']}/02/HDCG_ACC")
        myDict["info_02"]["hdcg_acc"] = None

    # Chest Acceleration Filer
    myDict["data_02"]["chst_acc"] = procFile.filterData(
        data=myDict["instrument"],
        sensorAttachment="CHST",
        sensorLocation="02",
        yUnits="G'S",
    )
    try:
        myDict["info_02"]["chst_acc"] = {}
        for j, file in enumerate(myDict["data_02"]["chst_acc"]["fileName"]):
            myDict["info_02"]["chst_acc"].update(
                {
                    f"{myDict['data_02']['chst_acc']['axis'][j]}": procFile.getSensorData(
                        filePath=os.path.join(myDict["file"], file)
                    )
                }
            )
    except TypeError:
        logger.error(f"Data not found at {itr+1} --> {myDict['file']}/02/CHST_ACC")
        myDict["info_02"]["chst_acc"] = None

    # Chest Deflection Filer
    myDict["data_02"]["chst_def"] = procFile.filterData(
        data=myDict["instrument"],
        sensorAttachment="CHST",
        sensorLocation="02",
        yUnits="MM",
    )
    try:
        myDict["info_02"]["chst_def"] = {}
        for j, file in enumerate(myDict["data_02"]["chst_def"]["fileName"]):
            myDict["info_02"]["chst_def"].update(
                {
                    f"{myDict['data_02']['chst_def']['axis'][j]}": procFile.getSensorData(
                        filePath=os.path.join(myDict["file"], file)
                    )
                }
            )
    except TypeError:
        logger.error(f"Data not found at {itr+1} --> {myDict['file']}/02/CHST_DEF")
        myDict["info_02"]["chst_def"] = None

    # Femur Force
    myDict["data_02"]["fmrr_f"] = procFile.filterData(
        data=myDict["instrument"],
        sensorAttachment="FMRR",
        sensorLocation="02",
        yUnits="NWT",
    )
    myDict["data_02"]["fmrl_f"] = procFile.filterData(
        data=myDict["instrument"],
        sensorAttachment="FMRL",
        sensorLocation="02",
        yUnits="NWT",
    )
    try:
        myDict["info_02"]["fmrr_f"] = {}
        for j, file in enumerate(myDict["data_02"]["fmrr_f"]["fileName"]):
            myDict["info_02"]["fmrr_f"].update(
                {
                    f"{myDict['data_02']['fmrr_f']['axis'][j]}": procFile.getSensorData(
                        filePath=os.path.join(myDict["file"], file)
                    )
                }
            )
    except TypeError:
        logger.error(f"Data not found at {itr+1} --> {myDict['file']}/02/FMRR_F")
        myDict["info_02"]["fmrr_f"] = None
    try:
        myDict["info_02"]["fmrl_f"] = {}
        for j, file in enumerate(myDict["data_02"]["fmrl_f"]["fileName"]):
            myDict["info_02"]["fmrl_f"].update(
                {
                    f"{myDict['data_02']['fmrl_f']['axis'][j]}": procFile.getSensorData(
                        filePath=os.path.join(myDict["file"], file)
                    )
                }
            )
    except TypeError:
        logger.error(f"Data not found at {itr+1} --> {myDict['file']}/02/FMRL_F")
        myDict["info_02"]["fmrl_f"] = None

    # Upper Neck Forces
    myDict["data_02"]["neku_f"] = procFile.filterData(
        data=myDict["instrument"],
        sensorAttachment="NEKU",
        sensorLocation="02",
        yUnits="NWT",
    )
    try:
        myDict["info_02"]["neku_f"] = {}
        for j, file in enumerate(myDict["data_02"]["neku_f"]["fileName"]):
            myDict["info_02"]["neku_f"].update(
                {
                    f"{myDict['data_02']['neku_f']['axis'][j]}": procFile.getSensorData(
                        filePath=os.path.join(myDict["file"], file)
                    )
                }
            )
    except TypeError:
        logger.error(f"Data not found at {itr+1} --> {myDict['file']}/02/neku_f")
        myDict["info_02"]["neku_f"] = None

    # Upper Neck Moments
    myDict["data_02"]["neku_m"] = procFile.filterData(
        data=myDict["instrument"],
        sensorAttachment="NEKU",
        sensorLocation="02",
        yUnits="NWM",
    )
    try:
        myDict["info_02"]["neku_m"] = {}
        for j, file in enumerate(myDict["data_02"]["neku_m"]["fileName"]):
            myDict["info_02"]["neku_m"].update(
                {
                    f"{myDict['data_02']['neku_m']['axis'][j]}": procFile.getSensorData(
                        filePath=os.path.join(myDict["file"], file)
                    )
                }
            )
    except TypeError:
        logger.error(f"Data not found at {itr+1} --> {myDict['file']}/02/neku_m")
        myDict["info_02"]["neku_m"] = None

handler.close()

"""
*****************************************************************************
============================== Analyzing  Data ==============================
*****************************************************************************
"""

for itr, myDict in enumerate(tqdm(data)):
    """============ Right Front Seat Data Filter ============"""
    myDict["proc_01"] = {}

    try:
        keys = list(myDict["info_01"]["hdcg_acc"].keys())
        myDict["proc_01"]["gsi"] = analysis.calGSI(
            ax=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[0]}"][1]),
            ay=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[1]}"][1]),
            az=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[2]}"][1]),
            t=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[2]}"][0]),
        )
    except:
        myDict["proc_01"]["gsi"] = None

    try:
        myDict["proc_01"]["hic"], myDict["proc_01"]["prob_head"] = analysis.calHIC(
            ax=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[0]}"][1]),
            ay=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[1]}"][1]),
            az=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[2]}"][1]),
            t=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[2]}"][0]),
        )
    except:
        myDict["proc_01"]["hic"] = None
        myDict["proc_01"]["prob_head"] = None

    try:
        myDict["proc_01"]["hic36"], myDict["proc_01"]["prob_head36"] = analysis.calHIC(
            ax=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[0]}"][1]),
            ay=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[1]}"][1]),
            az=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[2]}"][1]),
            t=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[2]}"][0]),
            opt="HIC36",
        )
    except:
        myDict["proc_01"]["hic36"] = None
        myDict["proc_01"]["prob_head36"] = None

    try:
        myDict["proc_01"]["hic15"], myDict["proc_01"]["prob_head15"] = analysis.calHIC(
            ax=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[0]}"][1]),
            ay=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[1]}"][1]),
            az=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[2]}"][1]),
            t=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[2]}"][0]),
            opt="HIC15",
        )
    except:
        myDict["proc_01"]["hic15"] = None
        myDict["proc_01"]["prob_head15"] = None

    try:
        myDict["proc_01"]["ms3h"] = analysis.calc3ms(
            ax=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[0]}"][1]),
            ay=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[1]}"][1]),
            az=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[2]}"][1]),
            t=np.array(myDict["info_01"]["hdcg_acc"][f"{keys[2]}"][0]),
            tol=80,
        )
    except:
        myDict["proc_01"]["ms3h"] = None

    try:
        keys = list(myDict["info_01"]["chst_acc"].keys())
        myDict["proc_01"]["ms3"] = analysis.calc3ms(
            ax=np.array(myDict["info_01"]["chst_acc"][f"{keys[0]}"][1]),
            ay=np.array(myDict["info_01"]["chst_acc"][f"{keys[1]}"][1]),
            az=np.array(myDict["info_01"]["chst_acc"][f"{keys[2]}"][1]),
            t=np.array(myDict["info_01"]["chst_acc"][f"{keys[2]}"][0]),
        )
    except:
        myDict["proc_01"]["ms3"] = None

    try:
        myDict["proc_01"]["csi"] = analysis.calcCSI(
            ax=np.array(myDict["info_01"]["chst_acc"][f"{keys[0]}"][1]),
            ay=np.array(myDict["info_01"]["chst_acc"][f"{keys[1]}"][1]),
            az=np.array(myDict["info_01"]["chst_acc"][f"{keys[2]}"][1]),
            t=np.array(myDict["info_01"]["chst_acc"][f"{keys[2]}"][0]),
        )
    except:
        myDict["proc_01"]["csi"] = None

    try:
        keys = list(myDict["info_01"]["chst_def"].keys())
        myDict["proc_01"]["cti"] = analysis.calcCTI(
            ms3=myDict["proc_01"]["ms3"],
            D=np.array(myDict["info_01"]["chst_def"][f"{keys[0]}"][1]),
        )
    except:
        myDict["proc_01"]["cti"] = None

    try:
        keys_l = list(myDict["info_01"]["fmrl_f"].keys())
        keys_r = list(myDict["info_01"]["fmrr_f"].keys())
        (
            myDict["proc_01"]["fval"],
            myDict["proc_01"]["f_flag"],
            myDict["proc_01"]["f_prob"],
        ) = analysis.calcFemurLoad(
            f=np.average(
                [
                    myDict["info_01"]["fmrl_f"][f"{keys_l[0]}"][1],
                    myDict["info_01"]["fmrr_f"][f"{keys_r[0]}"][1],
                ],
                0,
            )
        )
    except:
        try:
            keys_l = list(myDict["info_01"]["fmrl_f"].keys())
            (
                myDict["proc_01"]["fval"],
                myDict["proc_01"]["f_flag"],
                myDict["proc_01"]["f_prob"],
            ) = analysis.calcFemurLoad(f=myDict["info_01"]["fmrl_f"][f"{keys_l[0]}"][1])
        except:
            try:
                keys_r = list(myDict["info_01"]["fmrr_f"].keys())
                (
                    myDict["proc_01"]["fval"],
                    myDict["proc_01"]["f_flag"],
                    myDict["proc_01"]["f_prob"],
                ) = analysis.calcFemurLoad(
                    f=myDict["info_01"]["fmrr_f"][f"{keys_r[0]}"][1]
                )
            except:
                myDict["proc_01"]["fval"] = None
                myDict["proc_01"]["f_flag"] = None
                myDict["proc_01"]["f_prob"] = None

    try:
        keys_f = list(myDict["info_01"]["neku_f"].keys())
        keys_m = list(myDict["info_01"]["neku_m"].keys())
        (
            myDict["proc_01"]["nfa"],
            myDict["proc_01"]["nea"],
            myDict["proc_01"]["nfp"],
            myDict["proc_01"]["nep"],
        ) = analysis.calcNkm(
            fx=np.array(myDict["info_01"]["neku_f"][f"{keys_f[0]}"][1]),
            my=np.array(myDict["info_01"]["neku_m"][f"{keys_m[1]}"][1]),
            t=np.array(myDict["info_01"]["neku_m"][f"{keys_m[1]}"][0]),
        )
    except:
        myDict["proc_01"]["nfa"] = myDict["proc_01"]["nea"] = myDict["proc_01"][
            "nfp"
        ] = myDict["proc_01"]["nep"] = None

    """============ Left Front Seat Data Filter ============"""
    myDict["proc_02"] = {}

    try:
        keys = list(myDict["info_02"]["hdcg_acc"].keys())
        myDict["proc_02"]["gsi"] = analysis.calGSI(
            ax=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[0]}"][1]),
            ay=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[1]}"][1]),
            az=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[2]}"][1]),
            t=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[2]}"][0]),
        )
    except:
        myDict["proc_02"]["gsi"] = None

    try:
        myDict["proc_02"]["hic"], myDict["proc_02"]["prob_head"] = analysis.calHIC(
            ax=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[0]}"][1]),
            ay=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[1]}"][1]),
            az=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[2]}"][1]),
            t=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[2]}"][0]),
        )
    except:
        myDict["proc_02"]["hic"] = None
        myDict["proc_02"]["prob_head"] = None

    try:
        myDict["proc_02"]["hic36"], myDict["proc_02"]["prob_head36"] = analysis.calHIC(
            ax=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[0]}"][1]),
            ay=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[1]}"][1]),
            az=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[2]}"][1]),
            t=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[2]}"][0]),
            opt="HIC36",
        )
    except:
        myDict["proc_02"]["hic36"] = None
        myDict["proc_02"]["prob_head36"] = None

    try:
        myDict["proc_02"]["hic15"], myDict["proc_02"]["prob_head15"] = analysis.calHIC(
            ax=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[0]}"][1]),
            ay=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[1]}"][1]),
            az=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[2]}"][1]),
            t=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[2]}"][0]),
            opt="HIC15",
        )
    except:
        myDict["proc_02"]["hic15"] = None
        myDict["proc_02"]["prob_head15"] = None

    try:
        myDict["proc_02"]["ms3h"] = analysis.calc3ms(
            ax=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[0]}"][1]),
            ay=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[1]}"][1]),
            az=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[2]}"][1]),
            t=np.array(myDict["info_02"]["hdcg_acc"][f"{keys[2]}"][0]),
            tol=80,
        )
    except:
        myDict["proc_02"]["ms3h"] = None

    try:
        keys = list(myDict["info_02"]["chst_acc"].keys())
        myDict["proc_02"]["ms3"] = analysis.calc3ms(
            ax=np.array(myDict["info_02"]["chst_acc"][f"{keys[0]}"][1]),
            ay=np.array(myDict["info_02"]["chst_acc"][f"{keys[1]}"][1]),
            az=np.array(myDict["info_02"]["chst_acc"][f"{keys[2]}"][1]),
            t=np.array(myDict["info_02"]["chst_acc"][f"{keys[2]}"][0]),
        )
    except:
        myDict["proc_02"]["ms3"] = None

    try:
        myDict["proc_02"]["csi"] = analysis.calcCSI(
            ax=np.array(myDict["info_02"]["chst_acc"][f"{keys[0]}"][1]),
            ay=np.array(myDict["info_02"]["chst_acc"][f"{keys[1]}"][1]),
            az=np.array(myDict["info_02"]["chst_acc"][f"{keys[2]}"][1]),
            t=np.array(myDict["info_02"]["chst_acc"][f"{keys[2]}"][0]),
        )
    except:
        myDict["proc_02"]["csi"] = None

    try:
        keys = list(myDict["info_02"]["chst_def"].keys())
        myDict["proc_02"]["cti"] = analysis.calcCTI(
            ms3=myDict["proc_02"]["ms3"],
            D=np.array(myDict["info_02"]["chst_def"][f"{keys[0]}"][1]),
        )
    except:
        myDict["proc_02"]["cti"] = None

    try:
        keys_l = list(myDict["info_02"]["fmrl_f"].keys())
        keys_r = list(myDict["info_02"]["fmrr_f"].keys())
        (
            myDict["proc_02"]["fval"],
            myDict["proc_02"]["f_flag"],
            myDict["proc_02"]["f_prob"],
        ) = analysis.calcFemurLoad(
            f=np.average(
                [
                    myDict["info_02"]["fmrl_f"][f"{keys_l[0]}"][1],
                    myDict["info_02"]["fmrr_f"][f"{keys_r[0]}"][1],
                ],
                0,
            )
        )
    except:
        try:
            keys_l = list(myDict["info_02"]["fmrl_f"].keys())
            (
                myDict["proc_02"]["fval"],
                myDict["proc_02"]["f_flag"],
                myDict["proc_02"]["f_prob"],
            ) = analysis.calcFemurLoad(f=myDict["info_02"]["fmrl_f"][f"{keys_l[0]}"][1])
        except:
            try:
                keys_r = list(myDict["info_02"]["fmrr_f"].keys())
                (
                    myDict["proc_02"]["fval"],
                    myDict["proc_02"]["f_flag"],
                    myDict["proc_02"]["f_prob"],
                ) = analysis.calcFemurLoad(
                    f=myDict["info_02"]["fmrr_f"][f"{keys_r[0]}"][1]
                )
            except:
                myDict["proc_02"]["fval"] = None
                myDict["proc_02"]["f_flag"] = None
                myDict["proc_02"]["f_prob"] = None

    try:
        keys_f = list(myDict["info_02"]["neku_f"].keys())
        keys_m = list(myDict["info_02"]["neku_m"].keys())
        (
            myDict["proc_02"]["nfa"],
            myDict["proc_02"]["nea"],
            myDict["proc_02"]["nfp"],
            myDict["proc_02"]["nep"],
        ) = analysis.calcNkm(
            fx=np.array(myDict["info_02"]["neku_f"][f"{keys_f[0]}"][1]),
            my=np.array(myDict["info_02"]["neku_m"][f"{keys_m[1]}"][1]),
            t=np.array(myDict["info_02"]["neku_m"][f"{keys_m[1]}"][0]),
        )
    except:
        myDict["proc_02"]["nfa"] = myDict["proc_02"]["nea"] = myDict["proc_02"][
            "nfp"
        ] = myDict["proc_02"]["nep"] = None

"""
*****************************************************************************
============================ Plotting and Saving ============================
*****************************************************************************
"""

if os.path.exists(os.path.join(base_path, "Results")):
    shutil.rmtree(os.path.join(base_path, "Results"))

analysis.plotData(base_path, data, "All")

# arranged = procFile.plotData(data)
# print('\nCars Models:')
# print(arranged)
# print('\n')
# arranged = [i[0] for i in arranged]

# cdata = {}

# for itr, myDict in enumerate(data):
#     nname = myDict['file'].split('_')[-2]
#     if nname in arranged:
#         try:
#             cdata[f'{nname}'].append(
#                 [myDict['date']['year'], myDict['proc_01'], myDict['proc_02']])
#         except:
#             cdata.update(
#                 {f'{nname}': [[myDict['date']['year'], myDict['proc_01'], myDict['proc_02']]]})

# for item in cdata:
#     cdata[f'{item}'].sort(key=lambda x: int(x[0]))

# for item in tqdm(arranged):
#     analysis.plotData(base_path, cdata[f'{item}'], f'{item.title()}', 'line')

print("\nDone!")
"""
==============================================================================
"""
