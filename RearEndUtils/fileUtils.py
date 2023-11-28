import os
import copy
from collections import Counter


def stripList(myList):
    """
    Strip a list having strings

    Args:
        myList (list<str>): List of strings

    Returns:
        myList (list<str>): List of strings
    """

    for i in range(len(myList)):
        myList[i] = myList[i].strip()

    return myList


def genFileName(serial, fileNum):
    """
    Generate file name by padding zeros with file number

    Args:
        serial  (str): Vehicle serial number from EV5
        fileNum (str): File number from EV5

    Returns:
        fileName (str): File name
    """

    fileNum = fileNum.zfill(3)
    fileName = serial + "." + fileNum

    return fileName


def getFileInfo(dirPath):
    """
    Decode data files

    Args:
        dirPath (str): Directory having data files

    Returns:
        data (dict): Dictionary having files information
            |-data['date']
            |---|---date['day']
            |   |---date['month']
            |   |---date['year']
            |
            |-data['vehicle']
            |---|---vehicle['serial']
            |   |---vehicle['year']
            |   |---vehicle['make']
            |   |---vehicle['model']
            |
            |-data['instrument']
            |---|---instrument['vehicleNumber']
            |   |---instrument['fileName']
            |   |---instrument['sensorLocation']
            |   |---instrument['sensorType']
            |   |---instrument['sensorAttachment']
            |   |---instrument['axis']
            |   |---instrument['xUnits']
            |   |---instrument['yUnits']
    """

    files = os.listdir(dirPath)

    for i in files:
        if i.endswith(".EV5"):
            ev5File = i

    with open(os.path.join(dirPath, ev5File), "r") as file:
        lines = file.readlines()

    data = {}
    chkPts = []

    for i, line in enumerate(lines):
        if (
            ("--- TEST ---" in line)
            or ("--- VEHICLE ---" in line)
            or ("--- OCCUPANT ---" in line)
            or ("--- RESTRAINT ---" in line)
            or ("--- INSTRUMENTATION ---" in line)
            or ("--- END ---" in line)
        ):
            chkPts.append(i)

    # Extracting file name
    data["file"] = dirPath

    # Extracting date
    date = {}
    temp = lines[chkPts[0] + 1 : chkPts[1]]
    for i, line in enumerate(temp):
        tempS = stripList(line.split("|"))
        for j in tempS:
            if "/" in j and (not len(j) > 12 and not len(j) < 10):
                tempSS = j.split("/")
                date["day"] = tempSS[0]
                date["month"] = tempSS[1]
                date["year"] = tempSS[2]
                data["date"] = date

    # Extracting vehicle information
    vehicle = {}
    temp = os.path.split(dirPath)
    temp = stripList(temp[1].split("_"))
    vehicle["serial"] = temp[0]
    vehicle["year"] = temp[1]
    vehicle["make"] = temp[2]
    vehicle["model"] = temp[3]
    data["vehicle"] = vehicle

    # Extracting instrumentation data
    instrument = {}
    vehicleNumber = []
    fileName = []
    sensorLocation = []
    sensorType = []
    sensorAttachment = []
    axis = []
    xUnits = []
    yUnits = []
    temp = lines[chkPts[-2] + 1 : chkPts[-1]]
    for i, line in enumerate(temp):
        tempS = stripList(line.split("|")[:8])
        vehicleNumber.append(tempS[0])
        fileName.append(genFileName(data["vehicle"]["serial"], tempS[1]))
        sensorType.append(tempS[2])
        sensorLocation.append(tempS[3])
        sensorAttachment.append(tempS[4])
        axis.append(tempS[5])
        xUnits.append(tempS[6])
        yUnits.append(tempS[7])
    instrument["vehicleNumber"] = vehicleNumber
    instrument["fileName"] = fileName
    instrument["sensorLocation"] = sensorLocation
    instrument["sensorType"] = sensorType
    instrument["sensorAttachment"] = sensorAttachment
    instrument["axis"] = axis
    instrument["xUnits"] = xUnits
    instrument["yUnits"] = yUnits
    data["instrument"] = instrument

    return data


def filterData(
    data,
    vehicleNumber=None,
    sensorLocation=None,
    sensorType=None,
    sensorAttachment=None,
    axis=None,
    xUnits=None,
    yUnits=None,
):
    """
    Filter data for input parameters

    Args:
        data             (dict): Data files information
            |-data['instrument']
            |---|---instrument['vehicleNumber']
            |   |---instrument['fileName']
            |   |---instrument['sensorLocation']
            |   |---instrument['sensorType']
            |   |---instrument['sensorAttachment']
            |   |---instrument['axis']
            |   |---instrument['xUnits']
            |   |---instrument['yUnits']

        sensorType       (str) : Sensor type
        vehicleNumber    (str) : Vehicle number [optional]
        sensorLocation   (str) : Sensor location [optional]
        sensorAttachment (str) : Sensor attachment [optional]
        axis             (str) : Axis [optional]

    Returns:
        data (dict): Filtered data
            |-data['instrument']
            |---|---instrument['vehicleNumber']
            |   |---instrument['fileName']
            |   |---instrument['sensorLocation']
            |   |---instrument['sensorType']
            |   |---instrument['sensorAttachment']
            |   |---instrument['axis']
            |   |---instrument['xUnits']
            |   |---instrument['yUnits']
    """
    data = copy.deepcopy(data)
    fileNameList = data["fileName"]
    vehicleNumberList = data["vehicleNumber"]
    sensorLocationList = data["sensorLocation"]
    sensorTypeList = data["sensorType"]
    sensorAttachmentList = data["sensorAttachment"]
    axisList = data["axis"]
    xUnitsList = data["xUnits"]
    yUnitsList = data["yUnits"]

    if not sensorType == None:
        fileNameFilteredList = []
        vehicleNumberFilteredList = []
        sensorLocationFilteredList = []
        sensorTypeFilteredList = []
        sensorAttachmentFilteredList = []
        axisFilteredList = []
        xUnitsFilteredList = []
        yUnitsFilteredList = []

        for itr, value in enumerate(sensorTypeList):
            if value == sensorType:
                fileNameFilteredList.append(fileNameList[itr])
                vehicleNumberFilteredList.append(vehicleNumberList[itr])
                sensorLocationFilteredList.append(sensorLocationList[itr])
                sensorTypeFilteredList.append(sensorTypeList[itr])
                sensorAttachmentFilteredList.append(sensorAttachmentList[itr])
                axisFilteredList.append(axisList[itr])
                xUnitsFilteredList.append(xUnitsList[itr])
                yUnitsFilteredList.append(yUnitsList[itr])

        fileNameList = fileNameFilteredList
        vehicleNumberList = vehicleNumberFilteredList
        sensorLocationList = sensorLocationFilteredList
        sensorTypeList = sensorTypeFilteredList
        sensorAttachmentList = sensorAttachmentFilteredList
        axisList = axisFilteredList
        xUnitsList = xUnitsFilteredList
        yUnitsList = yUnitsFilteredList

    if not vehicleNumber == None:
        fileNameFilteredList = []
        vehicleNumberFilteredList = []
        sensorLocationFilteredList = []
        sensorTypeFilteredList = []
        sensorAttachmentFilteredList = []
        axisFilteredList = []
        xUnitsFilteredList = []
        yUnitsFilteredList = []

        for itr, value in enumerate(vehicleNumberList):
            if value == vehicleNumber:
                fileNameFilteredList.append(fileNameList[itr])
                vehicleNumberFilteredList.append(vehicleNumberList[itr])
                sensorLocationFilteredList.append(sensorLocationList[itr])
                sensorTypeFilteredList.append(sensorTypeList[itr])
                sensorAttachmentFilteredList.append(sensorAttachmentList[itr])
                axisFilteredList.append(axisList[itr])
                xUnitsFilteredList.append(xUnitsList[itr])
                yUnitsFilteredList.append(yUnitsList[itr])

        fileNameList = fileNameFilteredList
        vehicleNumberList = vehicleNumberFilteredList
        sensorLocationList = sensorLocationFilteredList
        sensorTypeList = sensorTypeFilteredList
        sensorAttachmentList = sensorAttachmentFilteredList
        axisList = axisFilteredList
        xUnitsList = xUnitsFilteredList
        yUnitsList = yUnitsFilteredList

    if not sensorLocation == None:
        fileNameFilteredList = []
        vehicleNumberFilteredList = []
        sensorLocationFilteredList = []
        sensorTypeFilteredList = []
        sensorAttachmentFilteredList = []
        axisFilteredList = []
        xUnitsFilteredList = []
        yUnitsFilteredList = []

        for itr, value in enumerate(sensorLocationList):
            if value == sensorLocation:
                fileNameFilteredList.append(fileNameList[itr])
                vehicleNumberFilteredList.append(vehicleNumberList[itr])
                sensorLocationFilteredList.append(sensorLocationList[itr])
                sensorTypeFilteredList.append(sensorTypeList[itr])
                sensorAttachmentFilteredList.append(sensorAttachmentList[itr])
                axisFilteredList.append(axisList[itr])
                xUnitsFilteredList.append(xUnitsList[itr])
                yUnitsFilteredList.append(yUnitsList[itr])

        fileNameList = fileNameFilteredList
        vehicleNumberList = vehicleNumberFilteredList
        sensorLocationList = sensorLocationFilteredList
        sensorTypeList = sensorTypeFilteredList
        sensorAttachmentList = sensorAttachmentFilteredList
        axisList = axisFilteredList
        xUnitsList = xUnitsFilteredList
        yUnitsList = yUnitsFilteredList

    if not sensorAttachment == None:
        fileNameFilteredList = []
        vehicleNumberFilteredList = []
        sensorLocationFilteredList = []
        sensorTypeFilteredList = []
        sensorAttachmentFilteredList = []
        axisFilteredList = []
        xUnitsFilteredList = []
        yUnitsFilteredList = []

        for itr, value in enumerate(sensorAttachmentList):
            if value == sensorAttachment:
                fileNameFilteredList.append(fileNameList[itr])
                vehicleNumberFilteredList.append(vehicleNumberList[itr])
                sensorLocationFilteredList.append(sensorLocationList[itr])
                sensorTypeFilteredList.append(sensorTypeList[itr])
                sensorAttachmentFilteredList.append(sensorAttachmentList[itr])
                axisFilteredList.append(axisList[itr])
                xUnitsFilteredList.append(xUnitsList[itr])
                yUnitsFilteredList.append(yUnitsList[itr])

        fileNameList = fileNameFilteredList
        vehicleNumberList = vehicleNumberFilteredList
        sensorLocationList = sensorLocationFilteredList
        sensorTypeList = sensorTypeFilteredList
        sensorAttachmentList = sensorAttachmentFilteredList
        axisList = axisFilteredList
        xUnitsList = xUnitsFilteredList
        yUnitsList = yUnitsFilteredList

    if not axis == None:
        fileNameFilteredList = []
        vehicleNumberFilteredList = []
        sensorLocationFilteredList = []
        sensorTypeFilteredList = []
        sensorAttachmentFilteredList = []
        axisFilteredList = []
        xUnitsFilteredList = []
        yUnitsFilteredList = []

        for itr, value in enumerate(axisList):
            if value == axis:
                fileNameFilteredList.append(fileNameList[itr])
                vehicleNumberFilteredList.append(vehicleNumberList[itr])
                sensorLocationFilteredList.append(sensorLocationList[itr])
                sensorTypeFilteredList.append(sensorTypeList[itr])
                sensorAttachmentFilteredList.append(sensorAttachmentList[itr])
                axisFilteredList.append(axisList[itr])
                xUnitsFilteredList.append(xUnitsList[itr])
                yUnitsFilteredList.append(yUnitsList[itr])

        fileNameList = fileNameFilteredList
        vehicleNumberList = vehicleNumberFilteredList
        sensorLocationList = sensorLocationFilteredList
        sensorTypeList = sensorTypeFilteredList
        sensorAttachmentList = sensorAttachmentFilteredList
        axisList = axisFilteredList
        xUnitsList = xUnitsFilteredList
        yUnitsList = yUnitsFilteredList

    if not xUnits == None:
        fileNameFilteredList = []
        vehicleNumberFilteredList = []
        sensorLocationFilteredList = []
        sensorTypeFilteredList = []
        sensorAttachmentFilteredList = []
        axisFilteredList = []
        xUnitsFilteredList = []
        yUnitsFilteredList = []

        for itr, value in enumerate(xUnitsList):
            if value == xUnits:
                fileNameFilteredList.append(fileNameList[itr])
                vehicleNumberFilteredList.append(vehicleNumberList[itr])
                sensorLocationFilteredList.append(sensorLocationList[itr])
                sensorTypeFilteredList.append(sensorTypeList[itr])
                sensorAttachmentFilteredList.append(sensorAttachmentList[itr])
                axisFilteredList.append(axisList[itr])
                xUnitsFilteredList.append(xUnitsList[itr])
                yUnitsFilteredList.append(yUnitsList[itr])

        fileNameList = fileNameFilteredList
        vehicleNumberList = vehicleNumberFilteredList
        sensorLocationList = sensorLocationFilteredList
        sensorTypeList = sensorTypeFilteredList
        sensorAttachmentList = sensorAttachmentFilteredList
        axisList = axisFilteredList
        xUnitsList = xUnitsFilteredList
        yUnitsList = yUnitsFilteredList

    if not yUnits == None:
        fileNameFilteredList = []
        vehicleNumberFilteredList = []
        sensorLocationFilteredList = []
        sensorTypeFilteredList = []
        sensorAttachmentFilteredList = []
        axisFilteredList = []
        xUnitsFilteredList = []
        yUnitsFilteredList = []

        for itr, value in enumerate(yUnitsList):
            if value == yUnits:
                fileNameFilteredList.append(fileNameList[itr])
                vehicleNumberFilteredList.append(vehicleNumberList[itr])
                sensorLocationFilteredList.append(sensorLocationList[itr])
                sensorTypeFilteredList.append(sensorTypeList[itr])
                sensorAttachmentFilteredList.append(sensorAttachmentList[itr])
                axisFilteredList.append(axisList[itr])
                xUnitsFilteredList.append(xUnitsList[itr])
                yUnitsFilteredList.append(yUnitsList[itr])

        fileNameList = fileNameFilteredList
        vehicleNumberList = vehicleNumberFilteredList
        sensorLocationList = sensorLocationFilteredList
        sensorTypeList = sensorTypeFilteredList
        sensorAttachmentList = sensorAttachmentFilteredList
        axisList = axisFilteredList
        xUnitsList = xUnitsFilteredList
        yUnitsList = yUnitsFilteredList

    data["fileName"] = fileNameList
    data["vehicleNumber"] = vehicleNumberList
    data["sensorLocation"] = sensorLocationList
    data["sensorType"] = sensorTypeList
    data["sensorAttachment"] = sensorAttachmentList
    data["axis"] = axisList
    data["xUnits"] = xUnitsList
    data["yUnits"] = yUnitsList

    if bool(fileNameList):
        return data
    else:
        return None


def getSensorData(filePath):
    with open(filePath, "r") as file:
        x_values = []
        y_values = []

        for line in file:
            parts = line.strip().split("\t")

            if len(parts) == 2:
                x_value, y_value = parts
                x_values.append(float(x_value))
                y_values.append(float(y_value))

    return [x_values, y_values]


def getArranged(ddata):
    models = []
    for itr, myDict in enumerate(ddata):
        temp = myDict["file"].split("_")
        models.append(temp[-2])
    counts = Counter(models)
    count_list = [[char, count] for char, count in counts.items()]
    count_list.sort(key=lambda x: x[1], reverse=True)
    result = [[char, count] for char, count in count_list if count >= 5]

    return result
