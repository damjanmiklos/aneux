### To Do: Change supported data type recognition

from vmtk import vmtkscripts
# import numpy as np

### Convert vtkPolyData to Numpy array tuple
def PolyData_to_Arrays(PolyData, SupportedDataTypes=[0,2,3,8]):
    NumberOfPointDataArrays = PolyData.GetPointData().GetNumberOfArrays()
    if NumberOfPointDataArrays not in SupportedDataTypes:
        print('ERROR: Unsupported input data type!')
        return
    
    if NumberOfPointDataArrays == 0 or NumberOfPointDataArrays == 2: ### Surface data contains 0 PointData arrays
        myPolyToArray = vmtkscripts.vmtkSurfaceToNumpy()
        myPolyToArray.Surface = PolyData
    elif NumberOfPointDataArrays == 3 or NumberOfPointDataArrays == 8: ### Centerlines data contains 3 PointData arrays
        myPolyToArray = vmtkscripts.vmtkCenterlinesToNumpy()
        myPolyToArray.Centerlines = PolyData
    else:
        print('ERROR: Unsupported input data type!')
        return
    
    myPolyToArray.Execute()
    
    return myPolyToArray.ArrayDict

### Convert Numpy array tuple to vtkPolyData
def Arrays_to_PolyData(ArrayDict):
    NumberOfPointDataArrays = len(ArrayDict["PointData"])
    
    if NumberOfPointDataArrays == 0 or NumberOfPointDataArrays == 2: ### Surface data contains 0 PointData arrays
        myArrayToPoly = vmtkscripts.vmtkNumpyToSurface()
        myArrayToPoly.ArrayDict = ArrayDict
        myArrayToPoly.Execute()
        return myArrayToPoly.Surface
    elif NumberOfPointDataArrays == 3 or NumberOfPointDataArrays == 8: ### Centerlines data contains 3 PointData arrays
        myArrayToPoly = vmtkscripts.vmtkNumpyToCenterlines()
        myArrayToPoly.ArrayDict = ArrayDict
        myArrayToPoly.Execute()
        return myArrayToPoly.Centerlines
    else:
        print('ERROR: Unsupported input data type!')
        return

### View_Surface
def view_surface(Surface):
    if type(Surface).__name__ == "vividict":
        Surface = Arrays_to_PolyData(Surface)
    
    mySurfaceVr = vmtkscripts.vmtkSurfaceViewer()
    mySurfaceVr.Surface = Surface
    mySurfaceVr.Execute()
    return

### Read Surface from file
def read_surface(FileLoc):
    mySurfaceReader = vmtkscripts.vmtkSurfaceReader()
    mySurfaceReader.InputFileName = FileLoc
    mySurfaceReader.Execute()
    return mySurfaceReader.Surface

def write_surface(Surface, OutputFileName):
    if type(Surface).__name__ == 'vividict':
        Surface = Arrays_to_PolyData(Surface)
    
    mySurfaceWriter = vmtkscripts.vmtkSurfaceWriter()
    mySurfaceWriter.Surface = Surface
    mySurfaceWriter.OutputFileName = OutputFileName
    mySurfaceWriter.Execute()
    