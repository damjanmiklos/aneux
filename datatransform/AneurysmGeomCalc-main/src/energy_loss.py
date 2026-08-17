from paraview.simple import *
import numpy as np
import pandas as pd
import math
import sys
import os
import glob

import vtk
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData,
    vtkPolyLine
)

from paraview import servermanager as sm
from paraview.vtk.numpy_interface import dataset_adapter as dsa

paraview.simple._DisableFirstRenderCameraReset()
vtk_out = vtk.vtkOutputWindow()
vtk_out.SetInstance(vtk_out)


def main_run(filepath, id1, id2, dfpath):
    file_path = os.path.abspath(filepath)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    directory = os.path.dirname(file_path)

    df = pd.read_csv(dfpath, index_col=0)

    def energyLossPlanes(filename, id1, id2):
        filename = filename.replace('_aneurysm_cut', '')

        id1 = int(id1)
        id2 = int(id2)
        forwardCl = XMLPolyDataReader(registrationName='forwardCl',
                                      FileName=os.path.join(directory, filename + '_forwardcl.vtp'))
        data = dsa.WrapDataObject(sm.Fetch(forwardCl))
        points = data.GetPoints()
        radius = data.PointData['MaximumInscribedSphereRadius']
        pointA = points[id1]
        pointB = points[id2]
        normalA = points[id1 - 1] - points[id1 + 1]
        normalB = points[id2 - 1] - points[id2 + 1]
        radiusA = radius[id1]
        radiusB = radius[id2]

        return pointA, normalA, radiusA, pointB, normalB, radiusB

    pointA, normalA, radiusA, pointB, normalB, radiusB = energyLossPlanes(file_name, id1, id2)

    for i in range(3):
        df.loc['planePointA_' + str(i)] = pointA[i]
    for i in range(3):
        df.loc['planeNormalA_' + str(i)] = normalA[i]
    df.loc['radiusA'] = radiusA

    for i in range(3):
        df.loc['planePointB_' + str(i)] = pointB[i]
    for i in range(3):
        df.loc['planeNormalB_' + str(i)] = normalB[i]
    df.loc['radiusB'] = radiusB
    print(df)
    df.to_csv(dfpath, index=True)


if __name__ == '__main__':
    main_run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
