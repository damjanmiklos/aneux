# TODO simplification, refactoring
# Run in Paraview 5.10 on MacOS 12.1 using conda

from paraview.simple import *
import numpy as np
import pandas as pd
import math
import sys
import os

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

# TODO file handling needs to be prettier


file_path = os.path.abspath(sys.argv[1])
file_name = str(os.path.splitext(os.path.basename(file_path))[0]).replace('_aneurysm_cut', '')
directory = os.path.dirname(file_path)
aneur_no = int(sys.argv[2])
dfpath = os.path.abspath(sys.argv[3])


def load_geometry(file_path):
    geom = XMLPolyDataReader(registrationName='geomLoader', FileName=[
        file_path])
    geom.TimeArray = 'None'
    UpdatePipeline(time=0.0, proxy=geom)
    return geom


def avgpoints(points):
    avgpoints = np.array([0.0, 0.0, 0.0])
    for x in range(3):
        avgpoints[x] = np.average(points[:, x])
    return avgpoints


def aneurysm_isodistance(vtk_data, clipping_value, resolution):
    # Clipping needed to remove lower parts of the cutted aneurysm
    # thus saving time in the neckplane detection algorythm
    # TODO is this necessary?
    clip1 = Clip(registrationName='Clip1', Input=vtk_data)
    clip1.Scalars = ['POINTS', 'Distance']
    clip1.ClipType = 'Scalar'
    clip1.Value = clipping_value
    UpdatePipeline(time=0.0, proxy=clip1)

    data = dsa.WrapDataObject(sm.Fetch(clip1))
    distances = data.PointData['Distance']
    delta_dist = (np.max(distances) - np.min(distances)) / resolution
    centerline_points = np.zeros((resolution, 3), dtype=float)

    contour1 = Contour(registrationName='Contour1', Input=clip1)
    for i in range(resolution):
        iso_values = np.max(distances) - i * delta_dist
        contour1.ContourBy = ['POINTS', 'Distance']
        contour1.Isosurfaces = [iso_values]
        contour1.PointMergeMethod = 'Uniform Binning'
        UpdatePipeline(time=0.0, proxy=contour1)

        data = dsa.WrapDataObject(sm.Fetch(contour1))
        centerline_points[i] = avgpoints(data.GetPoints())

    # TODO centerline smoothing is needed
    tangent = np.zeros((len(centerline_points), 3), dtype=float)
    '''
    for i in range(len(centerline_points)):
        #TODO centerlinesmoothing workaround
        if i == (len(centerline_points) - 5):
            tangent[i] = tangent[i - 5]
            break
        else:
            tangent[i] = centerline_points[i + 5] - centerline_points[i]
    '''
    # TODO WORKAROUND tangent based on first and last point of centerline
    tang = centerline_points[len(centerline_points) - 1] - centerline_points[0]
    tang = tang / np.linalg.norm(tang)
    for i in range(len(tangent)):
        tangent[i] = tang

    # WORKAROUND frenet style handling
    # TODO numpy array to centerline
    a = np.zeros((len(tangent), 3), dtype=float)
    b = np.zeros((len(tangent), 3), dtype=float)
    for i in range(len(tangent)):
        a[i] = np.cross(np.array([1, 0, 0]), tangent[i])
        b[i] = np.cross(a[i], tangent[i])

    writePolyLine(centerline_points, file_name)

    return centerline_points, tangent, b


# TODO align it with vmtks centerline object (aditional point data needed etc. see numpytovmtk)
def writePolyLine(centerline_points, filename):
    points = vtkPoints()
    for x in range(len(centerline_points)):
        points.InsertNextPoint(centerline_points[x])

    polyLine = vtkPolyLine()
    polyLine.GetPointIds().SetNumberOfIds(len(centerline_points))
    for i in range(0, len(centerline_points)):
        polyLine.GetPointIds().SetId(i, i)

    cells = vtkCellArray()
    cells.InsertNextCell(polyLine)
    polyData = vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(cells)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(os.path.join(directory, filename + '_distance_cl.vtp'))
    writer.SetInputData(polyData)
    writer.Write()


def rotate_vector(theta, u, v):
    c = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    R = math.cos(theta) * np.eye(3) + math.sin(theta) * c + (1 - math.cos(theta)) * np.tensordot(u, np.transpose(u), 0)
    _rot = np.dot(v, R)
    return _rot


def neckplane_detection(geometry, centerline_points, tangents, normal_vectors, alpha, alpha_max, beta, ratio):
    n_o_points = len(centerline_points)
    A = np.ones(n_o_points, dtype=float) * 10000
    normals = np.zeros((n_o_points, 3), dtype=float)
    alpha_rad = alpha * math.pi / 180 #alpha is the scanning resolution
    beta_rad = beta * math.pi / 180 #beta is the scanning resolution

    # creating and connecting filters and calling later for faster execution
    slice1 = Slice(registrationName='Slice1', Input=geometry)
    delaunay2D1 = Delaunay2D(registrationName='Delaunay2D1', Input=slice1)
    meshQuality1 = MeshQuality(registrationName='MeshQuality1', Input=delaunay2D1)
    integrateVariables1 = IntegrateVariables(registrationName='IntegrateVariables1', Input=meshQuality1)
    #going trough aneurysm centerline
    for i in range(int(n_o_points)):  # for i in range(n_o_points):
        #large number for area
        area = 10000.0
        for j in range(int(alpha_max / alpha)):
            for k in range(int(180 / beta)):
                # print('ijk', i,j,k)
                temp = rotate_vector(j * alpha_rad, normal_vectors[i], tangents[i])
                planenormal = rotate_vector(k * beta_rad, tangents[i], temp)
                slice1.SliceType = 'Plane'
                slice1.SliceType.Origin = centerline_points[i]
                slice1.SliceType.Normal = planenormal

                UpdatePipeline(time=0.0, proxy=slice1)

                data = dsa.WrapDataObject(sm.Fetch(slice1))
                try:
                    boolean = (np.min(data.PointData['PointSource']) == 0)
                except ValueError:  # raised if pointsource is empty.
                    pass

                #recreate plane from contour of slice
                if not boolean:
                    delaunay2D1.ProjectionPlaneMode = 'Best-Fitting Plane'
                    delaunay2D1.Tolerance = 0.0

                    UpdatePipeline(time=0.0, proxy=delaunay2D1)

                    meshQuality1.TriangleQualityMeasure = 'Area'
                    meshQuality1.QuadQualityMeasure = 'Area'

                    # cellDatatoPointData1.CellDataArraytoprocess = ['Quality']

                    # UpdatePipeline(time=0.0, proxy=cellDatatoPointData1)

                    UpdatePipeline(time=0.0, proxy=integrateVariables1)

                    data = dsa.WrapDataObject(sm.Fetch(integrateVariables1))

                    if (area > data.CellData['Area']):
                        area = data.CellData['Area']
                        A[i] = area
                        normals[i] = planenormal

                    Delete(delaunay2D1)
                    Delete(meshQuality1)

                    if j == 0: break

                else:
                    pass

        Delete(slice1)

        print(f'Point {i}/{n_o_points} finished with area: {A[i]}')

    # Check the minimum ostium area in the ratio range and return index
    min_index = np.where(A == np.min(A[:int(n_o_points * ratio)]))[0][0]
    if (min_index + 1 == int(n_o_points * ratio)):
        min_index = np.argmax(A < 10000.0)
    sackMaxDiameter = np.sqrt(np.max(A[np.where(A < 10000.0)]) * math.pi / 4)

    slice2 = Slice(registrationName='Slice2', Input=geometry)
    slice2.SliceType = 'Plane'
    slice2.SliceType.Origin = centerline_points[min_index]
    slice2.SliceType.Normal = normals[min_index]

    UpdatePipeline(time=0.0, proxy=slice2)
    delaunay2D2 = Delaunay2D(registrationName='Delaunay2D2', Input=slice2)

    # Save final neckplane normals and slices
    SaveData(
        os.path.join(directory, file_name + '_neckPlane.vtp'),
        proxy=delaunay2D2, PointDataArrays=['Distance', 'Normals_', 'PointSource', 'Scalars_'],
        CellDataArrays=['BadTriangle', 'CellSource', 'Distance', 'FreeEdge'],
        DataMode='Binary',
        CompressorType='LZMA')
    SaveData(
        os.path.join(directory, file_name + '_normalPoints.vtp'),
        proxy=slice2, PointDataArrays=['Distance', 'Normals_', 'PointSource', 'Scalars_'],
        CellDataArrays=['BadTriangle', 'CellSource', 'Distance', 'FreeEdge'],
        DataMode='Binary',
        CompressorType='LZMA')

    global_point = centerline_points[min_index]
    global_normal = normals[min_index] / np.linalg.norm(normals[min_index])
    global_point_no = min_index

    return A, normals, global_point, global_normal, global_point_no, sackMaxDiameter


def ostiumParameters(file_path):
    ostiumSlice = load_geometry(file_path)
    data = dsa.WrapDataObject(sm.Fetch(ostiumSlice))
    origin = avgpoints(data.GetPoints())
    points = data.GetPoints()

    dist = np.zeros(len(points), dtype=float)
    for i in range(len(points)):
        dist[i] = np.linalg.norm(origin - points[i])
    dist_min = np.min(dist)
    dist_max = np.max(dist)
    ratio = dist_min / dist_max

    return dist_min, dist_max, ratio


def sackAreaAndVolume(file_path, neckplane_points, neckplane_normal):
    sackGeom = load_geometry(file_path)
    clip1 = Clip(registrationName='Clip1', Input=sackGeom)
    clip1.ClipType = 'Plane'
    clip1.ClipType.Origin = neckplane_points
    clip1.ClipType.Normal = neckplane_normal
    clip1.Invert = 0
    UpdatePipeline(time=0.0, proxy=clip1)

    meshQuality2 = MeshQuality(registrationName='MeshQuality1', Input=clip1)
    meshQuality2.TriangleQualityMeasure = 'Area'
    meshQuality2.QuadQualityMeasure = 'Area'
    UpdatePipeline(time=0.0, proxy=meshQuality2)

    areaCalc = IntegrateVariables(registrationName='IntegrateVariables1', Input=meshQuality2)
    UpdatePipeline(time=0.0, proxy=clip1)

    data = dsa.WrapDataObject(sm.Fetch(areaCalc))
    area = data.CellData['Area'][0]

    clipClosedSurface1 = ClipClosedSurface(registrationName='ClipClosedSurface1', Input=sackGeom)
    clipClosedSurface1.ClippingPlane = 'Plane'
    clipClosedSurface1.ClippingPlane.Origin = neckplane_points
    clipClosedSurface1.ClippingPlane.Normal = neckplane_normal
    UpdatePipeline(time=0.0, proxy=clipClosedSurface1)

    delaunay3D1 = Delaunay3D(registrationName='Delaunay3D1', Input=clipClosedSurface1)
    UpdatePipeline(time=0.0, proxy=delaunay3D1)
    meshQuality1 = MeshQuality(registrationName='MeshQuality1', Input=delaunay3D1)
    meshQuality1.TetQualityMeasure = 'Volume'
    meshQuality1.HexQualityMeasure = 'Volume'
    UpdatePipeline(time=0.0, proxy=meshQuality1)

    integrateVariables1 = IntegrateVariables(registrationName='IntegrateVariables1', Input=meshQuality1)
    UpdatePipeline(time=0.0, proxy=integrateVariables1)

    SaveData(
        os.path.join(directory, file_name + '_aneurysmSack.vtp'),
        proxy=clipClosedSurface1,
        DataMode='Binary',
        CompressorType='LZMA')

    data = dsa.WrapDataObject(sm.Fetch(integrateVariables1))
    volume = data.CellData['Volume'][0]

    return area, volume


def aspectAndSizeRatio(file_path):
    interpolatedVoronoi = load_geometry(file_path)
    data = dsa.WrapDataObject(sm.Fetch(interpolatedVoronoi))

    centerlineLenght = 0.0
    for i in range(global_point_no, len(centerline_points) - 1):
        centerlineLenght += np.linalg.norm(centerline_points[i] - centerline_points[i + 1])

    aneurysmHeight = np.linalg.norm(centerline_points[global_point_no] - centerline_points[-1])
    aneurysmPerpHeight = np.linalg.norm(np.dot(centerline_points[-1] - global_point, global_normal))
    parentArterySize = np.max(data.PointData['MaximumInscribedSphereRadius'])*2

    ostiumDiameter = math.sqrt(4 * A[global_point_no] / math.pi)

    sphericalVolume = math.pi * sackMaxDiameter ** 3 / 6
    ellipsoidVolume = math.pi * (sackMaxDiameter ** 2) * aneurysmHeight / 6

    return aneurysmHeight, aneurysmPerpHeight, parentArterySize, ostiumDiameter, centerlineLenght, sphericalVolume, ellipsoidVolume


def aneurysmPlane(divergingPointsPath, neckPlanePath, centerPoint):
    divergingPoints = load_geometry(divergingPointsPath)
    data = dsa.WrapDataObject(sm.Fetch(divergingPoints))
    points = data.GetPoints()
    anePlaneA = (points[0] - points[1]) / np.linalg.norm(points[0] - points[1])
    anePlaneNormal = np.cross(anePlaneA, global_normal)

    neckPlane = load_geometry(neckPlanePath)
    slice1 = Slice(registrationName='Slice1', Input=neckPlane)
    slice1.SliceType = 'Plane'
    slice1.SliceType.Origin = centerPoint
    slice1.SliceType.Normal = anePlaneNormal
    UpdatePipeline(time=0.0, proxy=slice1)
    integrateVariables1 = IntegrateVariables(registrationName='IntegrateVariables1', Input=slice1)
    UpdatePipeline(time=0.0, proxy=integrateVariables1)
    data = dsa.WrapDataObject(sm.Fetch(integrateVariables1))
    W = data.CellData['Length']

    return anePlaneNormal, W


geometry = load_geometry(file_path)
print('______Geometry loaded____')

centerline_points, tangents, b = aneurysm_isodistance(geometry, -0.1, 100)  # 100 number of centerline points
print('______Aneurysm centerline created____')

A, normals, global_point, global_normal, global_point_no, sackMaxDiameter = neckplane_detection(geometry,
                                                                                                centerline_points,
                                                                                                tangents, b, 5, 60,
                                                                                                10, 0.4)  # ratio is 0.3
print('______Neckplane detection finished____')

ostiumMin, ostiumMax, ostiumRatio = ostiumParameters(os.path.join(directory, file_name + '_normalPoints.vtp'))
print(f'Ostium ratio: {ostiumRatio} Ostium minimum diameter:{ostiumMin} Ostium maximum diameter:{ostiumMax} ')

sackArea, sackVolume = sackAreaAndVolume(file_path, global_point, global_normal)
print(f'Sack Area: {sackArea} Sack Volume:{sackVolume}')

aneurysmHeight, aneurysmPerpHeight, parentArteryDiameter, ostiumDiameter, centerlineLenght, sphericalVolume, ellipsoidVolume = aspectAndSizeRatio(
    os.path.join(directory, file_name + '_parentartery.vtp'))

anePlaneNormal, aneNeckW = aneurysmPlane(os.path.join(directory, file_name + '_divergingpoints.vtp'),
                                         os.path.join(directory, file_name + '_neckPlane.vtp'), global_point)

# Exporting data
# TODO refactoring into function

df = pd.read_csv(dfpath, index_col=0)
for i in range(3):
    df.loc['ane_' + str(aneur_no) + '_p_' + str(i)] = global_point[i]
for i in range(3):
    df.loc['ane_' + str(aneur_no) + '_n_' + str(i)] = global_normal[i]
df.loc['ane_' + str(aneur_no) + '_A'] = A[global_point_no]
for i in range(3):
    df.loc['ane_' + str(aneur_no) + '_plane_n_' + str(i)] = anePlaneNormal[i]

df.loc['ane_' + str(aneur_no) + '_aneurysm_height'] = aneurysmHeight
df.loc['ane_' + str(aneur_no) + '_aneurysm_perp_height'] = aneurysmPerpHeight
df.loc['ane_' + str(aneur_no) + '_aneurysm_neck_W'] = aneNeckW
df.loc['ane_' + str(aneur_no) + '_parent_artery_d'] = parentArteryDiameter

df.loc['ane_' + str(aneur_no) + '_ostium_min'] = ostiumMin
df.loc['ane_' + str(aneur_no) + '_ostium_max'] = ostiumMax
df.loc['ane_' + str(aneur_no) + '_ostium_d'] = ostiumDiameter
df.loc['ane_' + str(aneur_no) + '_ostium_ratio'] = ostiumRatio

df.loc['ane_' + str(aneur_no) + '_sack_area'] = sackArea
df.loc['ane_' + str(aneur_no) + '_sack_volume'] = sackVolume
df.loc['ane_' + str(aneur_no) + '_sack_slice_max_diameter'] = sackMaxDiameter
df.loc['ane_' + str(aneur_no) + '_spherical_volume'] = sphericalVolume
df.loc['ane_' + str(aneur_no) + '_ellipsoid_volume'] = ellipsoidVolume

df.loc['ane_' + str(aneur_no) + '_sack_aspect_ratio'] = aneurysmHeight / ostiumDiameter
df.loc['ane_' + str(aneur_no) + '_sack_size_ratio'] = aneurysmHeight / parentArteryDiameter
df.loc['ane_' + str(aneur_no) + '_sack_bottleneck_factor'] = sackMaxDiameter / ostiumDiameter

df.to_csv(dfpath, index=True)

print('______CSV saved____')
