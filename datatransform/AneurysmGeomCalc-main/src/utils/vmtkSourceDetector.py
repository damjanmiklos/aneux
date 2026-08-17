#!/usr/bin/env python

import vtk
from vmtk import vtkvmtk
from vmtk import pypes
from vmtk import vmtkscripts
import numpy as np
from vmtkUtil import PolyData_to_Arrays
# import vmtkUtil

def calculate_cell_normal(CellPointCoords, NormVec=True):
    vec1 = np.subtract(CellPointCoords[1], CellPointCoords[0])
    vec2 = np.subtract(CellPointCoords[2], CellPointCoords[0])
    normalVec = np.cross(vec1, vec2)
    if NormVec:
        normalVec = normalVec/np.linalg.norm(normalVec)
    return normalVec

class SeedData():
    def __init__(self):
        ### Input Data
        self.CenterIds = None
        
        ### Output Data
        self.CenterCoords = None
        self.SeedCellCoords = None
        self.SeedCellIds = None
        self.SeedNormals = None
        self.InsSphereRadius = None
    
    def GetCenterCoords(self, ind=0):
        return self.CenterCoords[ind]
    
    def GetSeedCellIds(self, ind=0):
        return self.SeedCellIds[ind]
    
    def GetSeedNormals(self, ind=0):
        return self.SeedNormals[ind]
    
    def UpdateSeedCellCoords(self, Surface):
        Surface = PolyData_to_Arrays(Surface)
        self.SeedCellCoords = []
        
        try:
            iterIds = iter(self.CenterIds)
        except:
            self.CenterIds = list([self.CenterIds])
        
        for CellIds in self.SeedCellIds:
            CellCoords = np.zeros((len(CellIds), 3))
            PointIdList = []
            i = 0
            
            for CId in CellIds:
                PointIds = Surface["CellData"]["CellPointIds"][CId]
                for PId in PointIds:
                    
                    if PId not in self.CenterIds:
                        if PId not in PointIdList:
                            PointIdList.append(PId)
                            
                            PointCoords = Surface["Points"][PId]
                            CellCoords[i,:] = PointCoords
                            i += 1
                               
            self.SeedCellCoords.append(CellCoords)
                
    def UpdateCenterCoords(self, Surface):
        self.CenterCoords = []
        self.SeedCellIds = []
        
        try:
            iterIds = iter(self.CenterIds)
        except:
            self.CenterIds = list([self.CenterIds])
            iterIds = iter(self.CenterIds)
        
        for CId in iterIds:
            ### CenterPoint coordinates
            self.CenterCoords.append(Surface["Points"][CId])
            
            ### SeedCellIds
            foundCells = np.where(Surface["CellData"]["CellPointIds"] == CId)
            self.SeedCellIds.append(foundCells[0])
            
        self.CenterCoords = np.array(self.CenterCoords)
        self.SeedCellIds = self.SeedCellIds
    
    def UpdateSeedNormals(self, Surface):
        self.SeedNormals = []
        
        for Seed in range(len(self.CenterIds)):
            CellVecs = []
            for Cell in self.SeedCellIds[Seed]:
                CellPointIds = Surface["CellData"]["CellPointIds"][Cell]
                CellPointCoords = []
                for PointId in CellPointIds:
                    CellPointCoords.append(Surface["Points"][PointId])
                CellVecs.append(calculate_cell_normal(CellPointCoords, NormVec=True))
                
            CellVecs = np.array(CellVecs)
            CellVecsMean = np.mean(CellVecs, axis=0)
            self.SeedNormals.append(CellVecsMean/np.linalg.norm(CellVecsMean))
            
        self.SeedNormals = np.array(self.SeedNormals)
        
    def Update(self, Surface=None):
        ### Check input data
        if not self.CenterIds:
            print("ERROR: CenterIds needed for SeedData calculations!")
            return
        
        if not Surface:
            print("ERROR: Input Surface needed for SeedData calculations!")
            return
        
        SurfaceType = type(Surface).__name__
        if not SurfaceType == 'vtkPolyData' and not SurfaceType == 'vividict':
            print("ERROR: Input Surface type should be vtkPolyData or vividict!")
            return
        
        ### Convert Surface to ArrayDict
        if SurfaceType == 'vtkPolyData':
            print("Converting input Surface to vividict")
            PolyDataToArray = vmtkscripts.vmtkSurfaceToNumpy()
            PolyDataToArray.Surface = Surface
            PolyDataToArray.Execute()
            Surface = PolyDataToArray.ArrayDict
        
        ### Find CenterPoint coordinates and Cell Ids
        self.UpdateCenterCoords(Surface)
            
        ### Calculate Seed Normals
        self.UpdateSeedNormals(Surface)

class SourceDetector(pypes.pypeScript):
    def __init__(self):
        pypes.pypeScript.__init__(self)
        
        ### Input Data
        self.Surface = None
        self.Centerlines = None
        
        self.CapDisplacement = 0.0
        self.CheckNonManifold = 0
        self.CenterlineOutput = True
        self.RecalculateCtl = False
        self.CenterlinesResampleLength = 1
        
        ### Output Data
        self.CapCenterIds = None
        self.CapCenterPoints = None
        self.Source = None
        self.Target = None
        self.OpenProf = True
    
    def calculate_CtlData(self):
        CenterlinesArray = PolyData_to_Arrays(self.Centerlines)       

        CtlEndpointIds = []
        for Ctl in CenterlinesArray["CellData"]["CellPointIds"]:
            CtlEndpointIds.append(Ctl[0])
            CtlEndpointIds.append(Ctl[-1])
        
        CtlEndpointCoords = []
        CtlEndpointRadii = []
        for endPoint in CtlEndpointIds:
            CtlEndpointCoords.append(CenterlinesArray["Points"][endPoint])
            CtlEndpointRadii.append(CenterlinesArray["PointData"]["MaximumInscribedSphereRadius"][endPoint])
        
        _, idx = np.unique(CtlEndpointCoords, return_index=True, axis=0)
        CtlEndpointCoords = np.array(CtlEndpointCoords)
        CtlEndpointCoords = CtlEndpointCoords[np.sort(idx)]
        
        # _, idx = np.unique(CtlEndpointRadii, return_index=True)
        CtlEndpointRadii = np.array(CtlEndpointRadii)
        CtlEndpointRadii = CtlEndpointRadii[np.sort(idx)]
        
        if self.RecalculateCtl == True:
            sortIdList = np.argsort(CtlEndpointRadii)
            sortIdList = sortIdList[::-1]
            CtlEndpointCoords = CtlEndpointCoords[sortIdList]
            CtlEndpointRadii = CtlEndpointRadii[sortIdList]
            
        self.CtlEndpointCoords = CtlEndpointCoords
        self.CtlEndpointRadii = (CtlEndpointRadii)
        
    def Execute(self):
        
        if self.Surface == None:
            self.PrintError('Error: No input surface.')
            
        if self.CheckNonManifold:
            self.PrintLog('NonManifold check.')
            nonManifoldChecker = vmtkscripts.vmtkCenterlines.vmtkNonManifoldSurfaceChecker()
            nonManifoldChecker.Surface = self.Surface
            nonManifoldChecker.PrintError = self.PrintError
            nonManifoldChecker.Execute()
        
            if (nonManifoldChecker.NumberOfNonManifoldEdges > 0):
                self.PrintLog(nonManifoldChecker.Report)
                return
        
        self.PrintLog('Cleaning surface.')
        surfaceCleaner = vtk.vtkCleanPolyData()
        surfaceCleaner.SetInputData(self.Surface)
        surfaceCleaner.Update()
        
        self.PrintLog('Triangulating surface.')
        surfaceTriangulator = vtk.vtkTriangleFilter()
        surfaceTriangulator.SetInputConnection(surfaceCleaner.GetOutputPort())
        surfaceTriangulator.PassLinesOff()
        surfaceTriangulator.PassVertsOff()
        surfaceTriangulator.Update()

        self.PrintLog('Capping surface.')
        surfaceCapper = vtkvmtk.vtkvmtkCapPolyData()
        surfaceCapper.SetInputConnection(surfaceTriangulator.GetOutputPort())
        surfaceCapper.SetDisplacement(self.CapDisplacement)
        surfaceCapper.SetInPlaneDisplacement(self.CapDisplacement)
        surfaceCapper.Update()

        self.Surface = surfaceCapper.GetOutput()
        CappedCenters = surfaceCapper.GetCapCenterIds()
        
        if CappedCenters.GetNumberOfIds() == 0:
            self.PrintLog('Warning: No open profiles found.')
            self.OpenProf = False
            return
        
        self.CapCenterIds = [CappedCenters.GetId(i) for i in range(CappedCenters.GetNumberOfIds())]
        self.CapCenterPoints = np.array([self.Surface.GetPoint(CenterId) for CenterId in self.CapCenterIds])
        
        if self.CenterlineOutput == True:
            
            if not self.Centerlines:
                self.PrintLog('No input Centerlines found: Calculating replacement Centerlines')
                
                replCtl = vmtkscripts.vmtkCenterlines()
                replCtl.Surface = self.Surface
                replCtl.SeedSelectorName = 'idlist'
                replCtl.ResamplingStepLength = self.CenterlinesResampleLength
                replCtl.SourceIds = [self.CapCenterIds[0]]
                replCtl.TargetIds = self.CapCenterIds[1:]
                replCtl.Execute()
                
                self.Centerlines = replCtl.Centerlines
                self.RecalculateCtl = True

            
            self.calculate_CtlData()
        
            sortedCapPoints = []
            sortedCapIds = []
            for endPoint in self.CtlEndpointCoords:
                MinDist = -1
                MinPoint = None
                for i in range(len(self.CapCenterPoints)):
                    capPoint = self.CapCenterPoints[i]
                    PointDist = abs(np.linalg.norm(np.subtract(endPoint, capPoint)))
                    if MinDist == -1 or PointDist < MinDist:
                        MinDist = PointDist
                        MinPoint = capPoint
                        MinId = i
                    
                sortedCapPoints.append(MinPoint)
                sortedCapIds.append(self.CapCenterIds[MinId])
            
            self.CapCenterPoints = np.array(sortedCapPoints)
            self.CapCenterIds = np.array(sortedCapIds)
        
            if len(np.unique(self.CapCenterIds)) != len(self.CapCenterIds):
                self.PrintError("Centerline endpoint to capped surface pairing unsuccessfull")
                return



        self.Source = SeedData()
        self.Source.CenterIds = self.CapCenterIds[0]
        self.Source.Update(self.Surface)
        
        self.Target = SeedData()
        self.Target.CenterIds = list(self.CapCenterIds[1:])
        self.Target.Update(self.Surface)
        
        if self.CenterlineOutput == True:
            self.Source.InsSphereRadius = list([self.CtlEndpointRadii[0]])
            self.Target.InsSphereRadius = list(self.CtlEndpointRadii[1:])
        
            if self.RecalculateCtl == True:
                newCtl = vmtkscripts.vmtkCenterlines()
                newCtl.Surface = self.Surface
                newCtl.SeedSelectorName = 'idlist'
                newCtl.SourceIds = self.Source.CenterIds
                newCtl.TargetIds = self.Target.CenterIds
                newCtl.Execute()
                
                self.Centerlines = newCtl.Centerlines
                self.calculate_CtlData()
        
