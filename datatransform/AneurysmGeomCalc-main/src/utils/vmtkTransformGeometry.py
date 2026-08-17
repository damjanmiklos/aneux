#!/usr/bin/env python

# To do rotation for any two vectors or reference systems

from vmtk import pypes
import vmtkUtil
import numpy as np


class TransformGeometry(pypes.pypeScript):
    def __init__(self):
        pypes.pypeScript.__init__(self)
        
        self.Surface = None
        self.Centerlines = None
        self.Source = None
        self.OutputDataType = 'vividict'
        
        self.SourcePoint = None
        self.TargetPoint = [0,0,0]
        self.SourceSystem = None
        self.TargetSystem = [0,0,-1]
        
    
    def check_input_data(self):
        ### Check Output data type
        # sehff = ['vtkPolyData', 'viviDict']
        if not self.OutputDataType == 'vtkPolyData' and not self.OutputDataType == 'vividict':
            self.PrintError("Out data type has to be 'vtkPolyData' or 'vividict' !")
            return
        
        ### Check existance of input geometry
        if not self.Surface and not self.Centerlines:
            self.PrintError("No input geometry found")
            return
        else:
            self.PrintLog("Found input geometry")

        ### Check and conver Surface input
        if self.Surface:
            if not type(self.Surface).__name__ == 'vtkPolyData' and not type(self.Surface).__name__ == 'vividict':
                self.PrintError("Input Surface type has to be 'vtkPolyData' or 'vividict'!")
                return
            elif type(self.Surface).__name__ == 'vtkPolyData':
                self.Surface = vmtkUtil.PolyData_to_Arrays(self.Surface, SupportedDataTypes=[0,2])
        
        ### Check and conver Centerlines input
        if self.Centerlines:
            if not type(self.Centerlines).__name__ == 'vtkPolyData':
                self.PrintError("Input Centerlines type has to be vtkPolyData!")
                return
            else:
                self.Centerlines = vmtkUtil.PolyData_to_Arrays(self.Centerlines, SupportedDataTypes=[3,8])
        
        ### Check Transletion data
        if type(self.SourcePoint) != type(None) and type(self.TargetPoint) != type(None):
            Translate = True
            self.PrintLog("Translate data found")
        else:
            Translate = False
        
        ### Check Rotation data
        if type(self.SourceSystem) != type(None) and type(self.TargetSystem) != type(None):
            Rotate = True
            self.PrintLog("Rotate data found")
        else:
            Rotate = False
            
        return Translate, Rotate
    
    def rotate_matrix(self, matrix, refSystem, targetSystem):         
        ### Angle of two vectors
        def vector_anlge(vec1, vec2):
            vec1norm = vec1/np.linalg.norm(vec1)
            vec2norm = vec2/np.linalg.norm(vec2)
            dot_product = np.dot(vec1norm, vec2norm)
            angle = np.arccos(dot_product)
            return angle
        
        ### Rotate vector to xz plane
        RefNormXZ = np.array([refSystem[0], refSystem[1], 0])
        angle1 = vector_anlge(RefNormXZ, [1,0,0])
        if refSystem[1] > 0: angle1 *= -1
        
        rotM1 = np.array([[np.cos(angle1), -np.sin(angle1), 0],
                          [np.sin(angle1),  np.cos(angle1), 0],
                          [0, 0, 1]])
        matrix = np.array([np.dot(rotM1, point) for point in matrix])
        refSystem = np.array(np.dot(rotM1, refSystem))
        
        ### Rotate Frenet normal to -z
        angle2 = vector_anlge(refSystem, [0,0,-1])
        if refSystem[0] < 0: angle2 *= -1
        
        rotM2 = np.array([[ np.cos(angle2), 0, np.sin(angle2)],
                          [0, 1, 0],
                          [-np.sin(angle2), 0, np.cos(angle2)]])
        matrix = np.array([np.dot(rotM2, point) for point in matrix])
        refSystem = np.array(np.dot(rotM2, refSystem))
        
        ### Rotate Frenet Binormal to y
        # angle3 = vector_anlge(refSystem[1], targetSystem[1])
        # if refSystem[1][2] > 0: angle3 *= -1
        
        # rotM3 = np.array([[1, 0, 0],
        #                   [0, np.cos(angle3), -np.sin(angle3)],                      
        #                   [0, np.sin(angle3),  np.cos(angle3)]])
        # matrix = np.array([np.dot(rotM3, point) for point in matrix])
        # refSystem = np.array([np.dot(rotM3, point) for point in refSystem])
        
        return matrix
    
    def Execute(self):
        ### Check transformation types
        Translate, Rotate = self.check_input_data()
        if not Translate and not Rotate:
            self.PrintError("No Transformation data found!")
            return
        
        ### Transform Surface
        if self.Surface:
            if Translate:
                self.PrintLog("Translating Surface")
                SurfaceShiftVector = np.subtract(self.TargetPoint, self.SourcePoint)
                self.Surface["Points"] = np.array([np.add(row, SurfaceShiftVector) for row in self.Surface["Points"]])
                
            if Rotate:
                self.PrintLog("Rotating Surface")
                self.Surface["Points"] = self.rotate_matrix(self.Surface["Points"], self.SourceSystem, self.TargetSystem)
            
            if self.OutputDataType == 'vtkPolyData':
                self.Surface = vmtkUtil.Arrays_to_PolyData(self.Surface)
                
            self.PrintLog("Surface transformation complete")
        
        if self.Centerlines:
            if Translate:
                self.PrintLog("Translating Centerlines")
                CenterlinesShiftVector = np.subtract(self.TargetPoint, self.SourcePoint)
                self.Centerlines["Points"] = np.array([np.add(row, CenterlinesShiftVector) for row in self.Centerlines["Points"]])
            
            if Rotate:
                self.PrintLog("Rotating Centerlines")
                self.Centerlines["Points"] = self.rotate_matrix(self.Centerlines["Points"], self.SourceSystem, self.TargetSystem)
                
                FrenetVectors = ["FrenetBinormal", "FrenetNormal", "FrenetTangent"]
                CtlPointData = self.Centerlines["PointData"]
                for FVec in FrenetVectors:
                    if FVec in CtlPointData:
                        CtlPointData[FVec] = self.rotate_matrix(CtlPointData[FVec], self.SourceSystem, self.TargetSystem)
                    
                self.Centerlines["PointData"] = CtlPointData
            
            if self.OutputDataType == 'vtkPolyData':
                self.Centerlines = vmtkUtil.Arrays_to_PolyData(self.Centerlines)

            self.PrintLog("Centerlines transformation complete")

                
            