# AneurysmGeomCalc
Script for calculation of aneurysm sack geometric properties

## Dependencies:
1. Windows
2. conda package with vmtk:
```
conda create -n vmtk -c vmtk python=3.6 itk vtk vmtk llvm=3.3
```
3. Paraview 5.10 installed with pvpython alias set up

## Script usage:
```
python ./main.py PATH_TO_SURFACE NUMBER_OF_ANEURYSMS
```
pvpython needs to be used as an alias. Set up using:
My Computer/ Advanced Settings / Environment variables / Path /Edit
```
C:\Program Files\ParaView\bin
```

Neck detection is running on Windows with pvpython alias set up.
To run on Unix please run the following code after aneuryms removal:
```
pvpython neck_detector.py path/aneurysmname_aneurysm_cut.vtp number_of_aneurysm path_to_dataframe.csv
```

TODOs:
- [ ] bifurcation aneurysm handling
- [ ] running pvpyton on Unix with aliases (os.system  and subprocess cannot handle)
- [ ] testing in Unix environment
- [ ] aneurysm reference plane calculation
- [ ] additional metrics
- [ ] code cleanup and refactoring :(
- [X] for constantly reducing aneurysm, the neckplane will be the first valid one
- [X] something is not right during neck plane detection, 40-50 planes area goes to 0, but gives good results anyway.. --> it goes trough entire centerline, but acts only ratio --> wasted performance
- [x] Piccinelli scripts into function
- [x] Multiple aneurysm handling
- [ ] vmtksurfaceconnectivity largest cut selection
- [x] Transpose output csv
- [x] STL from arbitrary directory
- [x] Debug switches but not needed