import os
import glob
import pandas as pd
import vtk
import argparse

from vmtk import vmtkscripts, vmtkrenderer
from src.utils import vmtkUtil
from src.Piccinelli import patchandinterpolatecenterlines, clipvoronoidiagram, paralleltransportvoronoidiagram

debugSwitch = False


def generate_centerline(Surface, SeedSelectorName='pickpoint', SourceIds=None, TargetIds=None, endpoints=False,
                        obj=False):
    myCenterlines = vmtkscripts.vmtkCenterlines()
    myCenterlines.Surface = Surface
    myCenterlines.SeedSelectorName = SeedSelectorName
    myCenterlines.AppendEndPoints = endpoints
    myCenterlines.Execute()

    if obj:
        return myCenterlines
    else:
        return myCenterlines.Centerlines


def create_voronoi(surface, voronoi_out_filename):
    voronoi = vmtkscripts.vmtkDelaunayVoronoi()
    voronoi.Surface = surface
    voronoi.Execute()
    vmtkUtil.write_surface(voronoi.Surface, voronoi_out_filename)
    return voronoi


def create_centerline(surface, cl_name):
    inCenterlines = generate_centerline(surface, SeedSelectorName='pickpoint', obj=False, endpoints=True)
    myResampling = vmtkscripts.vmtkCenterlineResampling()
    myResampling.Centerlines = inCenterlines
    myResampling.Length = 0.1
    myResampling.Execute()
    vmtkUtil.write_surface(myResampling.Centerlines, cl_name)
    return myResampling


# TODO: hide renderer window after selection
def pick_IO(directory, file_name):
    surface = vmtkUtil.read_surface(os.path.join(directory, file_name + '_forwardcl.vtp'))

    computer = utils.seedSelector.vmtkPickPointSeedSelector()
    computer.vmtkRenderer = vmtkrenderer.vmtkRenderer()
    computer.vmtkRenderer.Initialize()
    computer.SetSurface(surface)
    computer.Execute()

    return computer.GetSourceSeedIds().GetId(0), computer.GetTargetSeedIds().GetId(0)


def surface_boolean(directory, file_name, aneur_no):
    surfA = vmtkUtil.read_surface(os.path.join(directory, file_name + '_' + str(aneur_no) + '_aneurysmsurface.vtp'))
    surfB = vmtkUtil.read_surface(os.path.join(directory, file_name + '_' + str(aneur_no) + '_reconstructedmodel.vtp'))

    surfaceBoolean = vmtkscripts.vmtkSurfaceBooleanOperation()
    surfaceBoolean.Surface = surfB
    surfaceBoolean.Surface2 = surfA
    surfaceBoolean.Tolerance = 1E-2
    surfaceBoolean.Operation = 'difference'
    surfaceBoolean.Execute()

    boolean_viewer = vmtkscripts.vmtkSurfaceViewer()
    boolean_viewer.Surface = surfaceBoolean.Surface
    boolean_viewer.Execute()

    vmtkUtil.write_surface(surfaceBoolean.Surface,
                           os.path.join(directory, file_name + '_' + str(aneur_no) + '_aneurysm_cut.vtp'))



def artery_centerline(directory, outdir, file_name, file_ext, aneur_no):
    surface = vmtkUtil.read_surface(os.path.join(directory, file_name + file_ext))
    create_voronoi(surface, os.path.join(outdir, str(aneur_no), file_name + '_' + str(aneur_no) + '_voronoi.vtp'))
    print('_____Voronoi diagram created_____')
    print('_____Forward centerline_____')
    print('Click to inlet, press Q, click on top of the aneurysm, click to the outlet, press Q')
    create_centerline(surface,
                      os.path.join(outdir, str(aneur_no), file_name + '_' + str(aneur_no) + '_forwardcl.vtp'))
    print('_____Forward centerline created_____')
    print('_____Backward centerline_____')
    print('Click to outlet, press Q, click on top of the aneurysm, click to the inlet, press Q')
    create_centerline(surface,
                      os.path.join(outdir, str(aneur_no), file_name + '_' + str(aneur_no) + '_backwardcl.vtp'))
    print('_____Backward centerline created_____')

def cut_aneurysm(directory, file_name, aneurysmType, aneur_no):
    vtk_out = vtk.vtkOutputWindow()
    vtk_out.SetInstance(vtk_out)

    if aneurysmType == 'lateral':
        patchandinterpolatecenterlines.patchandInterpolateCenterlines(directory, file_name, 'lateral', aneur_no)
        print('_____Lateral aneurysm process started_____')
        clipvoronoidiagram.clipVoronoiDiagram(directory, file_name, aneur_no)
        print('_____Voronoi diagram clipped_____')
        paralleltransportvoronoidiagram.parallelTransportVoronoiDiagram(directory, file_name, aneur_no)
        print('_____Artery patched_____')
    elif aneurysmType == 'bifurcation':
        print('Bifurcation type aneurysm removal not implemented currently. Exiting.')
        exit()
    else:
        print('wrong aneurysm type')

    print('_____Centerline and patch created_____')

def neck_detection(directory, file_name, aneur_no, dfpath):
    # TODO: cross-platform terminal command run
    
    command = 'pvpython ' + os.path.join('src','neck_detector.py ') + os.path.join(directory, file_name + '_' + str(
        aneur_no) + '_aneurysm_cut.vtp') + ' ' + str(aneur_no) + ' ' + str(dfpath)
    
    if os.name == 'posix':
        print('Please use pvpython neck_detector.py PATH_TO_aneurysm_cut.vtp NO_OF_ANEURYSM PATH_TO_CSV')
        # Methods that are not working:
        #os.system(command)
        # print('Neck detector running on MacOSX')
        # subprocess.run(['/bin/bash', '-i', '-c', 'pvpython ' + 'neck_detector.py ' + os.path.join(directory, file_name + '_aneurysm_cut.vtp')], shell=True, check=True, stdout=sys.stdout, stderr=subprocess.STDOUT)
        # asdf=subprocess.run(['/bin/bash', '-i', '-c', command], shell=True, check = True, stdout = asdf.PIPE)

        # Popen(['/bin/bash', '-i', '-c', command], shell=True)

        # def execute(command):
        #    subprocess.check_call(['-i',command], shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)

        # def execute(command):
        # subprocess.check_call(['/bin/bash', '-i', '-c', 'ls -a'], shell=True)

        # execute(command)
        # p = subprocess.Popen(['shopt -s expand_aliases','echo asdf'],shell=True, stdout=sys.stdout, stderr=sys.stderr, executable='/bin/bash')
        # print (p.communicate('n\n')[0])

    elif os.name == 'nt':
        print('Neck detector running on Windows with pvpython alias')
        os.system(command)


def energy_loss(directory, file_name, run_mode, id1, id2, aneur_no, dfpath):
    # TODO: cross-platform terminal command run

    command = 'pvpython ' + os.path.join('src','energy_loss.py ') + os.path.join(directory, file_name + '_' + str(
        aneur_no) + '_aneurysm_cut.vtp') + ' ' + str(
        id1) + ' ' + str(id2) + ' ' + str(dfpath)
    print(command)
    if os.name == 'posix':
        print('Please use pvpython on UNIX for energy loss calculation')

    elif os.name == 'nt':
        print('Energy loss plane definition running on Windows with pvpython alias')
        os.system(command)


def main_run(file, mode, aneurysm_no):
    aneurysm_type = 'lateral'
    paraview_mode = 'pvpython'

    vtk_out = vtk.vtkOutputWindow()
    vtk_out.SetInstance(vtk_out)

    file_path = os.path.abspath(file)
    aneurysm_name, file_ext = os.path.splitext(os.path.basename(file_path))
    directory = os.path.dirname(file_path)
    output_directory = os.path.join(directory, 'input')
    print('file_name', aneurysm_name)
    os.makedirs(output_directory, exist_ok=True)

    database = pd.DataFrame(index=[0])
    database['aneurysm_no'] = int(aneurysm_no)
    database = database.T
    database_path = os.path.join(output_directory, aneurysm_name + '_data.csv')
    database.to_csv(database_path, index=True, header=False)

    if mode == 'manual':
        for i in range(int(aneurysm_no)):
            os.makedirs(os.path.join(output_directory, str(i)), exist_ok=True)
            artery_centerline(directory, output_directory, aneurysm_name, file_ext, i)

            cut_aneurysm(os.path.join(output_directory, str(i)), aneurysm_name, aneurysm_type, i)
            surface_boolean(os.path.join(output_directory, str(i)), aneurysm_name, i)
            neck_detection(os.path.join(output_directory, str(i)), aneurysm_name, i, database_path)

            #id1, id2 = pick_IO(os.path.join(output_directory, '0'), aneurysm_name + '_0')
            #energy_loss(os.path.join(output_directory, str(i)), aneurysm_name, id1, id2, i, database_path)

        database = pd.read_csv(glob.glob(os.path.join(directory, 'input', '*_data*'))[0], index_col=0, header=None)

    elif mode == 'dev':
        pass

    print('Postprocessing finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('d', default='./', help='Path to directory', type=str)
    parser.add_argument('mode', help='Mode: auto, manual', type=str)
    parser.add_argument('no', help='Number of aneurysms', type=int)
    args = parser.parse_args()

    main_run(file=os.path.abspath(args.d),mode =args.mode,  aneurysm_no=args.no)