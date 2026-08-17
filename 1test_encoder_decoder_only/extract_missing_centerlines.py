"""
Parallelized Centerline Extraction
Splits missing patients into N chunks and processes them in parallel.
Uses 8 threads by default.
"""
import os
import sys
import pandas as pd
import subprocess
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from aneux_paths import CSV_PATH, VESSELS_AREA005, CENTERLINES, EXTRA_CENTERLINES

VESSEL_REM_DIR = VESSELS_AREA005
CENTERLINE_IN_DIR = CENTERLINES
CENTERLINE_OUT_DIR = EXTRA_CENTERLINES

NUM_WORKERS = 8 # Parallel processes

def run_vmtk_batch(worker_id, tasks):
    """Worker function to run a batch of tasks in one VMTK process."""
    if not tasks:
        return True

    task_file = f"vmtk_tasks_{worker_id}.json"
    worker_file = f"vmtk_worker_{worker_id}.py"

    with open(task_file, "w") as f:
        json.dump(tasks, f)

    vmtk_worker_code = f"""
import vtk
import numpy as np
import os
import json

os.environ['VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN'] = '1'

from vmtk import vmtkscripts

with open("{task_file}", "r") as f:
    tasks = json.load(f)

for i, task in enumerate(tasks):
    try:
        reader = vmtkscripts.vmtkSurfaceReader()
        reader.InputFileName = task['vessel']
        reader.Execute()

        surface = reader.Surface

        from vmtk import vtkvmtk
        b_extractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
        b_extractor.SetInputData(surface)
        b_extractor.Update()
        bnds = b_extractor.GetOutput()
        num_bnds = bnds.GetNumberOfCells()

        if num_bnds >= 2:
            # PROPER FIX: Cap the surface so the targets lie exactly on the mesh
            capper = vmtkscripts.vmtkSurfaceCapper()
            capper.Surface = surface
            capper.Interactive = 0
            capper.Method = 'centerpoint'
            capper.Execute()
            capped_surface = capper.Surface

            endpoints = []
            locator = vtk.vtkPointLocator()
            locator.SetDataSet(capped_surface)
            locator.BuildLocator()

            for j in range(num_bnds):
                pts = bnds.GetCell(j).GetPoints()
                bary = np.zeros(3)
                for k in range(pts.GetNumberOfPoints()):
                    bary += np.array(pts.GetPoint(k))
                bary /= pts.GetNumberOfPoints()
                
                # Find exact center point of the cap on the surface
                closest_id = locator.FindClosestPoint(bary)
                endpoints.append(capped_surface.GetPoint(closest_id))
            
            endpoints = np.array(endpoints)
            idx_src = np.argmin(endpoints[:, 2])
            
            c = vmtkscripts.vmtkCenterlines()
            c.Surface = capped_surface
            c.SeedSelectorName = 'pointlist'
            c.SourcePoints = endpoints[idx_src].tolist()
            c.TargetPoints = np.delete(endpoints, idx_src, axis=0).flatten().tolist()
            c.AppendEndPoints = 1
            c.Execute()
        else:
            # Closed mesh (no holes). Fallback to surface heuristic.
            npts = surface.GetNumberOfPoints()
            pts = np.zeros((npts, 3))
            for j in range(npts):
                pts[j] = surface.GetPoint(j)

            idx_start = np.argmin(pts[:, 2])
            dists = np.linalg.norm(pts - pts[idx_start], axis=1)
            idx_end = np.argmax(dists)

            c = vmtkscripts.vmtkCenterlines()
            c.Surface = surface
            c.SeedSelectorName = 'pointlist'
            c.SourcePoints = pts[idx_start].tolist()
            c.TargetPoints = pts[idx_end].tolist()
            c.Execute()

        if c.Centerlines.GetNumberOfPoints() > 0:
            writer = vmtkscripts.vmtkSurfaceWriter()
            writer.Surface = c.Centerlines
            writer.OutputFileName = task['output']
            writer.Execute()
            print(f"Worker {worker_id}: Success {{task['id']}}")
        else:
            print(f"Worker {worker_id}: Error (Empty CL) {{task['id']}}")
    except Exception as e:
        print(f"Worker {worker_id}: Exception on {{task['id']}}: {{e}}")
"""
    with open(worker_file, "w") as f:
        f.write(vmtk_worker_code)

    try:
        subprocess.run(["conda", "run", "-n", "vmtk_env", "python", worker_file], check=True)
        return True
    except Exception as e:
        print(f"Worker {worker_id} failed: {e}")
        return False
    finally:
        if os.path.exists(task_file): os.remove(task_file)
        if os.path.exists(worker_file): os.remove(worker_file)

def extract_centerlines():
    # 1. Identify missing files
    df = pd.read_csv(CSV_PATH)
    locations = ['ICA pcom', 'ICA oph', 'ICA cav', 'ICA bif']
    df_filtered = df[df['location'].isin(locations)]
    
    all_tasks = []
    for _, row in df_filtered.iterrows():
        dataset_id = row['dataset']
        if os.path.exists(os.path.join(VESSEL_REM_DIR, f"{dataset_id}.vtp")):
            out_path = os.path.join(CENTERLINE_OUT_DIR, f"{dataset_id}.vtp")
            if not os.path.exists(os.path.join(CENTERLINE_IN_DIR, f"{dataset_id}.vtp")) and not os.path.exists(out_path):
                rem_vessel = os.path.join(VESSEL_REM_DIR, f"{dataset_id}.vtp")
                all_tasks.append({'id': dataset_id, 'vessel': rem_vessel, 'output': os.path.abspath(out_path)})

    if not all_tasks:
        print("No missing centerlines found.")
        return

    print(f"Parallelizing {len(all_tasks)} tasks across {NUM_WORKERS} workers...")

    # 2. Split tasks into chunks
    chunks = np.array_split(all_tasks, NUM_WORKERS)
    
    # 3. Launch parallel processes
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(run_vmtk_batch, i, list(chunk)) for i, chunk in enumerate(chunks)]
        results = [f.result() for f in futures]

    print("\nParallel processing complete.")

if __name__ == "__main__":
    os.makedirs(CENTERLINE_OUT_DIR, exist_ok=True)
    extract_centerlines()
