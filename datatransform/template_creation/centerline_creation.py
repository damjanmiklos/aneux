import os
import sys
os.environ["VTK_OFFSCREEN"] = "1"
os.environ["EGL_PLATFORM"] = "surfaceless"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VTK_NUMBER_OF_THREADS"] = "1"

import argparse
import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from tqdm import tqdm

try:
    from vmtk import vmtkscripts
    from vmtk import vtkvmtk
    HAS_VMTK = True
except ImportError:
    HAS_VMTK = False
    print("Warning: VMTK Python bindings not found in default import path.")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from aneux_paths import CSV_PATH as DEFAULT_CSV_PATH, VESSELS_AREA005 as DEFAULT_VESSEL_DIR, TEMPLATE_OUTPUT as DEFAULT_OUTPUT_DIR

# --- DEFAULT CONFIGURATION PATHS ---
DEFAULT_NUM_SAMPLES = None
DEFAULT_NUM_WORKERS = 2
FILTER_LOCATIONS = ["ICA pcom", "ICA oph", "ICA cav", "ICA bif"]


def get_existing_path(primary_path, alt_path=None):
    """Returns primary_path if it exists, otherwise alt_path or primary_path."""
    if os.path.exists(primary_path):
        return primary_path
    if alt_path and os.path.exists(alt_path):
        return alt_path
    return primary_path


def to_vtk_poly(mesh):
    """Converts PyVista PolyData or VTK PolyData to pure vtkPolyData object for VMTK."""
    if mesh is None:
        return vtk.vtkPolyData()
    vtk_poly = vtk.vtkPolyData()
    vtk_poly.DeepCopy(mesh)
    if hasattr(vtk_poly, 'SetSource'):
        vtk_poly.SetSource(None)
    return vtk_poly


def apply_taubin_smoothing(surface_mesh, pass_band=0.1, n_iter=15, feature_angle=45.0):
    """
    Step 1: Volume-Preserving Surface Smoothing using Taubin (pass-band) smoothing.
    
    Prevents mesh collapse/shrinkage while eliminating high-frequency surface noise 
    that causes Voronoi diagram jitter.
    """
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(to_vtk_poly(surface_mesh))
    smoother.SetNumberOfIterations(n_iter)
    smoother.SetPassBand(pass_band)
    smoother.SetFeatureAngle(feature_angle)
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOn()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return to_vtk_poly(smoother.GetOutput())


def add_flow_extensions(surface_mesh, extension_length=5.0, extension_mode="boundarynormal"):
    """
    Step 2: Add Flow Extensions to open boundaries.
    
    Extrudes cylindrical tubes normal to inlet/outlet boundaries to push chaotic
    Voronoi boundary collapse away from the anatomical region of interest.
    """
    if not HAS_VMTK:
        print("VMTK is required for vmtkflowextensions. Returning smoothed surface without extensions.")
        return surface_mesh

    vtk_poly = to_vtk_poly(surface_mesh)

    # Cap open profiles to enable extension extrusion
    capper = vmtkscripts.vmtkSurfaceCapper()
    capper.Surface = vtk_poly
    capper.Interactive = 0
    capper.Execute()

    # Add cylindrical flow extensions
    extender = vmtkscripts.vmtkFlowExtensions()
    extender.Surface = capper.Surface
    extender.ExtensionLength = extension_length
    extender.ExtensionMode = extension_mode
    extender.Interactive = 0
    extender.Execute()

    return to_vtk_poly(extender.Surface)


def find_largest_open_profile_seed(surface_mesh):
    """
    Detects all open boundary profiles on the vessel surface and identifies inlet/outlets.

    The main ICA inlet is defined as the open boundary profile with the LARGEST geometric
    radius (mean distance from boundary points to their barycenter). This is orientation-
    independent and works robustly across all ICA aneurysm cases.

    Returns:
        source_pts: list with one point [x, y, z] — the main ICA inlet barycenter
        target_pts: list of points [x, y, z] — all outlet barycenters
    """
    if not HAS_VMTK:
        raise RuntimeError("VMTK is required for boundary extraction.")

    vtk_poly = to_vtk_poly(surface_mesh)
    boundary_extractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
    boundary_extractor.SetInputData(vtk_poly)
    boundary_extractor.Update()
    boundaries = boundary_extractor.GetOutput()
    num_boundaries = boundaries.GetNumberOfCells()

    if num_boundaries == 0:
        raise RuntimeError("No open boundary profiles found on vessel mesh.")

    profiles = []
    for i in range(num_boundaries):
        cell = boundaries.GetCell(i)
        cell_pts = cell.GetPoints()
        num_pts = cell_pts.GetNumberOfPoints()
        if num_pts == 0:
            continue
        pts_coords = np.array([cell_pts.GetPoint(j) for j in range(num_pts)])
        barycenter = np.mean(pts_coords, axis=0)
        radii = np.linalg.norm(pts_coords - barycenter, axis=1)
        mean_radius = float(np.mean(radii))
        profiles.append({
            'index': i,
            'barycenter': barycenter,
            'radius': mean_radius
        })

    if not profiles:
        raise RuntimeError("No valid boundary profiles could be measured on vessel mesh.")

    # Sort strictly by largest radius — the main ICA inlet has the widest open boundary
    profiles.sort(key=lambda p: p['radius'], reverse=True)

    print(f"  Boundary profiles found: {len(profiles)}")
    for p in profiles:
        print(f"    Profile {p['index']}: radius={p['radius']:.3f} mm, center={np.round(p['barycenter'], 2)}")
    print(f"  -> Selected inlet: Profile {profiles[0]['index']} (radius={profiles[0]['radius']:.3f} mm)")

    inlet_barycenter = profiles[0]['barycenter']
    outlet_barycenters = [p['barycenter'] for p in profiles[1:]]

    return [inlet_barycenter], outlet_barycenters


def extract_voronoi_centerlines(extended_surface, source_points, target_points):
    """
    Step 3: Voronoi Centerline Extraction & Automatic MISR Calculation.
    
    Runs vmtkcenterlines to compute the Voronoi diagram and extract centerlines.
    VMTK automatically calculates the Maximum Inscribed Sphere Radius (MISR)
    and stores it in the point data array 'MaximumInscribedSphereRadius'.
    
    source_points and target_points are required: list of [x, y, z] coordinates.
    """
    if not HAS_VMTK:
        raise RuntimeError("VMTK is required for Voronoi centerline extraction.")

    vtk_poly = to_vtk_poly(extended_surface)

    centerlines = vmtkscripts.vmtkCenterlines()
    centerlines.Surface = vtk_poly
    centerlines.SeedSelectorName = "pointlist"
    centerlines.SourcePoints = [coord for pt in source_points for coord in pt]
    centerlines.TargetPoints = [coord for pt in target_points for coord in pt]
    centerlines.Interactive = 0
    centerlines.Execute()
    return to_vtk_poly(centerlines.Centerlines)


def resample_and_smooth_centerline(centerline, sample_spacing=0.1, smoothing_factor=0.1, iterations=100):
    """
    Step 4: Spline Resampling and Trajectory Smoothing.
    
    Resamples the centerline at uniform intervals (e.g. 0.1 mm) and applies
    spline smoothing to ensure fluid trajectory without macroscopic jaggedness.
    """
    if not HAS_VMTK:
        return to_vtk_poly(centerline)

    vtk_poly = to_vtk_poly(centerline)

    # Spline resampling at uniform step size
    resampler = vmtkscripts.vmtkCenterlineResampling()
    resampler.Centerlines = vtk_poly
    resampler.Length = sample_spacing
    resampler.Execute()

    # Trajectory smoothing
    smoother = vmtkscripts.vmtkCenterlineSmoothing()
    smoother.Centerlines = resampler.Centerlines
    smoother.SmoothingFactor = smoothing_factor
    smoother.Iterations = iterations
    smoother.Execute()

    return to_vtk_poly(smoother.Centerlines)


def extract_branches(centerline):
    """
    Step 5: Branch Extraction and Splitting.
    
    Analyzes bifurcations and splits continuous centerline into distinct branch groups
    assigning GroupIds, TractIds, and CenterlineId arrays.
    """
    if not HAS_VMTK:
        return to_vtk_poly(centerline)

    vtk_poly = to_vtk_poly(centerline)

    extractor = vmtkscripts.vmtkBranchExtractor()
    extractor.Centerlines = vtk_poly
    extractor.Execute()

    return to_vtk_poly(extractor.Centerlines)


def clip_flow_extensions(centerline_mesh, original_vessel_mesh, max_margin=1.0):
    """
    Step 6: Clip Flow Extensions.
    
    Clips off the flow extensions added in Step 2 so that the final centerline
    terminates exactly at the anatomical inlet and outlet boundaries of the original vessel.
    """
    pv_centerline = pv.wrap(to_vtk_poly(centerline_mesh))
    pv_vessel = pv.wrap(to_vtk_poly(original_vessel_mesh))

    # Bounding box of original vessel mesh with a small margin
    bounds = pv_vessel.bounds
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    
    # Clip points outside expanded bounding volume of original vessel
    clipped = pv_centerline.clip_box(
        bounds=[
            xmin - max_margin, xmax + max_margin,
            ymin - max_margin, ymax + max_margin,
            zmin - max_margin, zmax + max_margin
        ],
        invert=False
    )
    if isinstance(clipped, pv.UnstructuredGrid) or not isinstance(clipped, pv.PolyData):
        clipped = clipped.extract_surface(algorithm=None)
    return to_vtk_poly(clipped)


def process_dataset(dataset_id, v_file, output_dir, extension_length=5.0, sample_spacing=0.1):
    """Runs the full centerline creation pipeline for a single dataset case."""
    print(f"\n=========================================\nProcessing Case: {dataset_id}")

    # Load raw vessel mesh
    vessel_mesh = pv.read(v_file)

    # Step 1: Taubin Volume-Preserving Surface Smoothing
    print("Step 1: Applying Taubin surface smoothing (volume preserving)...")
    smoothed_vessel = apply_taubin_smoothing(vessel_mesh, pass_band=0.1, n_iter=15)

    if not HAS_VMTK:
        print("VMTK Python library not found. Saving Taubin smoothed vessel reference mesh.")
        out_file = os.path.join(output_dir, f"{dataset_id}_smoothed.vtp")
        smoothed_pv = pv.wrap(to_vtk_poly(smoothed_vessel))
        smoothed_pv.save(out_file, binary=True)
        return

    # Detect inlet/outlet seed points on the smoothed vessel BEFORE flow extensions.
    print("Step 1b: Detecting inlet/outlet boundaries by open profile radius...")
    source_pts, target_pts = find_largest_open_profile_seed(smoothed_vessel)

    # Step 2: Add Flow Extensions
    print("Step 2: Adding flow extensions for boundary stability...")
    extended_vessel = add_flow_extensions(smoothed_vessel, extension_length=extension_length)

    # Step 3: Voronoi Centerline Extraction & Automatic MISR
    print("Step 3: Extracting Voronoi centerline & calculating MISR...")
    centerline = extract_voronoi_centerlines(extended_vessel, source_pts, target_pts)

    # Step 4: Spline Resampling & Trajectory Smoothing
    print("Step 4: Spline resampling (0.1mm) and trajectory smoothing...")
    smooth_centerline = resample_and_smooth_centerline(centerline, sample_spacing=sample_spacing)

    # Step 5: Branch Extraction and Splitting
    print("Step 5: Extracting branches and assigning Group IDs / Tract IDs...")
    branched_centerline = extract_branches(smooth_centerline)

    # Step 6: Clip Flow Extensions
    print("Step 6: Clipping flow extensions...")
    final_centerline = clip_flow_extensions(branched_centerline, vessel_mesh)

    # Save final centerline .vtp output
    out_file = os.path.join(output_dir, f"{dataset_id}.vtp")
    pv_final = pv.wrap(to_vtk_poly(final_centerline))
    if isinstance(pv_final, pv.UnstructuredGrid) or not isinstance(pv_final, pv.PolyData):
        pv_final = pv_final.extract_surface(algorithm=None)
    pv_final.save(out_file, binary=True)

    print(f"Successfully saved centerline to: {out_file}")


def _process_dataset_worker(task_tuple):
    """Top-level process worker wrapper for parallel dataset processing."""
    dataset_id, v_file, output_dir, extension_length, sample_spacing = task_tuple
    try:
        process_dataset(
            dataset_id=dataset_id,
            v_file=v_file,
            output_dir=output_dir,
            extension_length=extension_length,
            sample_spacing=sample_spacing
        )
        return (dataset_id, True, None)
    except Exception as e:
        return (dataset_id, False, str(e))


def main():
    parser = argparse.ArgumentParser(description="AneuX Centerline Extraction Pipeline")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to clinical.csv")
    parser.add_argument("--vessel-dir", type=str, default=DEFAULT_VESSEL_DIR, help="Directory containing remeshed vessel .vtp files")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for generated centerlines")
    parser.add_argument("--limit", type=int, default=DEFAULT_NUM_SAMPLES, help="Maximum number of dataset meshes to process")
    parser.add_argument("--workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of parallel worker processes")
    parser.add_argument("--extension-length", type=float, default=5.0, help="Length of flow extensions in mm")
    parser.add_argument("--sample-spacing", type=float, default=0.1, help="Resampling spacing in mm")

    args = parser.parse_args()

    csv_path = args.csv
    vessel_dir = args.vessel_dir
    output_dir = args.output_dir
    num_samples = args.limit

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Clinical CSV file not found at: {csv_path}")

    # Read and filter clinical dataset
    df = pd.read_csv(csv_path)
    df["location"] = df["location"].astype(str).str.strip()
    df_filtered = df[df["location"].isin(FILTER_LOCATIONS)]

    print(f"Filtered {len(df_filtered)} cases matching target locations: {FILTER_LOCATIONS}")

    # Build list of valid datasets — only requires vessel mesh to exist
    valid_datasets = [
        (r["dataset"], os.path.join(vessel_dir, f"{r['dataset']}.vtp"))
        for _, r in df_filtered.iterrows()
    ]
    valid_datasets = [
        d for d in valid_datasets if os.path.exists(d[1])
    ][:num_samples]

    num_cases = len(valid_datasets)
    requested_workers = args.workers
    
    num_workers = min(requested_workers, num_cases)
    num_workers = max(1, num_workers)

    print(f"CSV Path: {csv_path}")
    print(f"Vessel Dir: {vessel_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"Processing limit: {num_samples} samples | Valid dataset cases: {num_cases}")
    print(f"Parallel Workers: {num_workers} (Requested={requested_workers}, Active Workers={num_workers})")

    import subprocess
    import sys
    from queue import Queue
    import threading

    python_exe = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if num_workers > 1:
        print(f"Spawning worker pool with max active concurrent workers = {num_workers}...")
        task_queue = Queue()
        for item in valid_datasets:
            task_queue.put(item)

        pbar = tqdm(total=num_cases, desc="Processing Parallel Centerline Extraction")

        def worker_thread():
            while not task_queue.empty():
                try:
                    dataset_id, v_file = task_queue.get_nowait()
                except Exception:
                    break
                
                cmd = [
                    python_exe, "-c",
                    f"""
import sys, os
os.environ['VTK_OFFSCREEN'] = '1'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VTK_NUMBER_OF_THREADS'] = '1'
sys.path.insert(0, r'{script_dir}')
import centerline_creation
centerline_creation.process_dataset(
    dataset_id='{dataset_id}',
    v_file=r'{v_file}',
    output_dir=r'{output_dir}',
    extension_length={args.extension_length},
    sample_spacing={args.sample_spacing}
)
"""
                ]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                out, _ = proc.communicate()
                if proc.returncode != 0:
                    print(f"\n[ERROR] Case {dataset_id} failed (code {proc.returncode}):\n{out}")
                pbar.update(1)
                task_queue.task_done()

        threads = []
        for _ in range(num_workers):
            t = threading.Thread(target=worker_thread)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        pbar.close()
        print("Parallel execution complete. All worker processes finished.")
    else:
        for dataset_id, v_file in tqdm(valid_datasets, desc="Processing Centerline Pipeline"):
            res_id, success, err = _process_dataset_worker((dataset_id, v_file, output_dir, args.extension_length, args.sample_spacing))
            if not success:
                print(f"Error processing case {res_id}: {err}")


if __name__ == "__main__":
    main()
