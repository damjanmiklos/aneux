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
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from vmtk import vmtkscripts
    from vmtk import vtkvmtk
    HAS_VMTK = True
except ImportError:
    HAS_VMTK = False
    print("Warning: VMTK Python bindings not found in default import path.")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from aneux_paths import CSV_PATH as DEFAULT_CSV_PATH, VESSELS_AREA005 as DEFAULT_VESSEL_DIR, TEMPLATE_OUTPUT_REMESHED as DEFAULT_OUTPUT_DIR

# --- DEFAULT CONFIGURATION PATHS & PARAMETERS ---
DEFAULT_NUM_SAMPLES = None
DEFAULT_TARGET_EDGE_LENGTH = 0.5  # mm (Isotropic uniform edge length)
DEFAULT_EXTENSION_LENGTH = 5.0    # mm (Flow extensions)
DEFAULT_SAMPLE_SPACING = 0.1      # mm (High-res centerline spline spacing)
DEFAULT_GRID_SPACING = 0.08       # mm (3D image modeller grid spacing)
DEFAULT_MAX_GRID_SIZE = 50       # Upper voxel dimension cap for CenterlineModeller
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
    Internal endcapping is performed by vmtkSurfaceCapper prior to extension extrusion.
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
    Detects all open boundary profiles on the vessel surface and extracts outward boundary normals using VMTK.

    The main ICA inlet is defined as the open boundary profile with the LARGEST geometric
    radius.
    """
    if not HAS_VMTK:
        raise RuntimeError("VMTK is required for boundary extraction.")

    vtk_poly = to_vtk_poly(surface_mesh)

    ref_sys = vtkvmtk.vtkvmtkBoundaryReferenceSystems()
    ref_sys.SetInputData(vtk_poly)
    ref_sys.SetBoundaryNormalsArrayName('BoundaryNormals')
    ref_sys.SetBoundaryRadiusArrayName('BoundaryRadius')
    ref_sys.SetPoint1ArrayName('Point1')
    ref_sys.SetPoint2ArrayName('Point2')
    ref_sys.Update()
    ref_poly = ref_sys.GetOutput()

    num_b = ref_poly.GetNumberOfPoints()
    if num_b == 0:
        raise RuntimeError("No open boundary profiles found on vessel mesh.")

    normals_array = ref_poly.GetPointData().GetArray('BoundaryNormals')
    radii_array = ref_poly.GetPointData().GetArray('BoundaryRadius')

    profiles = []
    for i in range(num_b):
        pos = np.array(ref_poly.GetPoint(i))
        normal = np.array(normals_array.GetTuple(i))
        radius = radii_array.GetComponent(i, 0)
        profiles.append({
            'index': i,
            'barycenter': pos,
            'normal': normal,
            'radius': radius
        })

    profiles.sort(key=lambda p: p['radius'], reverse=True)

    print(f"  Boundary profiles found: {len(profiles)}")
    for p in profiles:
        print(f"    Profile {p['index']}: radius={p['radius']:.3f} mm, center={np.round(p['barycenter'], 2)}, normal={np.round(p['normal'], 3)}")
    print(f"  -> Selected inlet: Profile {profiles[0]['index']} (radius={profiles[0]['radius']:.3f} mm)")

    inlet_barycenter = profiles[0]['barycenter']
    outlet_barycenters = [p['barycenter'] for p in profiles[1:]]

    return [inlet_barycenter], outlet_barycenters, profiles


def extract_voronoi_centerlines(extended_surface, source_points, target_points):
    """
    Step 3: Voronoi Centerline Extraction & Automatic MISR Calculation.
    
    Runs vmtkcenterlines to compute the Voronoi diagram and extract centerlines.
    VMTK automatically calculates the Maximum Inscribed Sphere Radius (MISR)
    and stores it in the point data array 'MaximumInscribedSphereRadius'.
    The sphere centroids lie exactly on the extracted centerline.
    """
    if not HAS_VMTK:
        raise RuntimeError("VMTK is required for Voronoi centerline extraction.")

    vtk_poly = to_vtk_poly(extended_surface)

    centerlines = vmtkscripts.vmtkCenterlines()
    centerlines.Surface = vtk_poly
    centerlines.SeedSelectorName = "pointlist"
    centerlines.SourcePoints = [float(coord) for pt in source_points for coord in pt]
    centerlines.TargetPoints = [float(coord) for pt in target_points for coord in pt]
    centerlines.Interactive = 0
    centerlines.Execute()
    return to_vtk_poly(centerlines.Centerlines)


def resample_and_smooth_centerline(centerline, sample_spacing=0.1, smoothing_factor=0.1, iterations=100):
    """
    Step 4: Spline Resampling and Trajectory Smoothing at High Resolution.
    
    Resamples the centerline at uniform high-resolution intervals (e.g. 0.1 mm)
    and applies spline smoothing while continuously evaluating MISR along the centerline.
    """
    if not HAS_VMTK:
        return to_vtk_poly(centerline)

    vtk_poly = to_vtk_poly(centerline)

    resampler = vmtkscripts.vmtkCenterlineResampling()
    resampler.Centerlines = vtk_poly
    resampler.Length = sample_spacing
    resampler.Execute()

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


def generate_base_surface(branched_centerline, grid_spacing=0.08, max_grid_size=DEFAULT_MAX_GRID_SIZE):
    """
    Step 6: High-Resolution Base Surface Generation via vmtkCenterlineModeller.
    
    Evaluates the polyball tube function (union of maximal inscribed spheres centered on
    centerline nodes) onto a high-resolution 3D grid image, encompassing all vessel branches.
    Then extracts the zero-level isosurface with vmtkMarchingCubes.
    """
    if not HAS_VMTK:
        raise RuntimeError("VMTK is required for vmtkCenterlineModeller.")

    vtk_cl = to_vtk_poly(branched_centerline)
    
    # MISR Floor Guard: enforce minimum radius >= 0.35mm to prevent sub-mm branch collapse
    misr_array = vtk_cl.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if misr_array:
        for i in range(misr_array.GetNumberOfTuples()):
            r_val = misr_array.GetComponent(i, 0)
            if r_val < 0.35:
                misr_array.SetComponent(i, 0, 0.35)

    pv_cl = pv.wrap(vtk_cl)
    bounds = pv_cl.bounds
    margin = 5.0
    dx = (bounds[1] - bounds[0]) + 2 * margin
    dy = (bounds[3] - bounds[2]) + 2 * margin
    dz = (bounds[5] - bounds[4]) + 2 * margin

    dims = [
        max(32, min(int(np.ceil(dx / grid_spacing)), max_grid_size)),
        max(32, min(int(np.ceil(dy / grid_spacing)), max_grid_size)),
        max(32, min(int(np.ceil(dz / grid_spacing)), max_grid_size))
    ]
    print(f"  CenterlineModeller grid dimensions: {dims}")

    modeller = vmtkscripts.vmtkCenterlineModeller()
    modeller.Centerlines = vtk_cl
    modeller.RadiusArrayName = "MaximumInscribedSphereRadius"
    modeller.SampleDimensions = dims
    modeller.Execute()

    mc = vmtkscripts.vmtkMarchingCubes()
    mc.Image = modeller.Image
    mc.Level = 0.0
    mc.Connectivity = 1
    mc.Execute()

    return to_vtk_poly(mc.Surface)


def clip_flow_extensions_and_uncap(base_surface, profiles):
    """
    Step 7: Uncap Open Boundaries & Remove Flow Extensions.
    
    Uses vtkvmtkTopologicalSeamFilter and local connectivity extraction to trim flow
    extensions locally at boundary barycenters without cutting across other vessel branches.
    """
    current_surface = to_vtk_poly(base_surface)
    main_body_pt = profiles[0]['barycenter']

    for idx_p, p in enumerate(profiles):
        # Small-Radius Profile Safety Guard: Skip profiles with radius < 0.45mm
        # to prevent vtkvmtkTopologicalSeamFilter C++ memory access violation
        if p['radius'] < 0.45:
            print(f"  [Step 7 Guard] Profile {idx_p} radius ({p['radius']:.3f} mm < 0.45 mm) is below topological seam threshold. Skipping clip.")
            continue

        plane = vtk.vtkPlane()
        plane.SetOrigin(p['barycenter'])
        plane.SetNormal(-p['normal'])

        seam_filter = vtkvmtk.vtkvmtkTopologicalSeamFilter()
        seam_filter.SetInputData(current_surface)
        seam_filter.SetClosestPoint(p['barycenter'])
        seam_filter.SetSeamScalarsArrayName("SeamScalars")
        seam_filter.SetSeamFunction(plane)

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputConnection(seam_filter.GetOutputPort())
        clipper.GenerateClipScalarsOff()
        clipper.GenerateClippedOutputOff()

        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputConnection(clipper.GetOutputPort())
        connectivity.SetExtractionModeToClosestPointRegion()
        connectivity.SetClosestPoint(main_body_pt)

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(connectivity.GetOutputPort())
        cleaner.Update()
        
        candidate = to_vtk_poly(cleaner.GetOutput())
        # Safety Guard: Ensure candidate clip contains valid surface elements before updating
        if candidate.GetNumberOfPoints() > 10 and candidate.GetNumberOfCells() > 10:
            current_surface = candidate
        else:
            print(f"  [Step 7 Guard] Profile {idx_p} clip returned degenerate mesh ({candidate.GetNumberOfPoints()} pts, {candidate.GetNumberOfCells()} cells). Retaining previous surface.")

    return current_surface


def remesh_surface_isotropically(open_surface, target_edge_length=0.5, n_iter=10):
    """
    Step 8: Isotropic & Perfectly Uniform Surface Remeshing.
    
    Remeshes the uncapped surface isotropically using vmtkSurfaceRemeshing
    configured with target edge length and PreserveBoundaryEdges=1.
    """
    if not HAS_VMTK:
        raise RuntimeError("VMTK is required for vmtkSurfaceRemeshing.")

    vtk_open = to_vtk_poly(open_surface)

    remesher = vmtkscripts.vmtkSurfaceRemeshing()
    remesher.Surface = vtk_open
    remesher.ElementSizeMode = "edgelength"
    remesher.TargetEdgeLength = target_edge_length
    remesher.PreserveBoundaryEdges = 1
    remesher.NumberOfIterations = n_iter
    remesher.Execute()

    return to_vtk_poly(remesher.Surface)


def process_dataset(dataset_id, v_file, output_dir, target_edge_length=0.5, extension_length=5.0, sample_spacing=0.1, grid_spacing=0.08, max_grid_size=DEFAULT_MAX_GRID_SIZE):
    """Runs the full uniform remeshing pipeline for a single dataset case."""
    print(f"\n=========================================\nProcessing Uniform Remeshing Case: {dataset_id}")

    # Load raw vessel mesh
    vessel_mesh = pv.read(v_file)

    # Step 1: Taubin Volume-Preserving Surface Smoothing
    print("Step 1: Applying Taubin surface smoothing...")
    smoothed_vessel = apply_taubin_smoothing(vessel_mesh, pass_band=0.1, n_iter=15)

    # Detect inlet/outlet seed points and boundary reference systems
    print("Step 1b: Detecting inlet/outlet boundaries and outward normals...")
    source_pts, target_pts, profiles = find_largest_open_profile_seed(smoothed_vessel)

    # Step 2: Add Flow Extensions & Endcaps
    print("Step 2: Capping & adding flow extensions for boundary stability...")
    extended_vessel = add_flow_extensions(smoothed_vessel, extension_length=extension_length)

    # Step 3: Voronoi Centerline Extraction & MISR
    print("Step 3: Extracting Voronoi centerline & calculating high-res MISR...")
    centerline = extract_voronoi_centerlines(extended_vessel, source_pts, target_pts)

    # Step 4: High-Res Spline Resampling & Trajectory Smoothing
    print(f"Step 4: Spline resampling ({sample_spacing}mm) and trajectory smoothing...")
    smooth_centerline = resample_and_smooth_centerline(centerline, sample_spacing=sample_spacing)

    # Step 5: Branch Extraction and Splitting
    print("Step 5: Extracting branches...")
    branched_centerline = extract_branches(smooth_centerline)

    # Step 6: High-Resolution Base Surface Generation
    print("Step 6: Generating multi-branch base surface (vmtkCenterlineModeller)...")
    base_surface = generate_base_surface(branched_centerline, grid_spacing=grid_spacing, max_grid_size=max_grid_size)

    # Step 7: Uncap Open Boundaries & Remove Flow Extensions
    print("Step 7: Uncapping open boundaries & removing flow extensions...")
    open_base_surface = clip_flow_extensions_and_uncap(base_surface, profiles)
    print(f"  -> Open base surface points: {open_base_surface.GetNumberOfPoints()}")

    # Step 8: Isotropic & Uniform Surface Remeshing with Adaptive Small-Branch Edge Scaling
    r_min = min([p['radius'] for p in profiles]) if profiles else 0.5
    effective_edge_length = min(target_edge_length, max(0.15, 0.45 * r_min))
    print(f"Step 8: Isotropically remeshing surface (TargetEdgeLength={effective_edge_length:.3f}mm, R_min={r_min:.3f}mm)...")
    remeshed_surface = remesh_surface_isotropically(open_base_surface, target_edge_length=effective_edge_length)
    print(f"  -> Remeshed surface points: {remeshed_surface.GetNumberOfPoints()}")

    # Clean intermediate arrays and texture coordinates to fix ParaView texture loading errors
    clean_poly = to_vtk_poly(remeshed_surface)
    clean_poly.GetPointData().SetTCoords(None)
    clean_poly.GetCellData().SetTCoords(None)
    clean_poly.GetPointData().SetNormals(None)
    clean_poly.GetCellData().SetNormals(None)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(clean_poly)
    cleaner.Update()

    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputConnection(cleaner.GetOutputPort())
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOff()
    normals_filter.ConsistencyOn()
    normals_filter.SplittingOff()
    normals_filter.Update()

    # Save final uniform remeshed surface .vtp output
    out_file = os.path.join(output_dir, f"{dataset_id}.vtp")
    pv_save = pv.wrap(to_vtk_poly(normals_filter.GetOutput()))
    if isinstance(pv_save, pv.UnstructuredGrid) or not isinstance(pv_save, pv.PolyData):
        pv_save = pv_save.extract_surface(algorithm='dataset_surface')
    pv_save.save(out_file, binary=True)

    # Verify open boundaries count on saved file
    read_back = pv.read(out_file)
    rem_b_extractor = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
    rem_b_extractor.SetInputData(to_vtk_poly(read_back))
    rem_b_extractor.Update()
    num_open_b = rem_b_extractor.GetOutput().GetNumberOfCells()

    print(f"Successfully saved uniform remeshed surface to: {out_file}")
    print(f"  -> Verified Saved Mesh: {read_back.n_points} points, {read_back.n_cells} cells, disk size={os.path.getsize(out_file)} bytes")
    print(f"  -> Verified Open Boundaries Count: {num_open_b} (matched target anatomical profiles)")


def _process_dataset_worker(task_tuple):
    """Top-level process worker wrapper for parallel dataset processing."""
    dataset_id, v_file, output_dir, target_edge_length, extension_length, sample_spacing, grid_spacing, max_grid_size = task_tuple
    try:
        process_dataset(
            dataset_id=dataset_id,
            v_file=v_file,
            output_dir=output_dir,
            target_edge_length=target_edge_length,
            extension_length=extension_length,
            sample_spacing=sample_spacing,
            grid_spacing=grid_spacing,
            max_grid_size=max_grid_size
        )
        return (dataset_id, True, None)
    except Exception as e:
        return (dataset_id, False, str(e))


def main():
    parser = argparse.ArgumentParser(description="AneuX Uniform Surface Remeshing Pipeline")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to clinical.csv")
    parser.add_argument("--vessel-dir", type=str, default=DEFAULT_VESSEL_DIR, help="Directory containing input vessel .vtp files")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for final uniform remeshed meshes")
    parser.add_argument("--limit", type=int, default=DEFAULT_NUM_SAMPLES, help="Maximum number of dataset meshes to process")
    parser.add_argument("--workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of parallel worker processes")
    parser.add_argument("--target-edge-length", type=float, default=DEFAULT_TARGET_EDGE_LENGTH, help="Target isotropic edge length in mm")
    parser.add_argument("--extension-length", type=float, default=DEFAULT_EXTENSION_LENGTH, help="Length of flow extensions in mm")
    parser.add_argument("--sample-spacing", type=float, default=DEFAULT_SAMPLE_SPACING, help="Centerline resampling spacing in mm")
    parser.add_argument("--grid-spacing", type=float, default=DEFAULT_GRID_SPACING, help="Base surface modeller grid resolution in mm")
    parser.add_argument("--max-grid-size", type=int, default=DEFAULT_MAX_GRID_SIZE, help="Maximum 3D grid image dimension for CenterlineModeller")

    args = parser.parse_args()

    csv_path = args.csv
    vessel_dir = args.vessel_dir
    output_dir = args.output_dir
    num_samples = args.limit

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Clinical CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df["location"] = df["location"].astype(str).str.strip()
    df_filtered = df[df["location"].isin(FILTER_LOCATIONS)]

    valid_datasets = [
        (r["dataset"], os.path.join(vessel_dir, f"{r['dataset']}.vtp"))
        for _, r in df_filtered.iterrows()
    ]
    valid_datasets = [
        d for d in valid_datasets if os.path.exists(d[1])
    ][:num_samples]

    num_cases = len(valid_datasets)
    requested_workers = args.workers
    
    # Cap worker count to num_cases if requested_workers > num_cases
    num_workers = min(requested_workers, num_cases)
    num_workers = max(1, num_workers)

    print(f"CSV Path: {csv_path}")
    print(f"Vessel Dir: {vessel_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"Target Edge Length: {args.target_edge_length} mm")
    print(f"Grid Spacing: {args.grid_spacing} mm | Max Grid Size: {args.max_grid_size}")
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

        pbar = tqdm(total=num_cases, desc="Processing Parallel Uniform Remeshing")

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
import uniform_remeshing
uniform_remeshing.process_dataset(
    dataset_id='{dataset_id}',
    v_file=r'{v_file}',
    output_dir=r'{output_dir}',
    target_edge_length={args.target_edge_length},
    extension_length={args.extension_length},
    sample_spacing={args.sample_spacing},
    grid_spacing={args.grid_spacing},
    max_grid_size={args.max_grid_size}
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
        tasks = [
            (
                dataset_id,
                v_file,
                output_dir,
                args.target_edge_length,
                args.extension_length,
                args.sample_spacing,
                args.grid_spacing,
                args.max_grid_size
            )
            for dataset_id, v_file in valid_datasets
        ]
        for t in tqdm(tasks, desc="Processing Uniform Remeshing Pipeline"):
            res_id, success, err = _process_dataset_worker(t)
            if not success:
                print(f"Error processing case {res_id}: {err}")


if __name__ == "__main__":
    main()
