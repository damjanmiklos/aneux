"""Assemble total_clean_original_mesh from four sources with priority.

Priority (highest first):
  1. datatransform/hemoMesh/keep_one/surfaces
  2. datatransform/cleaned_data/vessels_cleaned_and_decapped
  3. datatransform/uncapped
  4. rawdata/.../vessels/original               (read-only; fallback only)

Keep-one naming
---------------
A keep-one mesh is a vessel with one aneurysm left on it, and most of these
datasets already have a way of naming exactly that, so the output takes the
dataset's own slot rather than a suffix of ours:

    C0028a        -> C0028a, C0028b, ...          (trailing letter)
    UPF_..._ID1   -> UPF_..._ID1, UPF_..._ID2     (the ID field)
    USFD_0051     -> USFD_0051, USFD_0052         (the case number itself)
    p..., SNF...  -> stem_1, stem_2               (no convention of their own)

That means a keep-one output frequently lands on a name the original dataset
already uses, which is the point: C0028b is the same patient in the same frame
as C0028a, so our version of that aneurysm replaces the dataset's. It is also
the danger, because the rule is arithmetic and the dataset's numbering is not.
USFD_0051 has two aneurysms and USFD_0052 is indeed the same patient, but
USFD_0053 is somebody else -- a third aneurysm there would silently overwrite an
unrelated vessel. So every displaced slot is checked geometrically before
anything is copied, and a mismatch stops the run rather than corrupting the set.

The rule itself is imported from remove_other_aneurysms so the two cannot drift.

Names are derived from the picked-points directory (which vessel has how many
aneurysms) rather than parsed back out of the filenames, because under this
scheme a filename no longer says which vessel it came from.
"""
from __future__ import annotations

import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aneux_paths import CLEANED_VESSELS, DATATRANSFORM, UNCAPPED_VESSELS, VESSELS_ORIGINAL

sys.path.insert(0, str(Path(DATATRANSFORM) / "hemoMesh"))
from remove_other_aneurysms import complete_aneurysm_count, keep_one_dest_name

MESH_EXTS = {".stl", ".vtp", ".vtk", ".ply"}

HEMOMESH = Path(DATATRANSFORM) / "hemoMesh"
KEEP_ONE_SURFACES = HEMOMESH / "keep_one" / "surfaces"
PICKED_POINTS = HEMOMESH / "picked_points"
DEST_DIR = Path(DATATRANSFORM) / "cleaned_data" / "total_clean_original_mesh"
# Generated output, so it goes to scratch/ (gitignored) rather than next to
# the script, where it would be committed every time the set is rebuilt.
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "scratch" / "total_clean_original_mesh_manifest.csv"

# Two meshes of the same patient sit in the same scan frame and overlap almost
# completely; the measured pairs span 97-100%, and the nearest unrelated vessel
# measured 41.7%. Anything under this is not the aneurysm we think it is.
SAME_PATIENT_MIN_OVERLAP = 0.80


def list_meshes(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")
    by_stem: dict[str, Path] = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in MESH_EXTS:
            if path.stem in by_stem:
                raise SystemExit(f"Duplicate stem in {folder}: {path.stem}")
            by_stem[path.stem] = path
    return by_stem


def keep_one_slots(picked_points_dir: Path, keep_one: dict[str, Path]):
    """{source vessel: [(slot name, file)]} for every keep-one mesh on disk.

    Reads the aneurysm count from the picked points, applies the naming rule,
    and matches the result against what is actually in the surfaces folder, so a
    file nobody can account for is reported instead of ignored.
    """
    slots: dict[str, list[tuple[str, Path]]] = {}
    claimed: dict[str, str] = {}
    stems = sorted({path.stem for path in picked_points_dir.glob("*_*.json")})
    for stem in {s.rsplit("_", 1)[0] for s in stems}:
        n = complete_aneurysm_count(picked_points_dir, stem)
        if n < 2:
            continue
        found = []
        for keep in range(1, n + 1):
            name = keep_one_dest_name(stem, keep)
            path = keep_one.get(name)
            if path is None:
                continue
            if name in claimed:
                raise SystemExit(
                    f"{name} is claimed by both {claimed[name]} and {stem}"
                )
            claimed[name] = stem
            found.append((name, path))
        if found:
            slots[stem] = found

    unaccounted = sorted(set(keep_one) - set(claimed))
    if unaccounted:
        raise SystemExit(
            "Keep-one file(s) no naming rule accounts for (refusing to guess): "
            + ", ".join(unaccounted)
        )
    return slots


def _bbox_overlap_fraction(a: Path, b: Path) -> float:
    """Shared bounding-box volume as a fraction of the smaller one."""
    import numpy as np
    import pyvista as pv

    box_a = np.asarray(pv.read(str(a)).bounds, dtype=float).reshape(3, 2)
    box_b = np.asarray(pv.read(str(b)).bounds, dtype=float).reshape(3, 2)
    lo = np.maximum(box_a[:, 0], box_b[:, 0])
    hi = np.minimum(box_a[:, 1], box_b[:, 1])
    shared = float(np.prod(np.clip(hi - lo, 0.0, None)))
    volumes = [float(np.prod(box[:, 1] - box[:, 0])) for box in (box_a, box_b)]
    smallest = min(volumes)
    return shared / smallest if smallest > 0 else 0.0


def check_displaced_slots(slots, original: dict[str, Path]) -> list[str]:
    """Refuse to overwrite an original that is not the same patient."""
    notes = []
    for source_stem, entries in sorted(slots.items()):
        for name, path in entries:
            displaced = original.get(name)
            if displaced is None or name == source_stem:
                continue
            overlap = _bbox_overlap_fraction(path, displaced)
            notes.append(f"  {name}: {overlap * 100:.1f}% of {source_stem}")
            if overlap < SAME_PATIENT_MIN_OVERLAP:
                raise SystemExit(
                    f"{source_stem} keep-one would be written as {name}, but the "
                    f"{name} already in the dataset overlaps it by only "
                    f"{overlap * 100:.1f}% -- that is a different vessel, not "
                    f"another view of this one. Refusing to overwrite it."
                )
    return notes


def choose_sources(
    original: dict[str, Path],
    cleaned: dict[str, Path],
    uncapped: dict[str, Path],
    slots,
) -> list[tuple[str, str, Path, str]]:
    """Return (dest_name, vessel_id, src_path, source_label) rows."""
    # Which original stem each keep-one output takes the place of.
    by_slot = {
        name: (source_stem, path)
        for source_stem, entries in slots.items()
        for name, path in entries
    }

    missing = sorted(set(slots) - set(original))
    if missing:
        raise SystemExit(
            "Keep-one vessel(s) not in original (refusing to guess): "
            + ", ".join(missing)
        )

    rows: list[tuple[str, str, Path, str]] = []
    for vessel_id in sorted(original):
        if vessel_id in by_slot:
            source_stem, path = by_slot[vessel_id]
            rows.append((path.name, source_stem, path, "keep_one"))
        elif vessel_id in slots:
            # This vessel's keep-one outputs are named something else (p... and
            # SNF... take a suffix), and they represent it. Its own un-split
            # mesh would be the same vessel a second time, still carrying every
            # aneurysm, so it is left out exactly as before.
            continue
        elif vessel_id in cleaned:
            rows.append((cleaned[vessel_id].name, vessel_id, cleaned[vessel_id], "cleaned_and_decapped"))
        elif vessel_id in uncapped:
            rows.append((uncapped[vessel_id].name, vessel_id, uncapped[vessel_id], "uncapped"))
        else:
            rows.append((original[vessel_id].name, vessel_id, original[vessel_id], "original"))

    # Keep-one outputs whose name is not an original stem of its own: the
    # suffixed p.../SNF... ones, and a slot like C0028c that the dataset never
    # had because it only ever named two aneurysms there.
    for name, (source_stem, path) in sorted(by_slot.items()):
        if name not in original:
            rows.append((path.name, source_stem, path, "keep_one"))

    dest_names = [row[0] for row in rows]
    if len(dest_names) != len(set(dest_names)):
        clashes = sorted({n for n in dest_names if dest_names.count(n) > 1})
        raise SystemExit("Destination filename collision: " + ", ".join(clashes))
    return rows


def copy_rows(rows, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for path in [p for p in dest_dir.iterdir() if p.is_file()]:
        path.unlink()
    for i, (dest_name, _vessel_id, src, _label) in enumerate(rows, start=1):
        shutil.copy2(src, dest_dir / dest_name)
        if i % 50 == 0 or i == len(rows):
            print(f"  copied {i}/{len(rows)}", flush=True)


def write_manifest(rows, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dest_name", "vessel_id", "source", "src_path"])
        for dest_name, vessel_id, src, label in rows:
            writer.writerow([dest_name, vessel_id, label, str(src)])


def main(dry_run: bool = False) -> None:
    original = list_meshes(Path(VESSELS_ORIGINAL))
    cleaned = list_meshes(Path(CLEANED_VESSELS))
    uncapped = list_meshes(Path(UNCAPPED_VESSELS))
    keep_one = list_meshes(KEEP_ONE_SURFACES)

    slots = keep_one_slots(PICKED_POINTS, keep_one)
    print(f"original stems: {len(original)}")
    print(f"keep_one files: {len(keep_one)} across {len(slots)} vessels")
    print(f"cleaned files: {len(cleaned)}")
    print(f"uncapped files: {len(uncapped)}")

    print("\nkeep-one outputs that take a name the dataset already uses:")
    notes = check_displaced_slots(slots, original)
    for note in notes or ["  (none)"]:
        print(note)

    rows = choose_sources(original, cleaned, uncapped, slots)

    files_by_source: dict[str, int] = defaultdict(int)
    vessels_by_source: dict[str, set] = defaultdict(set)
    for _dest, vid, _src, label in rows:
        files_by_source[label] += 1
        vessels_by_source[label].add(vid)
    print("\nvessels chosen from:")
    for label in ("keep_one", "cleaned_and_decapped", "uncapped", "original"):
        print(f"  {label}: {len(vessels_by_source[label])} vessels, "
              f"{files_by_source[label]} files")
    print(f"destination files: {len(rows)}")

    if dry_run:
        print("\ndry run: nothing copied")
        return

    print(f"copying into {DEST_DIR}")
    copy_rows(rows, DEST_DIR)
    write_manifest(rows, MANIFEST_PATH)
    written = [p for p in DEST_DIR.iterdir() if p.is_file() and p.suffix.lower() in MESH_EXTS]
    if len(written) != len(rows):
        raise SystemExit(f"Expected {len(rows)} dest files, found {len(written)}")
    print(f"wrote manifest {MANIFEST_PATH}")
    print("done")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv[1:])
