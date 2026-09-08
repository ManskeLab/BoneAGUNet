from __future__ import annotations

import re
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path

import SimpleITK as sitk

from . import candidates
from .config import SUPPORTED_MCPS, resolve_asset_root
from .imaging import apply_strip, case_name, combine_predictions, make_working_copy, split_channels
from .nnunet import predict, predict_directory
from .registration import register_atlas


def _validate_input(path: Path) -> sitk.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3-D joint stack, got {image.GetDimension()} dimensions")
    if min(image.GetSize()) < 2:
        raise ValueError(f"Invalid image size: {image.GetSize()}")
    return image


def _safe_case(path: Path) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", case_name(path)).strip("_")
    return value or "mcp"


def run(input_path, output_path, *, mcp: int, modality: str = "hrpqct",
        assets=None, work_dir=None, device: str = "cpu", keep_work: bool = False,
        threads: int | None = None, binary: bool = False):
    """Run the complete atlas-guided pipeline on one unprocessed MCP volume."""
    if mcp not in SUPPORTED_MCPS:
        raise ValueError(f"mcp must be one of {SUPPORTED_MCPS}, got {mcp}")
    if modality not in {"hrpqct", "sr-cbct"}:
        raise ValueError("modality must be 'hrpqct' or 'sr-cbct'")
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    _validate_input(source)
    asset_root = resolve_asset_root(assets)
    for executable in ("nnUNetv2_predict", "antsRegistration", "antsApplyTransforms"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required executable not found on PATH: {executable}")

    if work_dir:
        parent = Path(work_dir).expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix=f"{_safe_case(source)}-", dir=parent))
        context = nullcontext(str(root))
    elif keep_work:
        root = destination.parent / f".{_safe_case(source)}_boneagunet_work"
        root.mkdir(parents=True, exist_ok=True)
        context = nullcontext(str(root))
    else:
        context = tempfile.TemporaryDirectory(prefix="boneagunet-")

    with context as temporary:
        root = Path(temporary)
        name = _safe_case(source)
        working = root / f"{name}_1mm.nii.gz"
        native_spacing = make_working_copy(source, working)

        strip_dir = root / "01_strip"
        strip_dir.mkdir(exist_ok=True)
        strip_prediction = predict("strip", [working], strip_dir / "prediction",
                                   asset_root, device, name)
        stripped = apply_strip(working, strip_prediction, strip_dir, name)

        edge_path = predict("edge", [stripped["stripped"]], root / "02_edge",
                            asset_root, device, name)
        closed_path = predict("closed_edge", [stripped["stripped"], edge_path],
                              root / "03_closed_edge", asset_root, device, name)

        candidate_dirs = {}
        for bone in ("mc", "pp"):
            registration_dir = root / "04_registration" / bone
            registration_dir.mkdir(parents=True, exist_ok=True)
            registered_atlas = registration_dir / f"atlas_{bone}_registered.nii.gz"
            register_atlas(stripped["stripped"], stripped[f"{bone}_mask"],
                           asset_root / "atlases" / f"mcp{mcp}" / f"atlas_{bone}.nii.gz",
                           registered_atlas, registration_dir, threads)
            candidate_dir = root / "05_candidates" / bone
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidates.segment_erosions(
                str(registered_atlas), str(stripped["stripped"]), str(edge_path),
                str(closed_path), modality == "sr-cbct", str(candidate_dir),
                native_spacing=native_spacing)
            candidate_dirs[bone] = candidate_dir

        erosion_inputs = root / "06_erosion_inputs"
        erosion_inputs.mkdir(exist_ok=True)
        for bone, candidate_dir in candidate_dirs.items():
            for index, roi in enumerate(sorted(candidate_dir.glob("*_input*.nii.gz")), start=1):
                split_channels(roi, f"{name}_{bone.upper()}_erosion{index}", erosion_inputs)

        predictions = root / "07_erosion_predictions"
        predict_directory("erosion", erosion_inputs, predictions, asset_root, device)
        count = combine_predictions(predictions, stripped["stripped"], source,
                                    destination, binary=binary)
        return {"output": str(destination), "erosions": count,
                "mcp": mcp, "native_spacing": native_spacing,
                "work_dir": str(root) if (work_dir or keep_work) else None}
