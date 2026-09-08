from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def case_name(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def make_working_copy(source: Path, destination: Path) -> tuple[float, ...]:
    """Copy voxels and geometry, changing only spacing to the trained 1 mm contract."""
    image = sitk.ReadImage(str(source))
    native_spacing = image.GetSpacing()
    image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(image, str(destination), useCompression=True)
    return native_spacing


def apply_strip(image_path: Path, label_path: Path, output_dir: Path, name: str) -> dict[str, Path]:
    image = sitk.ReadImage(str(image_path))
    labels = sitk.ReadImage(str(label_path), sitk.sitkUInt8)
    if image.GetSize() != labels.GetSize():
        raise ValueError("Strip prediction does not match the input grid")
    array = sitk.GetArrayFromImage(image)
    mask = sitk.GetArrayFromImage(labels)
    outputs = {}
    selections = {"stripped": mask > 0, "mc_mask": mask == 1, "pp_mask": mask == 2}
    for key, selection in selections.items():
        data = selection.astype(np.uint8) if key.endswith("mask") else array * selection
        result = sitk.GetImageFromArray(data)
        result.CopyInformation(image)
        path = output_dir / f"{key}_{name}.nii.gz"
        sitk.WriteImage(result, str(path), useCompression=True)
        outputs[key] = path
    return outputs


def split_channels(source: Path, name: str, output_dir: Path) -> None:
    image = sitk.ReadImage(str(source))
    array = sitk.GetArrayFromImage(image)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"Expected a three-channel ROI, got shape {array.shape}")
    for channel in range(3):
        result = sitk.GetImageFromArray(array[..., channel])
        result.SetSpacing(image.GetSpacing()[:3])
        result.SetOrigin(image.GetOrigin()[:3])
        result.SetDirection(image.GetDirection()[:9])
        sitk.WriteImage(result, str(output_dir / f"{name}_000{channel}.nii.gz"), True)
    sample = sitk.ReadImage(str(output_dir / f"{name}_0000.nii.gz"))
    metadata = {"modality": {"0": "CT", "1": "Edge", "2": "Atlas"},
                "spacing": list(reversed(sample.GetSpacing())),
                "shape": list(reversed(sample.GetSize()))}
    (output_dir / f"{name}.json").write_text(json.dumps(metadata, indent=2))


def combine_predictions(prediction_dir: Path, working_ref: Path, native_ref: Path,
                        output_path: Path, binary: bool = False) -> int:
    reference = sitk.ReadImage(str(working_ref))
    native = sitk.ReadImage(str(native_ref))
    if native.GetSize() != reference.GetSize():
        raise ValueError("Native and working images must have the same voxel grid")
    combined = np.zeros(tuple(reversed(reference.GetSize())), dtype=np.uint16)
    placed = 0
    for prediction in sorted(prediction_dir.glob("*.nii.gz")):
        image = sitk.ReadImage(str(prediction))
        resampled = sitk.Resample(image, reference, sitk.Transform(),
                                  sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        foreground = sitk.GetArrayFromImage(resampled) > 0
        if foreground.any():
            placed += 1
            combined[foreground & (combined == 0)] = 1 if binary else placed
    result = sitk.GetImageFromArray(combined)
    result.CopyInformation(native)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(result, str(output_path), useCompression=True)
    return placed

