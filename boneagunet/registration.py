from __future__ import annotations

import os
import subprocess
from pathlib import Path

import SimpleITK as sitk


def register_atlas(image_path: Path, mask_path: Path, atlas_path: Path,
                   output_path: Path, work_dir: Path, threads: int | None = None) -> None:
    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
    atlas = sitk.ReadImage(str(atlas_path), sitk.sitkFloat32)
    masked = sitk.Mask(image, mask)
    matched = sitk.HistogramMatching(masked, atlas, 255, 64, True)
    fixed_path = work_dir / "masked_histogram_matched.nii.gz"
    sitk.WriteImage(matched, str(fixed_path), True)
    prefix = work_dir / "atlas_to_input_"
    warped = work_dir / "atlas_warped.nii.gz"
    environment = os.environ.copy()
    if threads:
        environment["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    subprocess.run([
        "antsRegistration", "--dimensionality", "3", "--float", "1",
        "--collapse-output-transforms", "1", "--output", f"[{prefix},{warped}]",
        "--interpolation", "BSpline[5]", "--use-histogram-matching", "1",
        "--winsorize-image-intensities", "[0.8,1]",
        "--initial-moving-transform", f"[{fixed_path},{atlas_path},1]",
        "--transform", "Similarity[0.1]", "--metric", f"MI[{fixed_path},{atlas_path},1,32,Regular,0.25]",
        "--convergence", "[150x100x50x0,1e-6,10]", "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "4x2x1x0vox",
        "--transform", "Affine[0.1]", "--metric", f"MI[{fixed_path},{atlas_path},1,32,Regular,0.25]",
        "--convergence", "[150x100x50x0,1e-6,10]", "--shrink-factors", "6x4x2x1",
        "--smoothing-sigmas", "4x2x1x0vox",
        "--transform", "SyN[0.2,3,0.25]", "--metric", f"MI[{fixed_path},{atlas_path},1,32,Regular,0.25]",
        "--convergence", "[120x100x70,1e-6,10]", "--shrink-factors", "16x8x4",
        "--smoothing-sigmas", "2x1x0vox"], env=environment, check=True)
    subprocess.run(["antsApplyTransforms", "-d", "3", "--float", "1",
                    "-i", str(atlas_path), "-o", str(output_path), "-r", str(image_path),
                    "-t", str(work_dir / "atlas_to_input_1Warp.nii.gz"),
                    "-t", str(work_dir / "atlas_to_input_0GenericAffine.mat")],
                   env=environment, check=True)
