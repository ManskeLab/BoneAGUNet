from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .config import MODEL_SPECS


def predict(model: str, inputs: list[Path], output_dir: Path, asset_root: Path,
            device: str, case: str) -> Path:
    dataset, trainer, checkpoint = MODEL_SPECS[model]
    input_dir = output_dir.parent / f"{output_dir.name}_input"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for channel, source in enumerate(inputs):
        target = input_dir / f"{case}_{channel:04d}.nii.gz"
        shutil.copy2(source, target)
    environment = os.environ.copy()
    environment["nnUNet_results"] = str(asset_root / "models" / model)
    # These are unused for prediction but suppress misleading nnU-Net warnings.
    environment.setdefault("nnUNet_raw", str(output_dir.parent / "nnUNet_raw"))
    environment.setdefault("nnUNet_preprocessed", str(output_dir.parent / "nnUNet_preprocessed"))
    command = ["nnUNetv2_predict", "-d", dataset, "-c", "3d_fullres",
               "-tr", trainer, "-p", "nnUNetPlans", "-f", "all",
               "-chk", checkpoint, "-i", str(input_dir), "-o", str(output_dir),
               "-device", device, "--disable_progress_bar"]
    subprocess.run(command, env=environment, check=True)
    prediction = output_dir / f"{case}.nii.gz"
    if not prediction.is_file():
        raise RuntimeError(f"nnU-Net did not produce {prediction}")
    return prediction


def predict_directory(model: str, input_dir: Path, output_dir: Path,
                      asset_root: Path, device: str) -> None:
    if not any(input_dir.glob("*_0000.nii.gz")):
        output_dir.mkdir(parents=True, exist_ok=True)
        return
    dataset, trainer, checkpoint = MODEL_SPECS[model]
    environment = os.environ.copy()
    environment["nnUNet_results"] = str(asset_root / "models" / model)
    environment.setdefault("nnUNet_raw", str(output_dir.parent / "nnUNet_raw"))
    environment.setdefault("nnUNet_preprocessed", str(output_dir.parent / "nnUNet_preprocessed"))
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["nnUNetv2_predict", "-d", dataset, "-c", "3d_fullres",
                    "-tr", trainer, "-p", "nnUNetPlans", "-f", "all",
                    "-chk", checkpoint, "-i", str(input_dir), "-o", str(output_dir),
                    "-device", device, "--disable_progress_bar"],
                   env=environment, check=True)
