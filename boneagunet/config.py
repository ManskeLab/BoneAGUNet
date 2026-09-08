from __future__ import annotations

import json
import os
from pathlib import Path

HF_REPO = "YousifKhoury/BoneAGUNet"
SUPPORTED_MCPS = (2, 3)
CONFIG_PATH = Path(os.environ.get("BONEAGUNET_HOME", Path.home() / ".boneagunet")) / "config.json"

# Each directory is an independent nnUNet_results root. This avoids collisions:
# three trained datasets are all named Dataset001_mcp.
MODEL_SPECS = {
    "strip": ("Dataset001_hand", "nnUNetTrainer", "checkpoint_best.pth"),
    "edge": ("Dataset001_mcp", "nnUNetTrainer", "checkpoint_final.pth"),
    "closed_edge": ("Dataset001_mcp", "nnUNetTrainerWithAttention", "checkpoint_final.pth"),
    "erosion": ("Dataset001_mcp", "nnUNetTrainerWithAttention", "checkpoint_final.pth"),
}


def save_asset_root(path: str | Path) -> Path:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({"asset_root": str(Path(path).resolve())}, indent=2))
    return CONFIG_PATH


def resolve_asset_root(explicit: str | Path | None = None) -> Path:
    if explicit:
        root = Path(explicit).expanduser()
    elif os.environ.get("BONEAGUNET_ASSETS"):
        root = Path(os.environ["BONEAGUNET_ASSETS"]).expanduser()
    elif CONFIG_PATH.exists():
        root = Path(json.loads(CONFIG_PATH.read_text())["asset_root"]).expanduser()
    else:
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(HF_REPO))
    validate_assets(root)
    return root


def model_dir(root: Path, name: str) -> Path:
    dataset, trainer, _ = MODEL_SPECS[name]
    return root / "models" / name / dataset / f"{trainer}__nnUNetPlans__3d_fullres"


def validate_assets(root: Path) -> None:
    missing = []
    for name, (_, _, checkpoint) in MODEL_SPECS.items():
        directory = model_dir(root, name)
        for relative in ("dataset.json", "plans.json", f"fold_all/{checkpoint}"):
            if not (directory / relative).is_file():
                missing.append(str(directory / relative))
    for mcp in SUPPORTED_MCPS:
        for bone in ("mc", "pp"):
            path = root / "atlases" / f"mcp{mcp}" / f"atlas_{bone}.nii.gz"
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("BoneAGUNet asset bundle is incomplete:\n  " + "\n  ".join(missing))

