from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import HF_REPO, save_asset_root, validate_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BoneAGUNet models and MCP2/3 atlases")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repo", default=HF_REPO, help="Hugging Face model repository")
    args = parser.parse_args()
    root = args.directory.expanduser().resolve()
    snapshot_download(
        repo_id=args.repo,
        local_dir=root,
        allow_patterns=["models/**", "atlases/mcp2/**", "atlases/mcp3/**", "manifest.json"],
    )
    validate_assets(root)
    config_path = save_asset_root(root)
    print(f"Assets installed in {root}\nConfiguration saved to {config_path}")

