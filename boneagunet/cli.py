from __future__ import annotations

import argparse

from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment erosions in a cropped MCP2 or MCP3 image")
    parser.add_argument("-i", "--input", required=True, help="cropped MCP joint NIfTI")
    parser.add_argument("-o", "--output", required=True, help="output erosion mask (.nii.gz)")
    parser.add_argument("--mcp", required=True, type=int, choices=(2, 3))
    parser.add_argument("--modality", choices=("hrpqct", "sr-cbct"), default="hrpqct")
    parser.add_argument("--assets", help="override installed asset directory")
    parser.add_argument("--work-dir", help="keep intermediates here (default: temporary directory)")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, help="ITK/ANTs thread count")
    parser.add_argument("--binary", action="store_true", help="write one binary erosion label")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    run(args.input, args.output, mcp=args.mcp, modality=args.modality,
        assets=args.assets, work_dir=args.work_dir, device=args.device,
        keep_work=args.keep_work, threads=args.threads, binary=args.binary)
