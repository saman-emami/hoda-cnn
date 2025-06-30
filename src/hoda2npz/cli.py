#!/usr/bin/env python3
"""
Hoda Dataset Processor - Command-Line Interface

Convert one or more Hoda *.cdb* files into a single compressed NumPy
archive (*.npz*). The resulting archive contains

    images : (N, H, W, 1)  float32
    labels : (N,)          int16

where *N* is the total number of samples collected from all input files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, List

from .cdb_processor import process_hoda_dataset


# validation helpers
def _validate_cdb_files(path_strings: List[str]) -> List[Path]:
    """Ensure every given path exists and ends with '.cdb'."""
    validated_paths: List[Path] = []
    for p in path_strings:
        path = Path(p)
        if path.suffix.lower() != ".cdb":
            raise ValueError(f"Expected a *.cdb file, got: {path}")
        if not path.exists():
            raise FileNotFoundError(f"CDB file not found: {path}")

        validated_paths.append(path)
    return validated_paths


def _validate_output(path_strings: str) -> Path:
    """
    Return a Path whose parent exists and whose suffix is '.npz'
    for the output.
    """
    output_path = Path(path_strings)
    if output_path.suffix.lower() != ".npz":
        output_path = output_path.with_suffix(".npz")
    if not output_path.parent.exists():
        raise ValueError(f"Output directory does not exist: {output_path.parent}")
    return output_path


def _parse_size(text: str) -> Tuple[int, int]:
    """Convert 'WxH' into (H, W) integers and raise on invalid forms."""
    try:
        width, height = map(int, text.lower().split("x"))
        if width <= 0 or height <= 0:
            raise ValueError
    except Exception:
        raise ValueError("SIZE must be of the form WIDTHxHEIGHT, e.g. 28x28")
    return height, width


#  CLI entry
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hoda2npz",
        description="Convert Hoda *.cdb files to a single *.npz archive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "cdb_files",
        nargs="+",
        metavar="CDB_FILE",
        help="One or more *.cdb files to combine",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="OUTPUT.npz",
        help="Destination *.npz file",
    )
    parser.add_argument(
        "--size",
        default="28x28",
        metavar="WxH",
        help="Resize and centre digits inside WIDTHxHEIGHT " "(default 28x28)",
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Scale pixel values from 0-255 to 0-1"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print extra progress information"
    )

    args = parser.parse_args()

    try:
        # validation
        inputs = _validate_cdb_files(args.cdb_files)
        out_path = _validate_output(args.output)
        target_size = _parse_size(args.size)

        if args.verbose:
            print("Input files :", [str(p) for p in inputs])
            print("Output file :", out_path)
            print("Target size :", f"{target_size[1]}x{target_size[0]}")
            print("Normalize   :", args.normalize)
            print()

        # processing
        print("Processing…")
        process_hoda_dataset(
            paths=inputs,
            save_as=out_path,
            size=target_size,
            normalize=args.normalize,
        )
        print(f"✓ Saved → {out_path}")

        if args.verbose:
            print(f"Archive size: {out_path.stat().st_size/1_048_576:.1f} MB")

    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
