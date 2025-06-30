"""
Utilities for reading and processing the Hoda dataset in CDB format.

This module provides core functions to:
- Parse and decode Hoda .cdb files containing handwritten digit images and labels.
- Handle the run-length encoding (RLE) of images inside the dataset.
- Resize and center images onto a fixed-size canvas while preserving aspect ratio, suitable for ML workflows.
- Batch-process multiple CDB files and save the merged dataset into a single compressed .npz file.
"""

from __future__ import annotations
import struct
from pathlib import Path
from typing import Final, List, Tuple
from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


# Total length (in bytes) of the *file-level* header that precedes the first
# record in every Hoda *.cdb* file.  It is the sum of five fixed-size blocks
# defined in the dataset specification:
#
#   10    ── core header fields
#   128*4 ── class-name table     (128 entries x 4-byte ID)
#   1     ── image-type flag      (0 = binary RLE, 1 = 8-bit grayscale)
#   256   ── reserved padding
#   245   ── copyright string
# -----------------------------------------------
#   1024 bytes in total
#
# The parser skips exactly this many bytes so that the cursor points at the
# start of the first image record.
_HEADER_BYTES: Final[int] = 10 + 128 * 4 + 1 + 256 + 245  # = 1024


def _decode_rle(
    file_view: memoryview, start_offset: int, width: int, height: int
) -> tuple[NDArray[np.uint8], int]:
    """
    Expand one run-length-encoded (RLE) binary image stored in a Hoda *.cdb file.

    Parameters
    ----------
    file_view : memoryview
        Zero-copy view of the entire file.
    start_offset : int
        Byte position where this image's RLE stream begins.
    width : int
        Pixel width of th(magic, version, etc.)e image.
    height : int
        Pixel height of the image.

    Returns
    -------
    decoded_image : NDArray[np.uint8]
        Array of shape ``(height, width)`` filled with 0 (white)
        and 255 (black).
    bytes_consumed : int
        Exact number of bytes that were read from *file_view*.
    """
    decoded_image: NDArray[np.uint8] = np.zeros((height, width), dtype=np.uint8)

    # Pointer that walks through the file buffer while we decode.
    read_position: int = start_offset

    # The RLE stream is organised row-by-row.
    # Each *run* byte tells how many consecutive pixels in the current
    # color follow next; the color toggles after every run.
    for row_index in range(height):
        column_index: int = 0

        # Every RLE row starts with a *white* run.
        # If the actual first pixel is black, that run’s length is 0,
        # so we still initialise the flag to “white” and the first toggle
        # will immediately switch to black.
        is_current_run_white: bool = True

        while column_index < width:
            run_length = file_view[read_position]
            read_position += 1
            pixel_value: int = 0 if is_current_run_white else 255
            decoded_image[row_index, column_index : column_index + run_length] = (
                pixel_value
            )

            # Move the column pointer and toggle the color for next run.
            column_index += run_length
            is_current_run_white = not is_current_run_white

    return decoded_image, read_position - start_offset


def read_hoda_cdb(path: str | Path) -> Tuple[List[NDArray[np.uint8]], List[int]]:
    """
    Parse a *.cdb file from the Hoda handwritten-digit dataset.

    Parameters
    ----------
    path : str or Path
        File location.

    Returns
    -------
    images : list of 2-D uint8 arrays
    labels : list of int
    """
    file_bytes = Path(path).read_bytes()
    file_view = memoryview(file_bytes)

    # Global header
    initial_width, initial_height = struct.unpack_from("2B", file_bytes, 4)
    total_records: int = struct.unpack_from("<I", file_bytes, 6)[0]
    fixed_size: bool = initial_width * initial_height > 0
    image_type: int = file_bytes[10 + 128 * 4]  # 0=binary RLE, 1=grayscale raw

    images: List[NDArray[np.uint8]] = []
    labels: List[int] = []

    cursor = _HEADER_BYTES

    for _ in range(total_records):
        cursor += 1  # Each record starts with 0xFF which is ignored
        label = file_bytes[cursor]
        labels.append(label)

        cursor += 1
        width, height = file_bytes[cursor], file_bytes[cursor + 1]

        if fixed_size:
            width, height = initial_width, initial_height
        else:
            width, height = file_bytes[cursor], file_bytes[cursor + 1]
            cursor += 2

        cursor += 2  # Two bytes that the spec defines but the loader doesn’t need.

        if width == 0 or height == 0:
            raise ValueError(f"Corrupt record: width={width}, height={height}")

        if image_type == 0:  # binary, run-length encoded
            image, bytes_used = _decode_rle(file_view, cursor, width, height)
            cursor += bytes_used
        else:  # raw 8-bit grayscale
            pixel_count: int = width * height
            image = np.frombuffer(
                file_bytes, dtype=np.uint8, count=pixel_count, offset=cursor
            ).reshape(width, height)
            cursor += pixel_count

        images.append(image)

    return images, labels


def _resize_center(
    image: NDArray[np.uint8], target_height: int, target_width: int
) -> NDArray[np.uint8]:
    """
    Resize *image* so that it fits inside ``(target_height, target_width)`` while
    preserving aspect-ratio, then embed the result in the centre of a black
    canvas of exactly that size.

    Parameters
    ----------
    image : NDArray[np.uint8]
        Source bitmap (2-D, single-channel, 0-255).
    target_height : int
        Desired output height in pixels.
    target_width : int
        Desired output width in pixels.

    Returns
    -------
    NDArray[np.uint8]
        A new array of shape ``(target_height, target_width)`` with the resized
        digit centred and the remaining area filled with 0 (black).
    """

    # Down-scale only when the image is larger than the target
    # Choose the same factor for both axes so aspect ratio is kept
    if image.shape[0] > target_height or image.shape[1] > target_width:
        scale = min(target_height / image.shape[0], target_width / image.shape[1])
    else:
        scale = 1

    resized = cv2.resize(
        image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )

    canvas = np.zeros((target_height, target_width), dtype=np.uint8)
    y_offset = (target_height - resized.shape[0]) // 2
    x_offset = (target_width - resized.shape[1]) // 2
    canvas[
        y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]
    ] = resized

    return canvas


def process_hoda_dataset(
    paths: Sequence[str | Path],
    save_as: str | Path,
    size: Tuple[int, int] = (28, 28),
    *,
    normalize: bool = False,
) -> None:
    """
    Read several Hoda *.cdb* files, concatenate the samples, and write
    the result to a single, easy-to-load NPZ file.

    Parameters
    ----------
    paths : list[str | Path]
        File names (or Path objects) of the source *.cdb* files.
    save_as : str | Path
        An ``*.npz`` archive containing two
        arrays -``images`` (object array of 2-D uint8 matrices) and
        ``labels`` (uint8)- is written to this location with gzip
        compression.

    Returns
    -------
    images : list[NDArray[np.uint8]]
        All images, in the original order in which the files were supplied.
    labels : list[int]
        Corresponding class labels.
    """
    image_list: List[NDArray[np.uint8]] = []
    label_list: List[int] = []

    for path in paths:
        img_block, lbl_block = read_hoda_cdb(path)
        image_list.extend(img_block)
        label_list.extend(lbl_block)

    height, width = size

    images = np.stack(
        [_resize_center(img, height, width).astype(np.float32) for img in image_list]
    )[..., None]

    if normalize:
        images /= 255.0

    labels = np.asarray(label_list, dtype=np.int16)

    np.savez_compressed(save_as, images=images, labels=labels)
