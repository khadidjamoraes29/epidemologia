from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoxCountingResult:
    dimension: float
    box_sizes: np.ndarray
    counts: np.ndarray
    log_inverse_scale: np.ndarray
    log_counts: np.ndarray


def _count_occupied_boxes(binary: np.ndarray, box_size: int) -> int:
    rows, cols = binary.shape
    padded_rows = int(np.ceil(rows / box_size) * box_size)
    padded_cols = int(np.ceil(cols / box_size) * box_size)

    padded = np.zeros((padded_rows, padded_cols), dtype=bool)
    padded[:rows, :cols] = binary
    blocks = padded.reshape(
        padded_rows // box_size,
        box_size,
        padded_cols // box_size,
        box_size,
    )
    return int(blocks.any(axis=(1, 3)).sum())


def box_counting_dimension(binary_mask: np.ndarray) -> BoxCountingResult:
    binary = np.asarray(binary_mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("A máscara deve ser bidimensional.")
    if not binary.any():
        empty_i = np.array([], dtype=int)
        empty_f = np.array([], dtype=float)
        return BoxCountingResult(0.0, empty_i, empty_i, empty_f, empty_f)

    min_side = min(binary.shape)
    max_power = int(np.floor(np.log2(min_side)))
    box_sizes = 2 ** np.arange(max_power + 1)
    counts = np.array(
        [_count_occupied_boxes(binary, int(size)) for size in box_sizes],
        dtype=int,
    )

    valid = counts > 0
    box_sizes = box_sizes[valid]
    counts = counts[valid]
    epsilon = box_sizes.astype(float) / float(max(binary.shape))
    x = np.log(1.0 / epsilon)
    y = np.log(counts.astype(float))

    dimension = 0.0 if len(x) < 2 else max(0.0, float(np.polyfit(x, y, 1)[0]))
    return BoxCountingResult(dimension, box_sizes, counts, x, y)
