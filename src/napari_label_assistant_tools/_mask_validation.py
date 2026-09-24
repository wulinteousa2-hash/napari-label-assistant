"""Cheap, bounded checks before component analysis of an Image layer."""

from __future__ import annotations

import numpy as np


def looks_like_grayscale_image(source, *, sample_size: int = 256) -> bool:
    """Detect dense gray-level texture without materializing a large source.

    This is a conservative warning check, not a segmentation classifier. An
    Image layer with many local values should be converted to a Labels mask
    before its connected components are analyzed.
    """
    height, width = map(int, source.shape)
    size = max(1, int(sample_size))
    for y_fraction, x_fraction in ((0.5, 0.5), (0.25, 0.25), (0.25, 0.75),
                                    (0.75, 0.25), (0.75, 0.75)):
        y0 = max(0, min(height - size, int(height * y_fraction) - size // 2))
        x0 = max(0, min(width - size, int(width * x_fraction) - size // 2))
        sample = np.asarray(source[y0:y0 + size, x0:x0 + size])
        if sample.size and np.unique(sample).size > 64:
            return True
    return False
