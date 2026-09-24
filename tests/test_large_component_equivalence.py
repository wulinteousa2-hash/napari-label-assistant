import numpy as np

from napari_label_assistant_tools._large_components import build_large_component_index
from napari_label_assistant_tools._widget import _build_fast_component_index


def _finish(generator):
    while True:
        try:
            next(generator)
        except StopIteration as result:
            return result.value


def _summaries(records):
    return sorted(
        (
            record.label_value,
            record.area,
            record.euler,
            record.bbox_y0,
            record.bbox_x0,
            record.bbox_y1,
            record.bbox_x1,
            round(record.centroid_y, 6),
            round(record.centroid_x, 6),
        )
        for record in records
    )


def test_tiled_results_match_existing_analysis_for_binary_and_multivalue_masks():
    for seed in range(4):
        rng = np.random.default_rng(seed)
        mask = rng.choice([0, 0, 0, 0, 1, 2], size=(18, 23)).astype(np.uint8)
        small = _build_fast_component_index(mask)
        large = _finish(build_large_component_index(mask, tile_size=5))
        try:
            assert _summaries(large.active_records()) == _summaries(
                small.active_records()
            )
        finally:
            large.close()
