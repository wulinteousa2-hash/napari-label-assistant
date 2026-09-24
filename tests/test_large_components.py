import numpy as np
from skimage.measure import euler_number, label

from napari_label_assistant_tools._large_components import (
    LargeAnalysisCancelled,
    build_large_component_index,
)


def _finish(generator):
    while True:
        try:
            next(generator)
        except StopIteration as result:
            return result.value


def test_tiled_analysis_joins_a_ring_across_both_tile_seams():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:9, 3:9] = 1
    mask[4:8, 4:8] = 0

    index = _finish(build_large_component_index(mask, tile_size=6))
    try:
        assert len(index.records) == 1
        record = index.records[1]
        assert record.area == int(mask.sum())
        assert record.euler == 0
        assert record.centroid_y == 5.5
        assert record.centroid_x == 5.5
        assert index.component_id_at((3, 3)) == 1
        assert index.component_id_at((3, 8)) == 1
        assert index.component_id_at((5, 5)) is None
        tiles = list(index.selected_tiles([1]))
        assert len(tiles) == 4
        assert sum(int(chunk.sum()) for _tile, chunk in tiles) == record.area
    finally:
        index.close()


def test_tiled_analysis_does_not_join_different_values_or_diagonals():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:4, 3] = 1
    mask[4:7, 3] = 2
    mask[3, 4] = 1
    mask[4, 5] = 1

    index = _finish(build_large_component_index(mask, tile_size=4))
    try:
        assert len(index.records) == 3
        assert index.component_id_at((3, 3)) != index.component_id_at((4, 3))
        assert index.component_id_at((3, 3)) == index.component_id_at((3, 4))
        assert sorted(record.euler for record in index.records.values()) == [1, 1, 1]
    finally:
        index.close()


def test_tiled_euler_matches_whole_image_on_random_masks():
    for seed in range(5):
        mask = (np.random.default_rng(seed).random((19, 23)) > 0.73).astype(
            np.uint8
        )
        index = _finish(build_large_component_index(mask, tile_size=6))
        try:
            assert len(index.records) == int(label(mask, connectivity=1).max())
            for component_id, record in index.records.items():
                assert record.euler == euler_number(
                    np.asarray(
                        [
                            [index.component_id_at((y, x)) == component_id for x in range(mask.shape[1])]
                            for y in range(mask.shape[0])
                        ],
                        dtype=bool,
                    ),
                    connectivity=2,
                )
        finally:
            index.close()


def test_tiled_analysis_can_cancel_and_cleanup():
    class Cancel:
        def is_set(self):
            return True

    generator = build_large_component_index(
        np.ones((8, 8), dtype=np.uint8), tile_size=4, cancel_event=Cancel()
    )
    try:
        next(generator)
    except LargeAnalysisCancelled:
        pass
    else:
        raise AssertionError("Expected the analysis to stop before reading tiles")
