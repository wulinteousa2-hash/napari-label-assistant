import numpy as np

from napari_label_assistant_tools._large_components import build_large_component_index


class SlicedOnlyMask:
    def __init__(self, data):
        self._data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.reads = []

    def __array__(self, dtype=None):
        raise AssertionError("The full mask must never be materialized")

    def __getitem__(self, key):
        self.reads.append(key)
        return self._data[key]


def test_large_analysis_reads_only_bounded_tiles():
    data = np.zeros((12, 12), dtype=np.uint8)
    data[2:10, 2:10] = 1
    data[4:8, 4:8] = 0
    source = SlicedOnlyMask(data)
    generator = build_large_component_index(source, tile_size=5)
    while True:
        try:
            next(generator)
        except StopIteration as result:
            index = result.value
            break

    try:
        assert len(index.records) == 1
        assert index.records[1].euler == 0
        assert len(source.reads) == 9
        assert all(
            y.stop - y.start <= 5 and x.stop - x.start <= 5
            for y, x in source.reads
        )
    finally:
        index.close()
