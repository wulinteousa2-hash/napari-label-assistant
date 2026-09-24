"""Disk-backed, seam-aware connected components for very large 2D masks.

Tiles are an I/O and memory boundary only. Provisional components that touch
across a tile edge are joined before records or Euler numbers are published.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import numpy as np
from skimage.measure import label, regionprops


LARGE_ANALYSIS_PIXEL_THRESHOLD = 20_000_000
LARGE_ANALYSIS_TILE_SIZE = 2048
LARGE_TABLE_PAGE_SIZE = 1000


@dataclass
class LargeComponentRecord:
    component_id: int
    label_value: int
    area: int
    euler: int
    centroid_y: float
    centroid_x: float
    bbox_y0: int
    bbox_x0: int
    bbox_y1: int
    bbox_x1: int
    grid_row: int | None = None
    grid_col: int | None = None
    grid_id: str = ""

    @property
    def bbox_text(self) -> str:
        return (
            f"y[{self.bbox_y0}:{self.bbox_y1}] "
            f"x[{self.bbox_x0}:{self.bbox_x1}]"
        )


@dataclass(frozen=True)
class _Tile:
    y0: int
    y1: int
    x0: int
    x1: int
    base: int
    count: int
    path: Path


@dataclass
class _Fragment:
    label_value: int
    area: int
    sum_y: float
    sum_x: float
    y0: int
    x0: int
    y1: int
    x1: int


class LargeAnalysisCancelled(Exception):
    """Raised when the user cancels a tile-based analysis."""


class _UnionFind:
    def __init__(self) -> None:
        self.parent = [0]
        self.rank = [0]

    def add(self) -> int:
        index = len(self.parent)
        self.parent.append(index)
        self.rank.append(0)
        return index

    def find(self, index: int) -> int:
        parent = self.parent
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(self, a: int, b: int) -> None:
        ra, rb = self.find(int(a)), self.find(int(b))
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _join_seam(
    union: _UnionFind,
    previous_ids: np.ndarray | None,
    previous_values: np.ndarray | None,
    current_ids: np.ndarray,
    current_values: np.ndarray,
) -> None:
    if previous_ids is None or previous_values is None:
        return
    touching = (
        (previous_ids > 0)
        & (current_ids > 0)
        & (previous_values == current_values)
    )
    if not np.any(touching):
        return
    pairs = np.unique(
        np.column_stack((previous_ids[touching], current_ids[touching])),
        axis=0,
    )
    for left, right in pairs:
        union.join(int(left), int(right))


def _euler_numerators(block: np.ndarray, count: int) -> np.ndarray:
    """Return 4*Euler for each ID using all 2x2 pixel configurations.

The +2 diagonal term is the 4-connected-foreground convention used by the
small-image analysis. Each 2x2 window is owned by its bottom-right pixel.
"""
    a = block[:-1, :-1]
    b = block[:-1, 1:]
    c = block[1:, :-1]
    d = block[1:, 1:]
    corners = (a, b, c, d)
    result = np.zeros(count + 1, dtype=np.int64)
    for position, ids in enumerate(corners):
        first = ids > 0
        for earlier in corners[:position]:
            first &= ids != earlier
        if not np.any(first):
            continue
        occurrences = np.zeros(ids.shape, dtype=np.uint8)
        for corner in corners:
            occurrences += corner == ids
        diagonal = ((a == d) & (ids == a)) | ((b == c) & (ids == b))
        weights = np.zeros(ids.shape, dtype=np.int8)
        weights[occurrences == 1] = 1
        weights[occurrences == 3] = -1
        weights[(occurrences == 2) & diagonal] = -2
        keep = first & (weights != 0)
        if np.any(keep):
            result += np.bincount(
                ids[keep].ravel(),
                weights=weights[keep].ravel(),
                minlength=count + 1,
            ).astype(np.int64)
    return result


class LargeComponentIndex:
    """Small metadata index plus on-disk tile labels; no full-size ID map."""

    def __init__(
        self,
        shape: tuple[int, int],
        tile_size: int,
        cache: tempfile.TemporaryDirectory,
        tiles: list[_Tile],
        provisional_to_final: np.ndarray,
        records: dict[int, LargeComponentRecord],
    ) -> None:
        self.shape = shape
        self.tile_size = tile_size
        self._cache = cache
        self.tiles = tiles
        self._tile_columns = (shape[1] + tile_size - 1) // tile_size
        self.provisional_to_final = provisional_to_final
        self.records = records
        self.deleted_component_ids: set[int] = set()

    def close(self) -> None:
        self._cache.cleanup()

    def active_records(self) -> list[LargeComponentRecord]:
        return [
            record
            for component_id, record in sorted(self.records.items())
            if component_id not in self.deleted_component_ids
        ]

    def reassign_grid(self, step_y: int | None, step_x: int | None) -> None:
        for record in self.records.values():
            if step_y is None or step_x is None:
                record.grid_row = record.grid_col = None
                record.grid_id = ""
            else:
                row = int(record.centroid_y // step_y)
                col = int(record.centroid_x // step_x)
                record.grid_row = row
                record.grid_col = col
                record.grid_id = f"R{row:02d}C{col:02d}"

    def _tile_for(self, y: int, x: int) -> _Tile:
        row, col = y // self.tile_size, x // self.tile_size
        return self.tiles[row * self._tile_columns + col]

    def _load_local(self, tile: _Tile) -> np.ndarray:
        if tile.count == 0:
            return np.zeros((tile.y1 - tile.y0, tile.x1 - tile.x0), dtype=np.int32)
        with np.load(tile.path, allow_pickle=False) as stored:
            return stored["labels"]

    def _final_tile(self, tile: _Tile) -> np.ndarray:
        local = self._load_local(tile)
        lookup = np.zeros(tile.count + 1, dtype=np.int32)
        lookup[1:] = self.provisional_to_final[
            tile.base + 1 : tile.base + tile.count + 1
        ]
        return lookup[local]

    def component_id_at(self, coords: tuple[int, int]) -> int | None:
        y, x = map(int, coords)
        if y < 0 or x < 0 or y >= self.shape[0] or x >= self.shape[1]:
            return None
        tile = self._tile_for(y, x)
        if tile.count == 0:
            return None
        local = self._load_local(tile)
        local_id = int(local[y - tile.y0, x - tile.x0])
        if local_id == 0:
            return None
        component_id = int(self.provisional_to_final[tile.base + local_id])
        return component_id if component_id not in self.deleted_component_ids else None

    def selected_tiles(
        self, component_ids: list[int]
    ) -> Iterator[tuple[_Tile, np.ndarray]]:
        selected = {
            int(component_id)
            for component_id in component_ids
            if int(component_id) in self.records
            and int(component_id) not in self.deleted_component_ids
        }
        if not selected:
            return
        bounds = [self.records[component_id] for component_id in selected]
        y0 = min(record.bbox_y0 for record in bounds)
        y1 = max(record.bbox_y1 for record in bounds)
        x0 = min(record.bbox_x0 for record in bounds)
        x1 = max(record.bbox_x1 for record in bounds)
        for tile in self.tiles:
            if tile.y1 <= y0 or tile.y0 >= y1 or tile.x1 <= x0 or tile.x0 >= x1:
                continue
            relevant = np.isin(
                self.provisional_to_final[
                    tile.base + 1 : tile.base + tile.count + 1
                ],
                list(selected),
            )
            if not np.any(relevant):
                continue
            local = self._load_local(tile)
            lookup = np.zeros(tile.count + 1, dtype=bool)
            lookup[1:] = relevant
            mask = lookup[local]
            if np.any(mask):
                yield tile, mask


def build_large_component_index(
    source,
    *,
    tile_size: int = LARGE_ANALYSIS_TILE_SIZE,
    cancel_event=None,
) -> Iterator[tuple[int, int, str]]:
    """Yield progress and return an exact global component index.

    ``source`` must support 2D slicing. No full-image ``np.asarray`` is used.
    """
    shape = tuple(int(value) for value in source.shape)
    if len(shape) != 2 or any(value <= 0 for value in shape):
        raise ValueError("Large-image analysis requires a nonempty 2D mask.")
    tile_size = int(tile_size)
    if tile_size < 2:
        raise ValueError("Tile size must be at least 2 pixels.")
    dtype = np.dtype(source.dtype)
    if dtype.kind not in "biu":
        raise ValueError("Large-image analysis requires an integer/binary mask layer.")

    cache = tempfile.TemporaryDirectory(prefix="label-assistant-components-")
    complete = False
    try:
        cache_path = Path(cache.name)
        tiles: list[_Tile] = []
        fragments: list[_Fragment | None] = [None]
        union = _UnionFind()
        cols = (shape[1] + tile_size - 1) // tile_size
        rows = (shape[0] + tile_size - 1) // tile_size
        total_tiles = rows * cols
        previous_bottom_ids: list[np.ndarray | None] = [None] * cols
        previous_bottom_values: list[np.ndarray | None] = [None] * cols

        for row in range(rows):
            current_bottom_ids: list[np.ndarray | None] = [None] * cols
            current_bottom_values: list[np.ndarray | None] = [None] * cols
            left_ids = None
            left_values = None
            for col in range(cols):
                if cancel_event is not None and cancel_event.is_set():
                    raise LargeAnalysisCancelled()
                y0, x0 = row * tile_size, col * tile_size
                y1, x1 = min(y0 + tile_size, shape[0]), min(x0 + tile_size, shape[1])
                data = np.asarray(source[y0:y1, x0:x1])
                if data.shape != (y1 - y0, x1 - x0):
                    raise ValueError("Mask source returned an unexpected tile shape.")
                if data.dtype.kind not in "biu":
                    raise ValueError("Mask source returned a non-integer tile.")
                local = label(data, background=0, connectivity=1).astype(
                    np.int32, copy=False
                )
                count = int(local.max())
                base = len(fragments) - 1
                for prop in regionprops(local):
                    ly0, lx0, ly1, lx1 = map(int, prop.bbox)
                    local_mask = local[ly0:ly1, lx0:lx1] == prop.label
                    first = int(np.flatnonzero(local_mask)[0])
                    dy, dx = divmod(first, lx1 - lx0)
                    value = int(data[ly0 + dy, lx0 + dx])
                    union.add()
                    fragments.append(
                        _Fragment(
                            value,
                            int(prop.area),
                            float(prop.centroid[0] + y0) * int(prop.area),
                            float(prop.centroid[1] + x0) * int(prop.area),
                            y0 + ly0,
                            x0 + lx0,
                            y0 + ly1,
                            x0 + lx1,
                        )
                    )
                if len(fragments) - 1 != base + count:
                    raise RuntimeError("Tile component numbering is inconsistent.")
                path = cache_path / f"tile_{row:05d}_{col:05d}.npz"
                if count:
                    np.savez_compressed(path, labels=local)
                tiles.append(_Tile(y0, y1, x0, x1, base, count, path))
                top_ids = np.where(local[0] > 0, local[0] + base, 0)
                first_ids = np.where(local[:, 0] > 0, local[:, 0] + base, 0)
                _join_seam(
                    union,
                    previous_bottom_ids[col],
                    previous_bottom_values[col],
                    top_ids,
                    data[0],
                )
                _join_seam(union, left_ids, left_values, first_ids, data[:, 0])
                current_bottom_ids[col] = np.where(
                    local[-1] > 0, local[-1] + base, 0
                )
                current_bottom_values[col] = data[-1].copy()
                left_ids = np.where(local[:, -1] > 0, local[:, -1] + base, 0)
                left_values = data[:, -1].copy()
                done = row * cols + col + 1
                yield done, total_tiles * 2, f"Indexing tile {done:,}/{total_tiles:,}"
            previous_bottom_ids = current_bottom_ids
            previous_bottom_values = current_bottom_values

        merged: dict[int, _Fragment] = {}
        for provisional in range(1, len(fragments)):
            root = union.find(provisional)
            fragment = fragments[provisional]
            if fragment is None:
                continue
            existing = merged.get(root)
            if existing is None:
                merged[root] = replace(fragment)
            else:
                existing.area += fragment.area
                existing.sum_y += fragment.sum_y
                existing.sum_x += fragment.sum_x
                existing.y0 = min(existing.y0, fragment.y0)
                existing.x0 = min(existing.x0, fragment.x0)
                existing.y1 = max(existing.y1, fragment.y1)
                existing.x1 = max(existing.x1, fragment.x1)
        roots = sorted(
            merged, key=lambda root: (
                merged[root].label_value,
                merged[root].y0,
                merged[root].x0,
            )
        )
        root_to_final = {root: i for i, root in enumerate(roots, start=1)}
        provisional_to_final = np.zeros(len(fragments), dtype=np.int32)
        for provisional in range(1, len(fragments)):
            provisional_to_final[provisional] = root_to_final[union.find(provisional)]
        records = {
            final_id: LargeComponentRecord(
                component_id=final_id,
                label_value=merged[root].label_value,
                area=merged[root].area,
                euler=0,
                centroid_y=merged[root].sum_y / merged[root].area,
                centroid_x=merged[root].sum_x / merged[root].area,
                bbox_y0=merged[root].y0,
                bbox_x0=merged[root].x0,
                bbox_y1=merged[root].y1,
                bbox_x1=merged[root].x1,
            )
            for root, final_id in root_to_final.items()
        }
        index = LargeComponentIndex(
            shape, tile_size, cache, tiles, provisional_to_final, records
        )

        numerators = np.zeros(len(records) + 1, dtype=np.int64)
        previous_bottom: list[np.ndarray | None] = [None] * cols
        for row in range(rows):
            current_bottom: list[np.ndarray | None] = [None] * cols
            left = None
            for col in range(cols):
                if cancel_event is not None and cancel_event.is_set():
                    raise LargeAnalysisCancelled()
                tile = tiles[row * cols + col]
                final = index._final_tile(tile)
                height, width = final.shape
                add_bottom = int(tile.y1 == shape[0])
                add_right = int(tile.x1 == shape[1])
                block = np.zeros(
                    (height + 1 + add_bottom, width + 1 + add_right),
                    dtype=np.int32,
                )
                block[1 : height + 1, 1 : width + 1] = final
                if previous_bottom[col] is not None:
                    block[0, 1 : width + 1] = previous_bottom[col]
                if left is not None:
                    block[1 : height + 1, 0] = left
                if row and col:
                    block[0, 0] = previous_bottom[col - 1][-1]
                numerators += _euler_numerators(block, len(records))
                current_bottom[col] = final[-1].copy()
                left = final[:, -1].copy()
                done = row * cols + col + 1
                yield total_tiles + done, total_tiles * 2, (
                    f"Checking ring topology {done:,}/{total_tiles:,}"
                )
            previous_bottom = current_bottom
        if np.any(numerators[1:] % 4):
            raise RuntimeError("Global Euler calculation was inconsistent.")
        for component_id, record in records.items():
            record.euler = int(numerators[component_id] // 4)
        complete = True
        return index
    finally:
        if not complete:
            cache.cleanup()
