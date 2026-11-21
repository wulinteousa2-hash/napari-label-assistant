"""Coordinate API compatibility across napari 0.9 and 0.10."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from napari_label_assistant_tools._editing_canvas import CanvasAdapter


@pytest.mark.parametrize('public_api', [False, True])
def test_coordinate_mapping_api_and_pixel_order(public_api):
    canvas = SimpleNamespace(view=object(), native=object())
    viewer = SimpleNamespace(
        window=SimpleNamespace(_qt_viewer=SimpleNamespace(canvas=canvas)),
        camera=object(), grid=object(),
    )
    mapping = Mock(return_value=(11, 17))
    if public_api:
        viewer.canvas_to_world = mapping
    else:
        canvas._map_canvas2world = mapping
    adapter = CanvasAdapter(viewer)
    assert adapter.world_at((123, 456)) == (11, 17)
    if public_api:
        mapping.assert_called_once_with((456, 123))
    else:
        mapping.assert_called_once_with((123, 456), canvas.view)


def test_public_coordinate_api_takes_precedence():
    canvas = SimpleNamespace(view=object(), native=object(), _map_canvas2world=Mock())
    viewer = SimpleNamespace(
        window=SimpleNamespace(_qt_viewer=SimpleNamespace(canvas=canvas)),
        scene=SimpleNamespace(camera=object()), canvas=SimpleNamespace(grid=object()),
        canvas_to_world=Mock(return_value=(11, 17)),
    )
    assert CanvasAdapter(viewer).world_at((123, 456)) == (11, 17)
    canvas._map_canvas2world.assert_not_called()
