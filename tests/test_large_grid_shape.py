from napari_label_assistant_tools._grid_map import build_visible_grid_label_layout
from napari_label_assistant_tools._widget import _build_grid_shapes


def test_grid_overlay_geometry_for_reported_84k_mask_is_bounded():
    shape = (84_175, 79_966)
    lines = _build_grid_shapes(shape, 100, 100)
    labels = build_visible_grid_label_layout(
        shape,
        100,
        100,
        center_y=42_000,
        center_x=40_000,
        viewport_height_px=1000,
        viewport_width_px=1000,
        zoom=1.0,
    )

    assert len(lines) == 1644
    assert 0 < len(labels.labels) <= 150
