from napari_label_assistant_tools._grid_map import (
    build_visible_grid_label_layout,
)


def _layout(*, zoom: float, max_labels: int = 150):
    return build_visible_grid_label_layout(
        (1000, 1000),
        100,
        100,
        center_y=500,
        center_x=500,
        viewport_height_px=1000,
        viewport_width_px=1000,
        zoom=zoom,
        max_labels=max_labels,
    )


def test_semantic_grid_shows_every_id_when_cells_are_large():
    layout = _layout(zoom=1.0)

    assert layout.stride == 1
    assert layout.opacity == 1.0
    assert len(layout.labels) == 100
    assert layout.labels[0] == "R00C00"
    assert layout.labels[-1] == "R09C09"


def test_semantic_grid_sparsifies_labels_at_mid_zoom():
    layout = _layout(zoom=0.5)

    assert layout.stride == 2
    assert layout.opacity == 0.82
    assert 0 < len(layout.labels) <= 150
    assert "R02C02" in layout.labels
    assert "R01C01" not in layout.labels


def test_semantic_grid_hides_labels_when_cells_are_too_small():
    layout = _layout(zoom=0.1)

    assert layout.labels == ()
    assert layout.stride == 0
    assert layout.opacity == 0.0


def test_semantic_grid_caps_labels_in_the_visible_viewport():
    layout = build_visible_grid_label_layout(
        (10_000, 10_000),
        100,
        100,
        center_y=5000,
        center_x=5000,
        viewport_height_px=1000,
        viewport_width_px=1000,
        zoom=1.0,
        max_labels=25,
    )

    assert 0 < len(layout.labels) <= 25
    assert layout.stride > 1
