import numpy as np

import napari_label_assistant_tools._widget as widget_module
from napari_label_assistant_tools._mask_validation import looks_like_grayscale_image


def test_grayscale_check_uses_bounded_samples():
    class LargeSource:
        shape = (84175, 79966)

        def __init__(self):
            self.read_shapes = []

        def __getitem__(self, key):
            height = key[0].stop - key[0].start
            width = key[1].stop - key[1].start
            self.read_shapes.append((height, width))
            return np.arange(height * width, dtype=np.uint8).reshape(height, width)

    source = LargeSource()
    assert looks_like_grayscale_image(source)
    assert source.read_shapes == [(256, 256)]


def test_component_target_defaults_to_labels_and_rejects_grayscale_image(
    make_napari_viewer, monkeypatch
):
    monkeypatch.setattr(widget_module, "LARGE_ANALYSIS_PIXEL_THRESHOLD", 100)
    viewer = make_napari_viewer()
    image = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    viewer.add_image(image, name="EM image")
    viewer.add_labels(np.zeros_like(image), name="segmentation")

    widget = widget_module.component_operations_widget(viewer)
    assert widget._target_combo.currentText() == "segmentation"

    widget._target_combo.setCurrentText("EM image")
    widget._analyze_button.click()
    assert widget._large_controller.worker is None
    assert "grayscale image" in widget._status_label.text()
    assert "Labels layer" in widget._status_label.text()


def test_binary_image_mask_is_not_rejected():
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[50:400, 100:200] = 255
    assert not looks_like_grayscale_image(mask)
