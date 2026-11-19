"""Wider custom U-Nets retain configuration, logits, and checkpoint weights."""
from dataclasses import replace

import pytest

from napari_label_assistant_tools._model_providers import (
    CUSTOM_UNET_WIDTHS, ModelSpec, build_model, custom_unet_parameter_count, spec_for,
)
from napari_label_assistant_tools._quick_train import (
    QuickTrainConfig, QuickTrainModel, load_model, save_model,
)


@pytest.mark.parametrize("width", CUSTOM_UNET_WIDTHS)
@pytest.mark.parametrize("outputs", (1, 3, 5))
def test_parameter_count_matches_network_without_allocating_weights(width, outputs):
    torch = pytest.importorskip("torch")
    from napari_label_assistant_tools._quick_train import build_unet

    with torch.device("meta"):
        network = build_unet(width, output_channels=outputs)
    assert custom_unet_parameter_count(width, outputs) == sum(
        p.numel() for p in network.parameters()
    )


@pytest.mark.parametrize("width", (0, 3, 20, 128))
def test_unsupported_widths_are_rejected(width):
    with pytest.raises(ValueError, match="width"):
        ModelSpec(width=width).validate()
    with pytest.raises(ValueError, match="width"):
        QuickTrainConfig(base_channels=width).validate()


@pytest.mark.parametrize("width,explicit_spec", ((32, True), (64, False)))
def test_wide_checkpoint_roundtrip_preserves_logits(tmp_path, width, explicit_spec):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    config = QuickTrainConfig(
        crop_pixels=32, input_pixels=32, base_channels=width, device="cpu",
        model_spec=ModelSpec(width=width).to_dict() if explicit_spec else None,
    )
    config.validate()
    network = build_model(config).eval()
    image = torch.rand(1, 1, 32, 48)
    with torch.no_grad():
        expected = network(image)
    assert expected.shape == image.shape
    trained = QuickTrainModel(network, config, 32, {}, "cpu")
    path = tmp_path / "wide.pt"
    save_model(trained, path)
    restored = load_model(path)
    assert restored.config.base_channels == width
    assert spec_for(restored.config).width == width
    with torch.no_grad():
        assert torch.equal(expected, restored.network(image))


def test_width_selector_counts_outputs_and_preserves_settings(make_napari_viewer, qtbot, monkeypatch):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget
    import napari_label_assistant_tools._quick_train as engine

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("Changing model configuration must not allocate a network.")

    monkeypatch.setattr(engine, "build_unet", unexpected_allocation)
    widget = QuickTrainWidget(make_napari_viewer())
    qtbot.addWidget(widget)
    panel = widget.models_panel
    assert [panel.width.itemData(i) for i in range(panel.width.count())] == list(CUSTOM_UNET_WIDTHS)
    assert panel.width.currentData() == 8
    assert "492,465 parameters" in panel.parameter_count.text()
    for width in (32, 64):
        panel.width.setCurrentIndex(panel.width.findData(width))
        assert widget._training_config(False).base_channels == width
        assert widget._training_config(False).model_spec["width"] == width
        assert f"{custom_unet_parameter_count(width):,} parameters" in panel.parameter_count.text()
        settings = widget.project_settings()
        panel.width.setCurrentIndex(panel.width.findData(8))
        widget.restore_project_settings(settings)
        assert panel.width.currentData() == width

    panel.class_names_edit.setText("axon, myelin")
    assert f"{custom_unet_parameter_count(64, 3):,} parameters" in panel.parameter_count.text()
    panel.output_mode.setCurrentIndex(panel.output_mode.findData(True))
    assert f"{custom_unet_parameter_count(64, 2):,} parameters" in panel.parameter_count.text()
    panel.provider.setCurrentIndex(panel.provider.findData("standard"))
    assert panel.parameter_count.isHidden()
    assert [panel.width.itemData(i) for i in range(panel.width.count())] == [16, 32, 64]


def test_continue_keeps_saved_width_when_new_model_selection_changes(make_napari_viewer, qtbot):
    from napari_label_assistant_tools._quick_train_widget import QuickTrainWidget

    widget = QuickTrainWidget(make_napari_viewer())
    qtbot.addWidget(widget)
    config = replace(QuickTrainConfig(), model_spec=ModelSpec(width=8).to_dict())
    widget.model = QuickTrainModel(None, config, config.crop_pixels, {}, "cpu")
    widget.models_panel.width.setCurrentIndex(widget.models_panel.width.findData(64))
    assert spec_for(widget._training_config(False)).width == 64
    continued = widget._training_config(True)
    assert spec_for(continued).width == continued.base_channels == 8
