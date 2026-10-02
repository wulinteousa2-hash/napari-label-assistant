import json
from threading import Event

import numpy as np
import pytest

from napari_label_assistant_tools._quick_train import (
    QuickTrainCancelled, QuickTrainConfig, build_crop_atlas, crop_summary,
    load_model, predict_area, prepare_crops, save_model, train_quick_unet,
)


class SliceOnlyArray:
    """A lazy-source stand-in that forbids accidental full-array conversion."""
    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.reads = []

    def __array__(self, *args, **kwargs):
        raise AssertionError("Do not materialize the full source.")

    def __getitem__(self, region):
        self.reads.append(region)
        assert all(isinstance(axis, slice) for axis in region)
        return self.array[region]


def consume(generator):
    updates = []
    while True:
        try:
            updates.append(next(generator))
        except StopIteration as result:
            return result.value, updates


def paired_image():
    yy, xx = np.indices((256, 256))
    truth = ((xx // 16) % 2).astype(np.uint8)
    image = (truth * 120 + 50 + yy % 8).astype(np.uint8)
    return image, truth


def tiny_config(**kwargs):
    return QuickTrainConfig(
        crop_pixels=64, input_pixels=32, samples=8, steps=2,
        base_channels=4, device="cpu", **kwargs,
    )


def test_paired_crops_are_bounded_and_spatially_separate():
    image, truth = paired_image()
    image, truth = SliceOnlyArray(image), SliceOnlyArray(truth)
    crops, updates = consume(prepare_crops(image, truth, tiny_config()))
    assert crops.images.shape == (6, 32, 32)
    assert crops.validation_images.shape == (2, 32, 32)
    assert updates[-1].completed == 8
    assert any("Scanning ground truth" in update.message for update in updates)
    assert len(image.reads) == 8
    assert crops.masks.any() and not crops.masks.all()
    training = [item for item in crops.locations if item["split"] == "training"]
    validation = [item for item in crops.locations if item["split"] == "validation"]
    for a in training:
        for b in validation:
            assert (a["y0"] + a["size"] <= b["y0"] or b["y0"] + b["size"] <= a["y0"]
                    or a["x0"] + a["size"] <= b["x0"] or b["x0"] + b["size"] <= a["x0"])
    assert all(axis.stop - axis.start <= 64 for read in image.reads for axis in read)
    assert all(axis.stop - axis.start <= 2048 for read in truth.reads for axis in read)


def test_empty_truth_and_shape_mismatch_are_actionable():
    image, truth = paired_image()
    with pytest.raises(ValueError, match="No labeled targets"):
        consume(prepare_crops(image, np.zeros_like(truth), tiny_config()))
    with pytest.raises(ValueError, match="same 2D shape"):
        consume(prepare_crops(image, truth[:100], tiny_config()))


def test_cancellation_during_crop_preparation():
    image, truth = paired_image()
    cancel = Event()
    generator = prepare_crops(image, truth, tiny_config(), cancel=cancel)
    next(generator)
    cancel.set()
    with pytest.raises(QuickTrainCancelled):
        next(generator)


def test_nonzero_labels_and_multiscale_sources_are_supported():
    image, truth = paired_image()
    truth = -truth.astype(np.int32)
    crops, _ = consume(prepare_crops([image, image[::2, ::2]], [truth, truth[::2, ::2]], tiny_config()))
    assert crops.masks.any()


def test_napari_multiscale_wrapper_reads_highest_resolution(make_napari_viewer):
    from napari_label_assistant_tools._quick_train import array_2d

    viewer = make_napari_viewer()
    image, truth = paired_image()
    image_layer = viewer.add_image([image, image[::2, ::2]], multiscale=True)
    truth_layer = viewer.add_labels([truth, truth[::2, ::2]], multiscale=True)
    assert array_2d(image_layer.data) is image
    assert array_2d(truth_layer.data) is truth
    crops, _ = consume(prepare_crops(image_layer.data, truth_layer.data, tiny_config()))
    assert crops.crop_pixels == 64
    assert crops.images.shape == (6, 32, 32)
    assert crops.masks.any()


def test_missing_optional_dependency_has_install_instructions(monkeypatch):
    import builtins
    from napari_label_assistant_tools._quick_train import _torch

    original_import = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("PyTorch is absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    with pytest.raises(RuntimeError, match=r"napari-label-assistant\[train\]"):
        _torch()


def test_actual_cpu_training_prediction_and_checkpoint_round_trip(tmp_path):
    pytest.importorskip("torch")
    image, truth = paired_image()
    model, updates = consume(train_quick_unet(image, truth, tiny_config()))
    assert model.training_device == "cpu"
    assert updates[-1].completed == 2
    assert all(0 <= value <= 1 for value in model.metrics.values())
    lazy_image = SliceOnlyArray(image)
    prediction = predict_area(model, lazy_image, (128, 128), device="cpu")
    assert prediction.probability.shape == (64, 64)
    assert prediction.origin == (96, 96)
    assert np.isfinite(prediction.probability).all()
    assert len(lazy_image.reads) == 1
    path = tmp_path / "model.pt"
    save_model(model, path)
    loaded = load_model(path)
    restored = predict_area(loaded, image, (128, 128), device="cpu")
    np.testing.assert_allclose(restored.probability, prediction.probability)
    assert loaded.metrics == model.metrics
    assert loaded.config == model.config
    assert loaded.history == model.history
    exported = json.loads(path.with_suffix(".history.json").read_text())
    assert exported["history"] == loaded.history
    record = loaded.history[0]
    assert record["mode"] == "new" and record["round"] == 1
    assert record["config"]["steps"] == 2
    assert record["duration_seconds"] >= 0
    assert [row["step"] for row in record["checkpoints"]] == [1, 2]
    assert record["retained_validation"] == loaded.metrics
    # The model is self-contained even without the readable export.
    payload = pytest.importorskip("torch").load(path, weights_only=True)
    payload.pop("history")
    pytest.importorskip("torch").save(payload, path)
    assert load_model(path).history == []


def test_cancellation_after_training_started():
    pytest.importorskip("torch")
    image, truth = paired_image()
    cancel = Event()
    generator = train_quick_unet(image, truth, tiny_config(), cancel=cancel)
    for update in generator:
        if "Training small U-Net" in update.message:
            break
    cancel.set()
    with pytest.raises(QuickTrainCancelled):
        next(generator)


def test_explicit_cuda_failure_is_actionable(monkeypatch):
    from napari_label_assistant_tools._quick_train import _device

    class NoCuda:
        @staticmethod
        def is_available():
            return False

    class Torch:
        cuda = NoCuda()

    assert _device(Torch(), "auto") == "cpu"
    with pytest.raises(RuntimeError, match="Choose CPU"):
        _device(Torch(), "cuda")


def test_sparse_single_pixel_targets_survive_and_get_positive_patches():
    rng = np.random.default_rng(4)
    image = rng.integers(30, 200, (1024, 1024), dtype=np.uint8)
    truth = np.zeros_like(image)
    for y, x in [(100, 100), (150, 750), (650, 150), (800, 800)]:
        truth[y, x] = 1
    config = QuickTrainConfig(crop_pixels=128, input_pixels=32, samples=16, steps=2, device="cpu")
    crops, _ = consume(prepare_crops(SliceOnlyArray(image), SliceOnlyArray(truth), config))
    positive = [p for p in crops.locations if p["requested"] == "positive"]
    assert len(positive) == 8
    assert all(p["foreground_pixels_original"] >= 1 and p["foreground_pixels_model"] >= 1 for p in positive)
    assert {p["split"] for p in positive} == {"training", "validation"}
    assert any(p["kind"] == "nearby background" for p in crops.locations)
    assert all(p["foreground_pixels_original"] == 0 for p in crops.locations if "background" in p["kind"])
    counts = crop_summary(crops)
    assert sum(c["positive"] for c in counts.values()) >= 8
    assert sum(c["background"] for c in counts.values()) > 0


def test_atlas_contains_exact_model_inputs_and_matching_masks():
    image, truth = paired_image()
    crops, _ = consume(prepare_crops(image, truth, tiny_config()))
    atlas = build_crop_atlas(crops)
    images = np.concatenate((crops.images, crops.validation_images))
    masks = np.concatenate((crops.masks, crops.validation_masks))
    for i, tile in enumerate(atlas.locations):
        y, x, size = tile["atlas_y0"], tile["atlas_x0"], tile["model_size"]
        np.testing.assert_array_equal(atlas.image[y:y + size, x:x + size], images[i])
        np.testing.assert_array_equal(atlas.mask[y:y + size, x:x + size], masks[i])
        assert tile["tile"] == i + 1
        assert tile["y0"] == crops.locations[i]["y0"]


def test_one_annotated_spatial_block_requires_more_representative_data():
    image = np.zeros((512, 512), dtype=np.uint8)
    truth = np.zeros_like(image)
    truth[10, 10] = 1
    with pytest.raises(ValueError, match="one spatial block"):
        consume(prepare_crops(image, truth, tiny_config()))


def test_continue_loaded_weights_and_cancel_preserve_original(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import napari_label_assistant_tools._quick_train as backend
    image, truth = paired_image()
    config = tiny_config()
    model, _ = consume(train_quick_unet(image, truth, config))
    path = tmp_path / "continue.pt"
    save_model(model, path)
    original = load_model(path)
    weights = {k: v.clone() for k, v in original.network.state_dict().items()}
    crops, _ = consume(prepare_crops(image, truth, config))
    real_scores = backend._scores
    calls = []

    def scores(torch_module, network, *args):
        calls.append({k: v.detach().cpu().clone() for k, v in network.state_dict().items()})
        metrics = real_scores(torch_module, network, *args)
        metrics["dice"] = 0.0  # Ties must still retain updated weights.
        return metrics

    monkeypatch.setattr(backend, "_scores", scores)
    continued, _ = consume(backend.train_prepared_unet(crops, config, initial_model=original, learning_rate=0.0001,
                                                                source_info={"image_layer": "EM", "ground_truth_layer": "Curated"}))
    for key in weights:
        assert torch.equal(calls[0][key], weights[key])
        assert torch.equal(original.network.state_dict()[key], weights[key])
    assert any(not torch.equal(calls[1][key], weights[key]) for key in weights)
    assert any(not torch.equal(continued.network.state_dict()[key], weights[key]) for key in weights)
    assert continued.network is not original.network
    assert continued.training_steps == 4 and continued.training_rounds == 2
    save_model(continued, path)
    restored = load_model(path)
    assert restored.training_steps == 4 and restored.training_rounds == 2
    assert len(restored.history) == 2 and len(original.history) == 1
    assert restored.history[0] == original.history[0]
    assert restored.history[1]["source"] == {"image_layer": "EM", "ground_truth_layer": "Curated"}
    assert restored.history[1]["learning_rate"] == 0.0001
    assert restored.history[1]["checkpoints"][0]["step"] == 0
    assert json.loads(path.with_suffix(".history.json").read_text())["history"] == restored.history
    cancel = Event()
    generator = backend.train_prepared_unet(crops, config, initial_model=original, cancel=cancel)
    next(generator)
    next(generator)
    cancel.set()
    with pytest.raises(QuickTrainCancelled):
        next(generator)
    for key in weights:
        assert torch.equal(original.network.state_dict()[key], weights[key])
    assert len(original.history) == 1


def test_sampling_seed_keeps_holdout_blocks_fixed():
    image, truth = paired_image()
    first, _ = consume(prepare_crops(image, truth, tiny_config(seed=42)))
    second, _ = consume(prepare_crops(image, truth, tiny_config(seed=43)))
    assert [(b["id"], b["split"]) for b in first.blocks] == [(b["id"], b["split"]) for b in second.blocks]
    assert first.locations != second.locations


def test_continuation_retains_starting_weights_if_validation_declines(monkeypatch):
    torch = pytest.importorskip("torch")
    import napari_label_assistant_tools._quick_train as backend
    image, truth = paired_image()
    config = tiny_config()
    original, _ = consume(train_quick_unet(image, truth, config))
    crops, _ = consume(prepare_crops(image, truth, config))
    scores = iter([1.0, 0.0, 0.0])
    monkeypatch.setattr(backend, "_scores", lambda *args: {"dice": next(scores), "iou": 0.0, "precision": 0.0, "recall": 0.0})
    continued, _ = consume(backend.train_prepared_unet(crops, config, initial_model=original))
    for key, value in original.network.state_dict().items():
        assert torch.equal(value, continued.network.state_dict()[key])
