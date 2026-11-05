"""Coordinate contracts for 2D masks, crops and derived layers.

All transforms use napari's public layer properties and Affine API.
Pixel arrays are never read or resampled to determine spatial compatibility.
"""
import numpy as np
from napari.utils.transforms import Affine


def _mask_axes(layer):
    ndim = len(layer.scale)
    if ndim == 2:
        return (0, 1)
    data = layer.data
    while isinstance(data, (list, tuple)):
        data = data[0]
    axes = tuple(i for i, size in enumerate(data.shape) if size != 1)
    if len(axes) != 2:
        raise ValueError("Coordinate mapping requires a 2D mask.")
    return axes


def mask_world_affine(layer):
    """Return the mask-plane mapping, rejecting loss of a third world axis."""
    axes = _mask_axes(layer)
    # Compose analytically: subtracting data_to_world(1)-data_to_world(0)
    # loses pixel-scale precision at large world origins.
    physical = Affine(scale=layer.scale, translate=layer.translate,
                      rotate=layer.rotate, shear=layer.shear).affine_matrix
    full = np.asarray(layer.affine.affine_matrix) @ physical
    zero = full[:-1, -1]
    linear = full[:-1, list(axes)]
    # napari aligns a derived 2D layer to the viewer's final two world axes,
    # even when the source mask was squeezed along a different singleton axis.
    world_axes = (len(zero)-2, len(zero)-1)
    omitted = list(range(len(zero)-2))
    if omitted and (np.any(np.abs(zero[omitted]) > 1e-12)
                    or np.any(np.abs(linear[omitted]) > 1e-12)):
        raise ValueError("This mask's transform leaves the 2D plane. Use aligned 2D layers before copying.")
    matrix = np.eye(3)
    matrix[:2, :2] = linear[list(world_axes)]
    matrix[:2, 2] = zero[list(world_axes)]
    if not np.isfinite(matrix).all():
        raise ValueError("Layer coordinates must be finite.")
    return matrix


def matching_mask_transforms(source, target):
    """Compare registration to within 1e-6 source pixels over the whole mask.

    Testing in pixel units avoids accepting a whole-pixel offset merely because
    world coordinates have a large origin or very small physical pixel sizes.
    """
    try:
        a, b = mask_world_affine(source), mask_world_affine(target)
        relative = np.linalg.solve(a[:2, :2], b[:2, :2]) - np.eye(2)
        offset = np.linalg.solve(a[:2, :2], b[:2, 2] - a[:2, 2])
        data = source.data
        while isinstance(data, (list, tuple)):
            data = data[0]
        h, w = (data.shape[i] for i in _mask_axes(source))
        corners = np.asarray([[0, 0], [h-1, 0], [0, w-1], [h-1, w-1]])
        errors = corners @ relative.T + offset
        return bool(np.isfinite(errors).all() and np.max(np.abs(errors)) <= 1e-6)
    except (ValueError, np.linalg.LinAlgError):
        return False


def crop_transform_kwargs(layer, origin=(0, 0)):
    """Map local crop pixel p to source pixel p + origin, exactly in world.

    For 2D layers retain scale/translate/rotate/shear and the extra affine.
    A crop adds the full world-space linear displacement to the extra affine's
    translation; it must not discard rotation or apply the origin twice.
    """
    matrix = mask_world_affine(layer)
    if len(layer.scale) == 2:
        affine = np.asarray(layer.affine.affine_matrix, dtype=float).copy()
        affine[:2, 2] += matrix[:2, :2] @ np.asarray(origin, dtype=float)
        return dict(scale=np.array(layer.scale, copy=True),
                    translate=np.array(layer.translate, copy=True),
                    rotate=np.array(layer.rotate, copy=True),
                    shear=np.array(layer.shear, copy=True), affine=affine)
    # Squeezed singleton axes can be projected only when they remain in the
    # zero 2D plane, as checked above. Preserve the retained pixel scales.
    scale = np.asarray(layer.scale)[list(_mask_axes(layer))]
    affine = np.eye(3)
    affine[:2, :2] = matrix[:2, :2] / scale
    affine[:2, 2] = matrix[:2, 2] + matrix[:2, :2] @ origin
    return dict(scale=scale, translate=(0, 0), affine=affine)


def apply_crop_transform(target, source, origin=(0, 0)):
    for name, value in crop_transform_kwargs(source, origin).items():
        setattr(target, name, value)
