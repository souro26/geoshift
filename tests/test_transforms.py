
"""Tests for GeoShift image preprocessing and augmentation."""

import numpy as np
import torch
from PIL import Image

from geoshift.data.transforms import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    build_train_transform,
)


def create_test_image() -> Image.Image:
    """Create a synthetic RGB image."""

    array = np.zeros((300, 400, 3), dtype=np.uint8)
    array[:, :, 0] = 120
    array[:, :, 1] = 160
    array[:, :, 2] = 200

    return Image.fromarray(array, mode="RGB")


def test_training_transform_returns_tensor():
    transform = build_train_transform()

    result = transform(create_test_image())

    assert isinstance(result, torch.Tensor)


def test_evaluation_transform_returns_tensor():
    transform = build_eval_transform()

    result = transform(create_test_image())

    assert isinstance(result, torch.Tensor)


def test_training_transform_has_correct_shape():
    result = build_train_transform()(create_test_image())

    assert result.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_evaluation_transform_has_correct_shape():
    result = build_eval_transform()(create_test_image())

    assert result.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_transforms_return_float32_tensors():
    train_result = build_train_transform()(create_test_image())
    eval_result = build_eval_transform()(create_test_image())

    assert train_result.dtype == torch.float32
    assert eval_result.dtype == torch.float32


def test_evaluation_transform_is_deterministic():
    transform = build_eval_transform()
    image = create_test_image()

    first = transform(image)
    second = transform(image)

    assert torch.equal(first, second)


def test_evaluation_transform_applies_imagenet_normalization():
    image = Image.new("RGB", (300, 400), color=(0, 0, 0))

    result = build_eval_transform()(image)

    expected = torch.tensor(
        [
            -mean / std
            for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)
        ],
        dtype=torch.float32,
    )

    actual = result[:, 0, 0]

    torch.testing.assert_close(actual, expected)


def test_transformed_tensors_contain_finite_values():
    train_result = build_train_transform()(create_test_image())
    eval_result = build_eval_transform()(create_test_image())

    assert torch.isfinite(train_result).all()
    assert torch.isfinite(eval_result).all()


def test_training_transform_preserves_three_channels():
    result = build_train_transform()(create_test_image())

    assert result.shape[0] == 3


def test_evaluation_transform_preserves_three_channels():
    result = build_eval_transform()(create_test_image())

    assert result.shape[0] == 3
