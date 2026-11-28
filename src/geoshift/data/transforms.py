
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from geoshift.config import load_config

_IMAGE_CONFIG = load_config("data.yml")["image"]
IMAGE_SIZE = _IMAGE_CONFIG["size"]
RESIZE_SIZE = _IMAGE_CONFIG["resize_size"]
IMAGENET_MEAN = tuple(_IMAGE_CONFIG["imagenet_mean"])
IMAGENET_STD = tuple(_IMAGE_CONFIG["imagenet_std"])
_TRAIN_CONFIG = _IMAGE_CONFIG["train"]


def build_train_transform() -> transforms.Compose:
    """Build image transformations for model training.

    Uses moderate augmentation to avoid excessively altering
    potentially useful geographic context.
    """

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=IMAGE_SIZE,
                scale=tuple(_TRAIN_CONFIG["random_resized_crop"]["scale"]),
                ratio=tuple(_TRAIN_CONFIG["random_resized_crop"]["ratio"]),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(
                **_TRAIN_CONFIG["color_jitter"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    """Build deterministic image transformations for evaluation.

    Evaluation preprocessing must remain identical across
    validation and test datasets.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                RESIZE_SIZE,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )
