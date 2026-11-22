
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGE_SIZE = 224
RESIZE_SIZE = 256

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform() -> transforms.Compose:
    """Build image transformations for model training.

    Uses moderate augmentation to avoid excessively altering
    potentially useful geographic context.
    """

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=IMAGE_SIZE,
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
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
