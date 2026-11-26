"""Model definitions and backbone builders for GeoShift."""

from geoshift.models.classifier import (
    SUPPORTED_BACKBONES,
    GeoClassifier,
    build_classifier,
)

__all__ = [
    "SUPPORTED_BACKBONES",
    "GeoClassifier",
    "build_classifier",
]