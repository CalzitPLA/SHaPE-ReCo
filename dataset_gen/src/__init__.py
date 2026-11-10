"""
FCMacros Dataset Generation Package

This package provides tools for generating FreeCAD STL datasets for CNN learning.
"""

from .dataset_utils import (
    TOPOLOGIES,
    MACRO_MAP,
    PARAMETER_SPECS,
    Rotation,
    Sample,
    sample_parameters,
    sample_rotation,
    build_filename,
    validate_affine_specs,
)

__version__ = "2.0.0"
__all__ = [
    "TOPOLOGIES",
    "MACRO_MAP",
    "PARAMETER_SPECS",
    "Rotation",
    "Sample",
    "sample_parameters",
    "sample_rotation",
    "build_filename",
    "validate_affine_specs",
]
