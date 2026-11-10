#!/usr/bin/env python3
"""
Shared utilities for FreeCAD STL dataset generation.

This module contains common functions used by both CLI and GUI frontends.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_SCALE_MM = 10.0
DEFAULT_SEED = 12345
DEFAULT_ROTATION_MIN = -180.0
DEFAULT_ROTATION_MAX = 180.0


# ============================================================================
# TOPOLOGIES
# ============================================================================

TOPOLOGIES = [
    "box",
    "cone",
    "cone_segment",
    "cube",
    "cylinder",
    "ellipsoid",
    "elliptical_cone",
    "elliptical_tube",
    "general_trapezoid",
    "paraboloid",
    "parallelepiped",
    "polycone",
    "polyhedron",
    "sphere",
    "torus",
    "torus_segment",
    "tube",
]


# ============================================================================
# MACRO MAPPING
# ============================================================================

MACRO_MAP = {
    "box": "box_gdml_dataset_learning_CNN.FCMacro",
    "cone": "cone_dataset_learning_CNN.FCMacro",
    "cone_segment": "cone_segment_gdml_dataset_learning_CNN.FCMacro",
    "cube": "cube_datset_learning_CNN.FCMacro",
    "cylinder": "zylinder_gdml_dataset_learning_CNN.FCMacro",
    "ellipsoid": "ellipsoid_gdml_dataset_learning_CNN.FCMacro",
    "elliptical_cone": "elliptical_cone_gdml_dataset_learning_CNN.FCMacro",
    "elliptical_tube": "elliptical_tube_gdml_dataset_learning_CNN.FCMacro",
    "general_trapezoid": "general_trapezoid_gdml_dataset_learning_CNN.FCMacro",
    "paraboloid": "paraboloid_gdml_dataset_learning_CNN.FCMacro",
    "parallelepiped": "parallelepiped_gdml_dataset_learning_CNN.FCMacro",
    "polycone": "polycone_gdml_dataset_learning_CNN.FCMacro",
    "polyhedron": "polyhedron_gdml_dataset_learning_CNN.FCMacro",
    "sphere": "sphere_dataset_learning_CNN.FCMacro",
    "torus": "torus_dataset_learning_CNN.FCMacro",
    "torus_segment": "torus_segment_gdml_dataset_learning_CNN.FCMacro",
    "tube": "tube_dataset_learning_CNN.FCMacro",
}


# ============================================================================
# PARAMETER SPECIFICATIONS
# ============================================================================

PARAMETER_SPECS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "box": {
        "x_half_ratio": {"min": 0.25, "max": 4.00},
        "y_half_ratio": {"min": 0.25, "max": 4.00},
        "z_half_ratio": {"min": 0.25, "max": 4.00},
    },
    "cone": {
        "radius1_ratio": {"min": 0.25, "max": 2.00},
        "radius2_ratio": {"min": 0.25, "max": 2.00},
        "height_ratio": {"min": 0.25, "max": 4.00},
    },
    "cone_segment": {
        "rmin1_ratio": {"min": 0.00, "max": 0.60},
        "rmax1_ratio": {"min": 0.50, "max": 1.50},
        "rmin2_ratio": {"min": 0.00, "max": 0.60},
        "rmax2_ratio": {"min": 0.50, "max": 1.50},
        "dz_ratio": {"min": 0.50, "max": 2.00},
        "startphi_deg": {"min": 0.00, "max": 360.00},
        "deltaphi_deg": {"min": 15.00, "max": 330.00},
    },
    "cube": {
        "half_size_ratio": {"min": 0.25, "max": 4.00},
    },
    "cylinder": {
        "radius_ratio": {"min": 0.25, "max": 2.00},
        "half_length_ratio": {"min": 0.25, "max": 4.00},
    },
    "ellipsoid": {
        "x_radius_ratio": {"min": 0.25, "max": 4.00},
        "y_radius_ratio": {"min": 0.25, "max": 4.00},
        "z_radius_ratio": {"min": 0.25, "max": 4.00},
    },
    "elliptical_cone": {
        "x_radius1_ratio": {"min": 0.25, "max": 2.00},
        "y_radius1_ratio": {"min": 0.25, "max": 2.00},
        "x_radius2_ratio": {"min": 0.25, "max": 2.00},
        "y_radius2_ratio": {"min": 0.25, "max": 2.00},
        "height_ratio": {"min": 0.25, "max": 4.00},
    },
    "elliptical_tube": {
        "outer_x_radius_ratio": {"min": 0.75, "max": 2.00},
        "outer_y_radius_ratio": {"min": 0.75, "max": 2.00},
        "inner_x_radius_ratio": {"min": 0.10, "max": 0.65},
        "inner_y_radius_ratio": {"min": 0.10, "max": 0.65},
        "half_length_ratio": {"min": 0.25, "max": 4.00},
    },
    "general_trapezoid": {
        "dz_ratio": {"min": 0.25, "max": 4.00},
        "theta_deg": {"min": -35.0, "max": 35.0},
        "phi_deg": {"min": 0.0, "max": 360.0},
        "dy1_ratio": {"min": 0.25, "max": 3.00},
        "dx1_ratio": {"min": 0.25, "max": 3.00},
        "alpha1_deg": {"min": -30.0, "max": 30.0},
        "dy2_ratio": {"min": 0.25, "max": 3.00},
        "dx2_ratio": {"min": 0.25, "max": 3.00},
        "alpha2_deg": {"min": -30.0, "max": 30.0},
    },
    "paraboloid": {
        "radius_ratio": {"min": 0.25, "max": 3.00},
        "height_ratio": {"min": 0.25, "max": 4.00},
    },
    "parallelepiped": {
        "dx_ratio": {"min": 0.25, "max": 3.00},
        "dy_ratio": {"min": 0.25, "max": 3.00},
        "dz_ratio": {"min": 0.25, "max": 3.00},
        "alpha_deg": {"min": -30.0, "max": 30.0},
        "theta_deg": {"min": 0.0, "max": 35.0},
        "phi_deg": {"min": 0.0, "max": 360.0},
    },
    "polycone": {
        "rmin1_ratio": {"min": 0.00, "max": 0.50},
        "rmax1_ratio": {"min": 0.50, "max": 1.50},
        "rmin2_ratio": {"min": 0.00, "max": 0.50},
        "rmax2_ratio": {"min": 0.50, "max": 1.50},
        "rmin3_ratio": {"min": 0.00, "max": 0.50},
        "rmax3_ratio": {"min": 0.50, "max": 1.50},
        "z2_ratio": {"min": 0.25, "max": 1.00},
        "z3_ratio": {"min": 1.00, "max": 3.00},
        "startphi_deg": {"min": 0.00, "max": 360.00},
        "deltaphi_deg": {"min": 15.00, "max": 330.00},
    },
    "polyhedron": {
        "num_sides": {"choices": [3, 4, 5, 6, 8, 10, 12, 16]},
        "radius_ratio": {"min": 0.25, "max": 3.00},
        "half_length_ratio": {"min": 0.25, "max": 4.00},
        "startphi_deg": {"min": 0.00, "max": 360.00},
        "deltaphi_deg": {"min": 15.00, "max": 360.00},
    },
    "sphere": {
        "radius_ratio": {"min": 0.25, "max": 4.00},
    },
    "torus": {
        "major_radius_ratio": {"min": 0.75, "max": 4.00},
        "minor_radius_ratio": {"min": 0.10, "max": 0.60},
        "angle1_deg": {"min": -180.0, "max": 0.0},
        "angle2_deg": {"min": 0.0, "max": 180.0},
        "angle3_deg": {"min": 15.0, "max": 360.0},
    },
    "torus_segment": {
        "major_radius_ratio": {"min": 0.75, "max": 4.00},
        "minor_radius_ratio": {"min": 0.10, "max": 0.60},
        "angle1_deg": {"min": -180.0, "max": 0.0},
        "angle2_deg": {"min": 0.0, "max": 180.0},
        "angle3_deg": {"min": 15.0, "max": 330.0},
        "startphi_deg": {"min": 0.0, "max": 360.0},
        "deltaphi_deg": {"min": 15.0, "max": 330.0},
    },
    "tube": {
        "inner_radius_ratio": {"min": 0.10, "max": 0.65},
        "outer_radius_ratio": {"min": 0.75, "max": 2.00},
        "half_length_ratio": {"min": 0.25, "max": 4.00},
        "startphi_deg": {"min": 0.0, "max": 360.0},
        "deltaphi_deg": {"min": 15.0, "max": 330.0},
    },
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Rotation:
    rx: float
    ry: float
    rz: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass
class Sample:
    index: int
    mode: str
    topology: str
    parameters: Dict[str, Any]
    rotation: Rotation
    scale_mm: float
    stl_path: str
    seed: int
    voxel_path: Optional[str] = None
    placement: Optional[Dict[str, Any]] = None      # size / position drawn for an in-scale sample (src/inscale.py)


# ============================================================================
# VALIDATION
# ============================================================================

def is_finite_number(value: Any) -> bool:
    """Check if value is a finite number."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def validate_parameter_spec(parameter_name: str, spec: Any) -> None:
    """Validate a parameter specification."""
    if isinstance(spec, dict):
        has_min = "min" in spec
        has_max = "max" in spec
        has_choices = "choices" in spec
        
        if has_min or has_max:
            if not (has_min and has_max):
                raise ValueError(
                    f"Parameter '{parameter_name}' must contain both 'min' and 'max'."
                )
            if not is_finite_number(spec["min"]):
                raise ValueError(f"Parameter '{parameter_name}': min must be finite.")
            if not is_finite_number(spec["max"]):
                raise ValueError(f"Parameter '{parameter_name}': max must be finite.")
            if float(spec["min"]) > float(spec["max"]):
                raise ValueError(f"Parameter '{parameter_name}': min > max.")
            if has_choices:
                raise ValueError(
                    f"Parameter '{parameter_name}' cannot contain both range and choices."
                )
            return
        
        if has_choices:
            choices = spec["choices"]
            if not isinstance(choices, list) or not choices:
                raise ValueError(
                    f"Parameter '{parameter_name}': choices must be a non-empty list."
                )
            for value in choices:
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f"Parameter '{parameter_name}': non-finite choice."
                    )
            return
        
        raise ValueError(f"Unsupported parameter specification: {spec}")
    
    if isinstance(spec, list):
        if not spec:
            raise ValueError(f"Parameter '{parameter_name}' has an empty choice list.")
        return
    
    if is_finite_number(spec):
        return
    
    if isinstance(spec, str):
        return
    
    raise ValueError(
        f"Unsupported parameter specification for '{parameter_name}': {type(spec).__name__}"
    )


def validate_affine_specs(affine_specs: Dict[str, Any]) -> None:
    """Validate all affine specifications."""
    if not isinstance(affine_specs, dict):
        raise ValueError("Affine specification must be a JSON object.")
    
    for topology, topology_spec in affine_specs.items():
        if not isinstance(topology, str) or not topology.strip():
            raise ValueError("Topology name must be a non-empty string.")
        if not isinstance(topology_spec, dict):
            raise ValueError(f"Specification for '{topology}' must be an object.")
        for parameter_name, parameter_spec in topology_spec.items():
            validate_parameter_spec(parameter_name, parameter_spec)


# ============================================================================
# ROTATION
# ============================================================================

def euler_xyz_to_quaternion(
    rx_deg: float, ry_deg: float, rz_deg: float
) -> Tuple[float, float, float, float]:
    """
    Convert XYZ Euler angles to quaternion [qx, qy, qz, qw].
    
    Uses XYZ intrinsic rotation and normalizes the result.
    Ensures qw >= 0 for canonical representation.
    """
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    
    cx = math.cos(rx * 0.5)
    sx = math.sin(rx * 0.5)
    cy = math.cos(ry * 0.5)
    sy = math.sin(ry * 0.5)
    cz = math.cos(rz * 0.5)
    sz = math.sin(rz * 0.5)
    
    qw = cx * cy * cz - sx * sy * sz
    qx = sx * cy * cz + cx * sy * sz
    qy = cx * sy * cz - sx * cy * sz
    qz = cx * cy * sz + sx * sy * cz
    
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-15:
        raise RuntimeError("Generated zero-length quaternion.")
    
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm
    
    if qw < 0.0:
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    
    return (qx, qy, qz, qw)


def random_unit_quaternion(rng: random.Random) -> List[float]:
    """
    Generate a uniform random rotation on SO(3).
    
    Uses the standard uniform unit-quaternion construction.
    """
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()
    
    sqrt1 = math.sqrt(1.0 - u1)
    sqrt2 = math.sqrt(u1)
    
    theta1 = 2.0 * math.pi * u2
    theta2 = 2.0 * math.pi * u3
    
    qx = sqrt1 * math.sin(theta1)
    qy = sqrt1 * math.cos(theta1)
    qz = sqrt2 * math.sin(theta2)
    qw = sqrt2 * math.cos(theta2)
    
    return [qx, qy, qz, qw]


def quaternion_to_euler_xyz(q: List[float]) -> Tuple[float, float, float]:
    """
    Convert quaternion [x,y,z,w] to XYZ Euler angles.
    
    Used for human-readable filenames. The quaternion remains authoritative.
    """
    x, y, z, w = q
    
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def sample_rotation(
    rng: random.Random, minimum: float, maximum: float
) -> Rotation:
    """Generate a random rotation with both Euler and quaternion representations."""
    rx = rng.uniform(minimum, maximum)
    ry = rng.uniform(minimum, maximum)
    rz = rng.uniform(minimum, maximum)
    
    qx, qy, qz, qw = euler_xyz_to_quaternion(rx, ry, rz)
    
    return Rotation(rx=rx, ry=ry, rz=rz, qx=qx, qy=qy, qz=qz, qw=qw)


# ============================================================================
# PARAMETER SAMPLING
# ============================================================================

def sample_parameter(spec: Dict[str, Any], rng: random.Random) -> Any:
    """Sample a single parameter from its specification."""
    if "choices" in spec:
        return rng.choice(spec["choices"])
    return rng.uniform(float(spec["min"]), float(spec["max"]))


def sample_parameters(topology: str, rng: random.Random) -> Dict[str, Any]:
    """Generate random parameters for a topology."""
    if topology not in PARAMETER_SPECS:
        raise RuntimeError(f"No parameter specification for topology '{topology}'.")
    
    specs = PARAMETER_SPECS[topology]
    parameters = {
        name: sample_parameter(spec, rng) for name, spec in specs.items()
    }
    
    return repair_parameters(topology, parameters)


def repair_parameters(topology: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Repair dependent parameters after random sampling.
    
    Ensures basic geometric validity while keeping values in intended range.
    """
    p = dict(parameters)
    
    if topology == "cone_segment":
        p["rmax1_ratio"] = max(p["rmax1_ratio"], p["rmin1_ratio"] + 0.01)
        p["rmax2_ratio"] = max(p["rmax2_ratio"], p["rmin2_ratio"] + 0.01)
        p["deltaphi_deg"] = min(max(p["deltaphi_deg"], 1.0), 359.0)
    
    elif topology == "tube":
        p["outer_radius_ratio"] = max(p["outer_radius_ratio"], p["inner_radius_ratio"] + 0.01)
        p["deltaphi_deg"] = min(max(p["deltaphi_deg"], 1.0), 359.0)
    
    elif topology == "elliptical_tube":
        p["outer_x_radius_ratio"] = max(p["outer_x_radius_ratio"], p["inner_x_radius_ratio"] + 0.01)
        p["outer_y_radius_ratio"] = max(p["outer_y_radius_ratio"], p["inner_y_radius_ratio"] + 0.01)
    
    elif topology in ("torus", "torus_segment"):
        p["major_radius_ratio"] = max(p["major_radius_ratio"], p["minor_radius_ratio"] + 0.01)
    
    if topology == "torus_segment":
        p["deltaphi_deg"] = min(max(p["deltaphi_deg"], 1.0), 359.0)
    
    elif topology == "polycone":
        p["rmax1_ratio"] = max(p["rmax1_ratio"], p["rmin1_ratio"] + 0.01)
        p["rmax2_ratio"] = max(p["rmax2_ratio"], p["rmin2_ratio"] + 0.01)
        p["rmax3_ratio"] = max(p["rmax3_ratio"], p["rmin3_ratio"] + 0.01)
        p["z3_ratio"] = max(p["z3_ratio"], p["z2_ratio"] + 0.01)
        p["deltaphi_deg"] = min(max(p["deltaphi_deg"], 1.0), 359.0)
    
    return p


# ============================================================================
# FILENAME UTILITIES
# ============================================================================

def clean_number(value: float) -> str:
    """Convert a number to a filename-safe compact representation."""
    if abs(value) < 1e-12:
        value = 0.0
    
    if value.is_integer():
        return str(int(value))
    
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def integer_angle(value: float) -> str:
    """Filename representation for Euler angles."""
    value = int(round(float(value)))
    while value > 180:
        value -= 360
    while value <= -180:
        value += 360
    return str(value)


def safe_name(text: str) -> str:
    """Make a string safe for use as a directory/file component."""
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return text


def hash_json(data: Any) -> str:
    """Create a short hash of JSON data for unique identifiers."""
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def build_filename(
    topology: str,
    index: int,
    scale_mm: float,
    rotation: Rotation,
    index_width: int = 6,
) -> str:
    """Build a descriptive STL filename."""
    return (
        f"{topology}_i{index:0{index_width}d}_s{clean_number(scale_mm)}_"
        f"cx0_cy0_cz0_rx{integer_angle(rotation.rx)}_"
        f"ry{integer_angle(rotation.ry)}_rz{integer_angle(rotation.rz)}.stl"
    )


# ============================================================================
# JSON HELPERS
# ============================================================================
# SUBPROCESS ENVIRONMENT
# ============================================================================

def clean_subprocess_env() -> Dict[str, str]:
    """os.environ, with interpreter-specific variables stripped.

    Needed whenever spawning an external executable (freecadcmd,
    stltovoxel, a plain python3) from a process that might itself be
    running inside a FreeCAD AppImage: AppImages set PYTHONHOME/
    PYTHONPATH/LD_LIBRARY_PATH to point at their own bundled runtime,
    which a *different* interpreter/executable then inherits and tries
    (and fails) to bootstrap against - e.g. a system Python 3.13
    subprocess crashing with "Fatal Python error: Failed to import
    encodings module" because PYTHONHOME pointed at a bundled 3.11.
    Harmless to strip when not running inside an AppImage.
    """
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH"):
        env.pop(key, None)
    return env


# ============================================================================

def json_dump_compact(value: Any) -> str:
    """Dump JSON with minimal whitespace."""
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


# ============================================================================
# ADDITIONAL SAMPLING UTILITIES (for Qt frontend compatibility)
# ============================================================================

def sample_from_spec(spec: Any, rng: random.Random) -> Any:
    """Sample a value from a parameter specification."""
    if isinstance(spec, dict):
        if "min" in spec and "max" in spec:
            value = rng.uniform(float(spec["min"]), float(spec["max"]))
            if not math.isfinite(value):
                raise ValueError("Generated non-finite value.")
            return value
        if "choices" in spec:
            return rng.choice(spec["choices"])
        raise ValueError(f"Unsupported specification: {spec}")
    
    if isinstance(spec, list):
        return rng.choice(spec)
    
    if is_finite_number(spec):
        return float(spec)
    
    return spec


def sample_affine_parameters(affine_spec: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Sample parameters from an affine specification."""
    return {name: sample_from_spec(spec, rng) for name, spec in affine_spec.items()}


def repair_affine_parameters(topology: str, affine: Dict[str, Any]) -> Dict[str, Any]:
    """
    Repair geometric constraints in affine parameters.
    
    This is a separate function from repair_parameters for Qt frontend compatibility.
    """
    a = dict(affine)
    
    if topology == "cone_segment":
        a["rmax1_ratio"] = max(a["rmax1_ratio"], a["rmin1_ratio"] + 0.05)
        a["rmax2_ratio"] = max(a["rmax2_ratio"], a["rmin2_ratio"] + 0.05)
        a["deltaphi_deg"] = min(max(a["deltaphi_deg"], 1.0), 359.0)
    
    elif topology == "tube":
        a["outer_radius_ratio"] = max(a["outer_radius_ratio"], a["inner_radius_ratio"] + 0.05)
        a["deltaphi_deg"] = min(max(a["deltaphi_deg"], 1.0), 359.0)
    
    elif topology == "elliptical_tube":
        a["outer_x_radius_ratio"] = max(a["outer_x_radius_ratio"], a["inner_x_radius_ratio"] + 0.05)
        a["outer_y_radius_ratio"] = max(a["outer_y_radius_ratio"], a["inner_y_radius_ratio"] + 0.05)
    
    elif topology in ("torus", "torus_segment"):
        a["major_radius_ratio"] = max(a["major_radius_ratio"], a["minor_radius_ratio"] + 0.10)
    
    if topology == "torus_segment":
        a["deltaphi_deg"] = min(max(a["deltaphi_deg"], 1.0), 359.0)
    
    elif topology == "polycone":
        a["rmax1_ratio"] = max(a["rmax1_ratio"], a["rmin1_ratio"] + 0.05)
        a["rmax2_ratio"] = max(a["rmax2_ratio"], a["rmin2_ratio"] + 0.05)
        a["rmax3_ratio"] = max(a["rmax3_ratio"], a["rmin3_ratio"] + 0.05)
        a["z2_ratio"] = max(a["z2_ratio"], 0.05)
        a["z3_ratio"] = max(a["z3_ratio"], a["z2_ratio"] + 0.05)
        a["deltaphi_deg"] = min(max(a["deltaphi_deg"], 1.0), 359.0)
    
    return a


# ============================================================================
# FREECAD ENVIRONMENT (for Qt frontend compatibility)
# ============================================================================

def build_environment_from_sample(
    sample: Dict[str, Any],
    output_stl: str,
    scale_mm: float = DEFAULT_SCALE_MM,
) -> Dict[str, str]:
    """
    Build environment variables for FreeCAD macro execution from a sample dict.
    
    This is a compatibility wrapper for the Qt frontend which uses a different
    sample structure than the CLI tool.
    """
    env = os.environ.copy()
    
    affine = sample.get("affine", {})
    quaternion = sample.get("realization", {}).get("rotation_quaternion", [0, 0, 0, 1])
    
    output_stl_abs = os.path.abspath(output_stl)
    output_dir_abs = os.path.dirname(output_stl_abs)
    
    affine_json = json.dumps(affine, ensure_ascii=False, separators=(",", ":"))
    rotation_json = json.dumps(quaternion, ensure_ascii=False, separators=(",", ":"))
    
    env["DATASET_GENERATOR_MODE"] = "1"
    env["DATASET_PARAMS_JSON"] = affine_json
    env["DATASET_AFFINE_JSON"] = affine_json
    env["DATASET_AFFINE_PARAMS_JSON"] = affine_json
    env["DATASET_ROTATION_JSON"] = rotation_json
    env["DATASET_ROTATION_QUATERNION_JSON"] = rotation_json
    env["DATASET_ROTATION_X"] = str(quaternion[0])
    env["DATASET_ROTATION_Y"] = str(quaternion[1])
    env["DATASET_ROTATION_Z"] = str(quaternion[2])
    env["DATASET_ROTATION_W"] = str(quaternion[3])
    env["DATASET_SCALE_MM"] = str(scale_mm)
    env["DATASET_REALIZATION_SCALE_MM"] = str(scale_mm)
    env["DATASET_SAMPLE_ID"] = str(sample.get("sample_id", ""))
    env["DATASET_TOPOLOGY"] = str(sample.get("topology", ""))
    env["DATASET_OUT_STL"] = output_stl_abs
    env["DATASET_OUT_DIR"] = output_dir_abs
    
    for parameter_name, value in affine.items():
        safe_param_name = safe_name(parameter_name)
        env[f"DATASET_{safe_param_name}"] = str(value)
        env[f"DATASET_AFFINE_{safe_param_name}"] = str(value)
    
    return env
