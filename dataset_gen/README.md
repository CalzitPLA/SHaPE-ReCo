# FCMacros Dataset Generator

A Python toolkit for generating STL datasets from FreeCAD macros for CNN training on 3D geometry recognition.

## Overview

This project provides tools to generate datasets of 3D geometric primitives (STL files) using FreeCAD macros. The datasets are designed for training CNNs to learn:

- **TYPE**: Topology/shape type recognition
- **MEASURE**: Dimension/proportion estimation
- **ROTATION**: Orientation estimation

## Project Structure

```
FCMacros/
├── README.md
├── docs/
│   └── 3D_CNN_architecture.txt    # Conceptual CNN design document
├── examples/
│   └── commands.txt               # Example command-line usage
├── src/
│   ├── __init__.py
│   ├── dataset_generator.py       # CLI dataset generator
│   ├── dataset_utils.py           # Shared utilities
│   └── qt_frontend.py             # PyQt5 GUI frontend
└── FCMacros/                      # FreeCAD macro files
    ├── box_gdml_dataset_learning_CNN.FCMacro
    ├── cone_dataset_learning_CNN.FCMacro
    ├── ...
```

## Installation

### Requirements

- Python 3.8+
- FreeCAD (with `freecadcmd` executable)
- PyQt5 (for GUI frontend)

### Setup

```bash
# Clone or navigate to the project directory
cd ~/Documents/Sandbox/FCMacros

# No additional Python packages required beyond standard library
# For GUI: pip install PyQt5
```

## Usage

### CLI Dataset Generator

The primary interface is the command-line tool `src/dataset_generator.py`.

#### TYPE Dataset (Topology Recognition)

Generate samples with varying dimensions and rotations for each topology:

```bash
python3 -m src.dataset_generator \
    --mode type \
    --topologies box cylinder sphere torus \
    --count-per-type 100 \
    --macro-dir ./FCMacros \
    --output-dir ./dataset/stl/type \
    --seed 12345
```

#### MEASURE Dataset (Dimension Estimation)

Generate samples with fixed dimension ratios but varying rotations:

```bash
python3 -m src.dataset_generator \
    --mode measure \
    --topologies box cylinder \
    --measure-ratios-per-type 10 \
    --count-per-measure 20 \
    --macro-dir ./FCMacros \
    --output-dir ./dataset/stl/measure \
    --seed 12345
```

#### ROTATION Dataset (Orientation Estimation)

Generate samples with fixed rotations but varying dimensions:

```bash
python3 -m src.dataset_generator \
    --mode rotation \
    --topologies box cylinder \
    --rotation-groups-per-type 10 \
    --count-per-rotation 20 \
    --macro-dir ./FCMacros \
    --output-dir ./dataset/stl/rotation \
    --seed 12345
```

#### Resume Interrupted Generation

Skip existing STL files and continue generation:

```bash
python3 -m src.dataset_generator \
    --mode type \
    --topologies box cylinder \
    --count-per-type 100 \
    --macro-dir ./FCMacros \
    --output-dir ./dataset/stl/type \
    --resume
```

#### Additional Options

```bash
--freecad /path/to/freecadcmd    # FreeCAD executable path (default: /usr/bin/freecadcmd)
--scale-mm 10.0                   # Physical realization scale in mm (default: 10.0)
--rotation-min -180.0             # Minimum rotation angle (default: -180.0)
--rotation-max 180.0              # Maximum rotation angle (default: 180.0)
--verbose                         # Enable verbose output
```

### GUI Frontend

A PyQt5-based graphical interface is available:

```bash
python3 src/qt_frontend.py
```

The GUI provides:
- Interactive topology selection
- Parameter specification editing
- Real-time generation progress
- Visual validation of FreeCAD installation

## Supported Topologies

The following 3D geometric primitives are supported:

- `box` - Rectangular box
- `cone` - Conical frustum
- `cone_segment` - Segment of a cone
- `cube` - Cube
- `cylinder` - Cylinder
- `ellipsoid` - Ellipsoid
- `elliptical_cone` - Elliptical cone
- `elliptical_tube` - Elliptical tube
- `general_trapezoid` - General trapezoid
- `paraboloid` - Paraboloid
- `parallelepiped` - Parallelepiped
- `polycone` - Polycone
- `polyhedron` - Polyhedron
- `sphere` - Sphere
- `torus` - Torus
- `torus_segment` - Torus segment
- `tube` - Annular tube

## Parameter Specifications

Each topology has defined parameter ranges in `src/dataset_utils.py`. Parameters are dimensionless ratios that are multiplied by the scale factor to produce actual dimensions.

Example for a box:
```python
"box": {
    "x_half_ratio": {"min": 0.25, "max": 4.00},
    "y_half_ratio": {"min": 0.25, "max": 4.00},
    "z_half_ratio": {"min": 0.25, "max": 4.00},
}
```

## Metadata

Each generated dataset includes a `dataset_metadata.jsonl` file with one JSON line per sample containing:

- `index`: Sample index
- `mode`: Dataset mode (type/measure/rotation)
- `topology`: Topology name
- `parameters`: Dimensionless parameter values
- `rotation`: Rotation (both Euler angles and quaternion)
- `scale_mm`: Physical scale in mm
- `stl_path`: Relative path to STL file
- `seed`: Random seed used

## FreeCAD Macro Interface

FreeCAD macros receive parameters via environment variables:

- `DATASET_TOPOLOGY`: Topology name
- `DATASET_PARAMS_JSON`: JSON string of parameters
- `DATASET_ROTATION_QUATERNION_JSON`: JSON string of quaternion [x,y,z,w]
- `DATASET_SCALE_MM`: Physical scale in mm
- `DATASET_OUT_STL`: Output STL path
- Individual parameters as `DATASET_<PARAMETER_NAME>`

## Development

### Using as a Library

```python
from src.dataset_utils import (
    sample_parameters,
    sample_rotation,
    build_filename,
    Rotation,
)

import random

rng = random.Random(12345)

# Sample parameters for a box
params = sample_parameters("box", rng)

# Sample a rotation
rotation = sample_rotation(rng, -180.0, 180.0)

# Build a filename
filename = build_filename(
    topology="box",
    index=0,
    scale_mm=10.0,
    rotation=rotation,
)
```

## Documentation

- `docs/3D_CNN_architecture.txt` - Conceptual design for a 3D CNN for primitive recognition and iterative decomposition
- `examples/commands.txt` - Additional command-line examples

## License

This project is provided as-is for research and educational purposes.

## Version History

- **2.0.0** - Major refactoring: consolidated generators, extracted shared utilities, integrated GUI
- **1.x** - Initial versions with multiple generator scripts
