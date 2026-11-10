"""Reconstruction from the closest-surface-point vector field of an STL/STEP model (branch vector-field).

The base is the GEOMETRY of the part, not a voxel grid: dataset_gen/src/closest_point_field.py gives, for any point x, the vector
V(x) to the nearest surface point; reco_field fits analytic primitives (dataset_gen/src/sdf_primitives.py) to it."""
