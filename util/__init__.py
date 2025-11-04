"""
Utility package for 3D Slicer visualization and interaction.

This package is organized into focused modules:
- io: Scene and file I/O operations
- geometry: Geometry and matrix utilities
- projection: Photo projection onto surfaces
- interaction: UI, camera, and interaction handling
"""

# Import commonly used functions to maintain backward compatibility
from .io import save_scene_to_directory, create_envelopes, load_stl_surface, write_stl_surface
from .geometry import (
    vtkMatrixToNumpy, numpyToVtkMatrix, extractRotationScale, rotationFromVectors,
    get_poly_normals, subdivide_model, sample_scalar_along_normals, 
    ras_to_lps_polydata, load_photo_masks
)
from .projection import Projection
from .interaction import (
    PhotoTransformObserver, setup_interactive_transform, 
    center_camera_on_projection, setup_ui_widgets, setup_interactor
)

__all__ = [
    # I/O
    'save_scene_to_directory', 'create_envelopes', 'load_stl_surface', 'write_stl_surface',
    # Geometry
    'vtkMatrixToNumpy', 'numpyToVtkMatrix', 'extractRotationScale', 'rotationFromVectors',
    'get_poly_normals', 'subdivide_model', 'sample_scalar_along_normals', 
    'ras_to_lps_polydata', 'load_photo_masks',
    # Projection
    'Projection',
    # Interaction
    'PhotoTransformObserver', 'setup_interactive_transform', 
    'center_camera_on_projection', 'setup_ui_widgets', 'setup_interactor'
]
