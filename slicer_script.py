# slicer_script.py
# Author: Sjors
# Date: 14-10-2025

"""
Description: This script is intended to be run inside `3D Slicer` via the --python-script argument.
It automates the loading and setup for manual registration of an intraoperative photograph to MRI.
Developed and tested with 3D Slicer 5.8.1 and SlicerFreeSurfer 05eccb9.

- Loads: T1 volume, FreeSurfer pial surfaces, envelope surfaces (or creates them), intraoperative photograph
- Creates a textured plane with the photograph
- Adds a linear transform so to interactively align the photo
- Creates a markups curve and starts place mode for manual annotation
- Exports curve control points (RAS coordinates) to CSV (when you call export function)
"""

import os
import sys
import argparse
import slicer
import numpy as np
import vtk
import nibabel as nib
import util
from NiBabelModelIO import VolGeom  # type: ignore

def read_fs_surf(path, modelNode, calculateNormals=True):
    """Read a FreeSurfer surface file and set it to the provided model node."""

    try:
        points, polys, metadata = nib.freesurfer.io.read_geometry(path, True)

        geom = VolGeom.from_surf_footer(metadata)

        points = nib.affines.apply_affine(geom.tkreg2scanner(), points).astype('f4')
    
        polyData = vtk.vtkPolyData()
        pointsVTK = vtk.vtkPoints()
        pointsVTK.SetData(vtk.util.numpy_support.numpy_to_vtk(points))
        polyData.SetPoints(pointsVTK)
    
        cellArray = vtk.vtkCellArray()
        polys = polys.astype(np.int64)
        polys = polys.flatten()
        idTypeArray =  vtk.util.numpy_support.numpy_to_vtkIdTypeArray(polys, deep=True)
        cellArray.SetData(3, idTypeArray)
        polyData.SetPolys(cellArray)

        if calculateNormals:
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(polyData)
            normals.ComputePointNormalsOn()
            normals.SplittingOff()
            normals.ConsistencyOn()
            normals.AutoOrientNormalsOn()
            normals.Update()
            polyData = normals.GetOutput()

        # attempt to read corresponding .curv and .sulc files
        curv_path = path.replace("pial", "curv")
        sulc_path = path.replace("pial", "sulc")
        try:
            if os.path.exists(curv_path):
                curv = nib.freesurfer.io.read_morph_data(curv_path)
            else:
                curv = None
        except Exception:
            curv = None
        
        try:
            if os.path.exists(sulc_path):
                sulc = nib.freesurfer.io.read_morph_data(sulc_path)
            else:
                sulc = None
        except Exception:
            sulc = None

        # convert to VTK array for point data
        if curv is not None:
            curvArray = vtk.util.numpy_support.numpy_to_vtk(curv.astype("f4"), deep=True)
            curvArray.SetName("curv")
            polyData.GetPointData().AddArray(curvArray)
        if sulc is not None:
            sulcArray = vtk.util.numpy_support.numpy_to_vtk(sulc.astype("f4"), deep=True)
            sulcArray.SetName("sulc")
            polyData.GetPointData().AddArray(sulcArray)

        modelNode.SetAndObservePolyData(polyData)
    
    except:
        import traceback
        traceback.print_exc()
        return False
    return True

def load_inputs(t1_path, ribbon_path, lh_pial_path, rh_pial_path, lh_envelope_path, rh_envelope_path, brain_envelope_path, photo_path, create_env_mode=False):
    """Load all input data: T1 volume, cortical surface, envelope model, photograph as volume."""

    # Pial surfaces
    print("Loading L pial surface (FreeSurfer curve):", lh_pial_path)
    lh_pialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "lh_pial")
    load_success_lh = read_fs_surf(lh_pial_path, lh_pialNode, calculateNormals=True)
    if not load_success_lh:
        if create_env_mode:
            raise RuntimeError("Failed to load lh.pial model.")
    else:
        print("Cortical model loaded:", lh_pialNode.GetName())

    print("Loading R pial surface (FreeSurfer curve):", rh_pial_path)
    rh_pialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "rh_pial")
    load_success_rh = read_fs_surf(rh_pial_path, rh_pialNode, calculateNormals=True)
    if not load_success_rh:
        if create_env_mode:
            raise RuntimeError("Failed to load rh.pial model.")
    else:
        print("Cortical model loaded:", rh_pialNode.GetName())

    # Envelope surfaces
    if os.path.exists(lh_envelope_path) and os.path.exists(rh_envelope_path) and os.path.exists(brain_envelope_path):
        print("Loading existing envelope surfaces.")
        lh_envelopeNode = util.load_stl_surface(lh_envelope_path, "lh_envelope")
        rh_envelopeNode = util.load_stl_surface(rh_envelope_path, "rh_envelope")
        brain_envelopeNode = util.load_stl_surface(brain_envelope_path, "brain_envelope")
    else:
        print("Envelope surfaces not found. Creating from pial surfaces (this may take a minute)...")
        slicer.app.processEvents()
        # Create envelope models from pial surfaces
        try:
            lh_envelopeNode, rh_envelopeNode, brain_envelopeNode = util.create_envelopes(
                lh_pialNode, rh_pialNode, surf_dir=os.path.dirname(lh_envelope_path))
        except Exception as e:
            print("Error creating envelope surfaces:", e)
            if create_env_mode:
                print("In envelope creation mode, exiting Slicer.")
                slicer.app.processEvents()
                sys.exit(1)

    # If in create envelope mode, skip loading photo and return here
    if create_env_mode:
        return None, None, None, None, lh_envelopeNode, rh_envelopeNode, brain_envelopeNode, None

    # T1 volume
    print("Loading reference T1:", t1_path)
    t1Node = slicer.util.loadVolume(t1_path)
    if not t1Node:
        raise RuntimeError("Failed to load T1 volume.")
    print("T1 loaded:", t1Node.GetName())

    # Ribbon volume
    print("Loading ribbon volume (FreeSurfer GM mask):", ribbon_path)
    ribbonNode = slicer.util.loadVolume(ribbon_path)
    if not ribbonNode:
        raise RuntimeError("Failed to load ribbon volume.")
    print("Ribbon volume loaded:", ribbonNode.GetName())

    # Photograph
    print("Loading photo as volume (will be used as texture):", photo_path)
    # Slicer can load PNG/JPG as a scalar volume node - used as texture source
    photoVolumeNode = slicer.util.loadVolume(photo_path)
    if not photoVolumeNode:
        raise RuntimeError("Failed to load photo image as volume. Check image path.")
    print("Photo image loaded as volume node:", photoVolumeNode.GetName())

    # Show all surfaces except brain envelope
    for modelNode in (lh_pialNode, rh_pialNode, lh_envelopeNode, rh_envelopeNode, brain_envelopeNode):
        displayNode = modelNode.GetDisplayNode()
        if not displayNode:
            displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
            modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())

    # Set initial display properties
    lh_pialDisplayNode = lh_pialNode.GetDisplayNode()
    rh_pialDisplayNode = rh_pialNode.GetDisplayNode()
    lh_envelopeDisplayNode = lh_envelopeNode.GetDisplayNode()
    rh_envelopeDisplayNode = rh_envelopeNode.GetDisplayNode()
    brain_envelopeDisplayNode = brain_envelopeNode.GetDisplayNode()
    lh_pialDisplayNode.SetColor(0.8, 0.8, 1.0)
    lh_pialDisplayNode.SetOpacity(0.8)
    lh_pialDisplayNode.SetVisibility(True)
    rh_pialDisplayNode.SetColor(0.8, 0.8, 1.0)
    rh_pialDisplayNode.SetOpacity(0.8)
    rh_pialDisplayNode.SetVisibility(True)
    lh_envelopeDisplayNode.SetColor(0.0, 0.0, 1.0)
    lh_envelopeDisplayNode.SetOpacity(0.2)
    lh_envelopeDisplayNode.SetVisibility(False)
    rh_envelopeDisplayNode.SetColor(0.0, 0.0, 1.0)
    rh_envelopeDisplayNode.SetOpacity(0.2)
    rh_envelopeDisplayNode.SetVisibility(False)
    brain_envelopeDisplayNode.SetColor(0.0, 0.0, 1.0)
    brain_envelopeDisplayNode.SetOpacity(0.8)
    brain_envelopeDisplayNode.SetVisibility(True)

    return t1Node, ribbonNode, lh_pialNode, rh_pialNode, lh_envelopeNode, rh_envelopeNode, brain_envelopeNode, photoVolumeNode

# Create a plane model which will receive the texture
def create_textured_plane(photoVolumeNode, planeName="PhotoPlane", width=120.0, height=120.0, opacity=0.6):
    # Create vtkPlaneSource
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(-width/2.0, -height/2.0, 0.0)
    plane.SetPoint1(width/2.0, -height/2.0, 0.0)
    plane.SetPoint2(-width/2.0, height/2.0, 0.0)
    plane.SetXResolution(1)
    plane.SetYResolution(1)
    plane.Update()

    # Add as model node
    planeNode = slicer.modules.models.logic().AddModel(plane.GetOutputPort())
    planeNode.SetName(planeName)

    # Make semi-transparent
    displayNode = planeNode.GetModelDisplayNode()
    displayNode.SetOpacity(opacity)
    displayNode.SetBackfaceCulling(False)
    displayNode.SetSelectable(True)
    displayNode.SetVisibility(False)

    # Texture assignment:
    # flip image vertically (needed for correct orientation)
    flip = vtk.vtkImageFlip()
    flip.SetFilteredAxis(1)  # flip vertical axis
    # Connect the photo volume image data
    flip.SetInputConnection(photoVolumeNode.GetImageDataConnection())
    flip.Update()

    # Connect the flipped image pipeline to model display as texture
    displayNode.SetTextureImageDataConnection(flip.GetOutputPort())

    # Naming note: the texture is referenced via the display node's connection (not saved as separate node)
    print("Created textured plane:", planeNode.GetName(), " (texture connected)")

    return planeNode, flip

def setup_interactive_photo_plane(planeNode, envelopeNode, offset_mm=5.0):
    """
    - Places plane on envelope surface + offset
    - Attaches transform
    - Enables immediate interactive handles
    - Constrains dragging to envelope surface + in-plane rotation + uniform scale
    """
    # Create linear transform and attach
    transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "PhotoTransform")
    planeNode.SetAndObserveTransformNodeID(transformNode.GetID())

    # Attach draggable constraint observer
    transformObserver = util.PhotoTransformObserver(transformNode, planeNode, envelopeNode, offset_mm=offset_mm)
    transformObserver.start()

    # Setup interactive handles
    util.setup_interactive_transform(transformNode, visibility=False)

    return transformObserver, transformNode

def main(t1_path, ribbon_path, lh_pial_path, rh_pial_path, lh_envelope_path, rh_envelope_path, brain_envelope_path, photo_path, mask_path, output_dir, create_envelope_mode=False):
    """Main execution function."""

    # On startup, set layout to 3D view only and open Models module and Python console
    if not create_envelope_mode:
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
        slicer.util.selectModule('Models')
        # TODO: Open python interactor

    # Load all inputs
    try:
        t1Node, ribbonNode, lh_pialNode, rh_pialNode, lh_envelopeNode, rh_envelopeNode, brain_envelopeNode, photoVolumeNode = \
            load_inputs(t1_path, ribbon_path, lh_pial_path, rh_pial_path, lh_envelope_path, rh_envelope_path, brain_envelope_path, photo_path, create_envelope_mode)
    except Exception as e:
        print("Error loading inputs:", e)
        if create_envelope_mode:
            slicer.app.processEvents()
            sys.exit(1)
        return

    Nodes = {
        "t1Node": t1Node,
        "ribbonNode": ribbonNode,
        "lh_pialNode": lh_pialNode,
        "rh_pialNode": rh_pialNode,
        "lh_envelopeNode": lh_envelopeNode,
        "rh_envelopeNode": rh_envelopeNode,
        "brain_envelopeNode": brain_envelopeNode,
        "photoVolumeNode": photoVolumeNode
    }

    # If in envelope creation mode, stop here (after loading and creating envelopes)
    if create_envelope_mode:
        print("Envelope creation mode: envelopes created/loaded. Stopping")
        slicer.app.processEvents()
        sys.exit(0)

    # Load photo masks into photo volume
    if mask_path and os.path.exists(mask_path):
        util.load_photo_masks(mask_path, photoVolumeNode)
    else:
        print("No mask file provided or found; proceeding without masking.")

    # Create textured plane sized roughly to cover the envelope — adjust width/height if needed
    # Make width/height relative to image aspect ratio
    imageData = photoVolumeNode.GetImageData()
    dims = imageData.GetDimensions()
    aspect = dims[0] / dims[1] if dims[1] != 0 else 1.0
    base_height = 100.0
    base_width = base_height * aspect
    plane_dims = (base_width, base_height)
    planeNode, _flip = create_textured_plane(
        photoVolumeNode, planeName="ResectionPhotoPlane", width=base_width, height=base_height, opacity=0.6)
    Nodes["planeNode"] = planeNode

    # Attach transform so user can manipulate
    transformObserver, transformNode = setup_interactive_photo_plane(planeNode, brain_envelopeNode, offset_mm=1.0)
    Nodes["transformNode"] = transformNode

    # Setup projection onto brain envelope
    # util.setup_projection_onto_model(Nodes, plane_dims, cam_dist_mm=150.0, proj_slab_thickness_mm=15.0)
    MainProjection = util.Projection(
        brain_envelopeNode, photoVolumeNode, transformNode, plane_dims,
        cam_dist_mm=150.0, proj_slab_thickness_mm=15.0,
        rgb=True, visualize_camera=False
    )

    # Setup slider to control projection camera distance
    cam_dist_slider, observer_toggle, _ = util.setup_ui_widgets(
        MainProjection, transformObserver, transformNode,
        min_mm=1.0, max_mm=300.0, initial_mm=150.0
    )

    # Setup keypress interactor
    util.setup_interactor(Nodes, plane_dims, mask_path, MainProjection, transformObserver, output_dir)

    # Provide variables in global namespace to allow interactive commands after script finishes
    globals()['Nodes'] = Nodes
    globals()['planeNode'] = planeNode
    globals()['transformNode'] = transformNode
    globals()['t1Node'] = t1Node
    globals()['ribbonNode'] = ribbonNode
    globals()['lh_pialNode'] = lh_pialNode
    globals()['rh_pialNode'] = rh_pialNode
    globals()['lh_envelopeNode'] = lh_envelopeNode
    globals()['rh_envelopeNode'] = rh_envelopeNode
    globals()['brain_envelopeNode'] = brain_envelopeNode
    globals()['photoVolumeNode'] = photoVolumeNode
    globals()['photoAspect'] = aspect
    globals()['photoDims'] = dims
    globals()['planeDims'] = (base_width, base_height)
    globals()['MainProjection'] = MainProjection
    globals()['transformObserver'] = transformObserver
    globals()['cam_dist_slider'] = cam_dist_slider
    globals()['observer_toggle'] = observer_toggle
    globals()['output_dir'] = output_dir
    globals()['mask_path'] = mask_path

    print("\n==============================================================================\n")
    print("Setup complete. You can now interactively align the photo projection and draw the resection curve.")
    print("Press <enter> to drop in projection + interactor in current view.")
    print("Press <space> to align the 3D camera with the projection plane.")
    print("Press '1' to toggle the 3D model view.")
    print("Press '2' to toggle the 2D photo view.")
    print("Press 'a' for optional (final) auto-align (WARNING: Not stable yet)")
    print("Press 'v' to create volumetric resection mask.")
    print("Press 's' to save the current scene and resection mask.")
    print("Press 'q' to quit Slicer when done.")
    print("\n==============================================================================\n")

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Slicer automation script for photo-to-MRI alignment.")
    parser.add_argument("--t1_path", type=str, help="Path to T1 volume")
    parser.add_argument("--ribbon_path", type=str, help="Path to 'ribbon' volume (=FreeSurfer's GM mask)")
    parser.add_argument("--lh_pial_path", type=str, help="Path to left hemisphere pial surface")
    parser.add_argument("--rh_pial_path", type=str, help="Path to right hemisphere pial surface")
    parser.add_argument("--lh_envelope_path", type=str, help="Path to left envelope model")
    parser.add_argument("--rh_envelope_path", type=str, help="Path to right envelope model")
    parser.add_argument("--brain_envelope_path", type=str, help="Path to whole brain envelope model")
    parser.add_argument("--photo_path", type=str, help="Path to intraoperative photograph")
    parser.add_argument("--mask_path", type=str, help="Path to photo masks .npz file")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    parser.add_argument("--create_envelope_mode", action="store_true", help="Flag to enable envelope creation loop")
    args = parser.parse_args()

    # Assign to variables
    t1_path = args.t1_path
    ribbon_path = args.ribbon_path
    lh_pial_path = args.lh_pial_path
    rh_pial_path = args.rh_pial_path
    lh_envelope_path = args.lh_envelope_path
    rh_envelope_path = args.rh_envelope_path
    brain_envelope_path = args.brain_envelope_path
    photo_path = args.photo_path
    mask_path = args.mask_path
    output_dir = args.output_dir
    create_envelope_mode = args.create_envelope_mode

    # Run main function
    main(t1_path, ribbon_path, lh_pial_path, rh_pial_path, lh_envelope_path, rh_envelope_path, brain_envelope_path, photo_path, mask_path, output_dir, create_envelope_mode)