import os
import slicer
import qt               # type: ignore
import vtk
import ctk              # type: ignore
import numpy as np
from vtkmodules.util import numpy_support
import subprocess
import surf2vol
import optimizer

def save_scene_to_directory(directory_path, Nodes):
    """Save the current Slicer scene to a specified directory as a Slicer Data Bundle."""

    print(f"Saving scene to directory: {directory_path}")

    # Remove non-essential elements from scene before saving and setup display for saving.
    slicer.mrmlScene.RemoveNode(Nodes['lh_envelopeNode'])
    slicer.mrmlScene.RemoveNode(Nodes['rh_envelopeNode'])
    setup_interactive_transform(Nodes['transformNode'], visibility=False, limit_to_surf_aligned=False)
    Nodes['lh_pialNode'].GetDisplayNode().SetOpacity(0.7)
    Nodes['rh_pialNode'].GetDisplayNode().SetOpacity(0.7)
    Nodes['brain_envelopeNode'].GetDisplayNode().SetOpacity(0.6)
    center_camera_on_projection(Nodes)

    # Make directory if needed
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    # Process events to ensure scene is up to date
    slicer.app.processEvents()

    # Save scene
    success = slicer.app.applicationLogic().SaveSceneToSlicerDataBundleDirectory(directory_path)
    if success:
        print(f"Scene saved.")
    else:
        print(f"Failed to save scene!")

    # Add screenshot of current 3D view
    image = slicer.app.layoutManager().threeDWidget(0).threeDView().grab()
    screenshot_path = os.path.join(directory_path, "scene_screenshot.png")
    image.save(screenshot_path)

    return success

def create_envelopes(lh_pialNode, rh_pialNode, surf_dir):
    """Create envelope models from pial surface models and save as STL. Uses external MATLAB script."""

    # First, write pial model to STL (in LPS)
    write_stl_surface(lh_pialNode, os.path.join(surf_dir, "lh_pial.stl"))
    write_stl_surface(rh_pialNode, os.path.join(surf_dir, "rh_pial.stl"))

    # Now run the MATLAB script to create envelope from STL
    matlab_scripts_root = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\scripts_other\\res_pic2mri\\MATLAB"
    result = subprocess.run(
        [
            "matlab", "-batch",
            f"addpath('{matlab_scripts_root}', '{os.path.join(matlab_scripts_root, 'Functions')}'); create_envelopes('{surf_dir}')"
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Error running create_envelope.m", result.stderr)
        raise RuntimeError("Envelope creation failed.")
    print(f"Created STL envelopes in: {surf_dir}")

    # Load the created envelope STLs into model nodes
    lh_envelopeNode = load_stl_surface(os.path.join(surf_dir, "lh_envelope.stl"), "lh_envelope")
    rh_envelopeNode = load_stl_surface(os.path.join(surf_dir, "rh_envelope.stl"), "rh_envelope")
    brain_envelopeNode = load_stl_surface(os.path.join(surf_dir, "brain_envelope.stl"), "brain_envelope")

    return lh_envelopeNode, rh_envelopeNode, brain_envelopeNode

def sample_scalar_along_normals(envelopeNode, pialNode, scalarName="curv", rayLength=25.0, attachToEnvelope=True):
    """
    For each point on the envelope surface, cast a ray along the negative normal
    and sample the scalar value from the pial surface beneath it.
    
    Parameters
    ----------
    envelopeNode : vtkMRMLModelNode
        The outer envelope surface node.
    pialNode : vtkMRMLModelNode
        The pial surface node containing the scalar array.
    scalarName : str
        Name of the scalar array on the pial surface to sample.
    rayLength : float
        Maximum length along the negative normal to search for intersection (mm).
    attachToEnvelope : bool
        If True, attach the sampled scalar as a new array on the envelope surface.
    
    Returns
    -------
    sampledScalars : numpy.ndarray
        Array of sampled scalar values for each envelope point (NaN if no intersection).
    """

    print(f"Sampling scalar '{scalarName}' from pial surface onto envelope surface...", flush=True)
    slicer.app.processEvents()

    # Check scalar
    pialScalars = pialNode.GetPolyData().GetPointData().GetArray(scalarName)
    if not pialScalars:
        raise RuntimeError(f"Scalar '{scalarName}' not found on pial surface")
    
    # Get polydata
    pialPoly = pialNode.GetPolyData()
    envPoly = envelopeNode.GetPolyData()
    
    # Compute envelope normals if missing
    envNormals = envPoly.GetPointData().GetNormals()
    if not envNormals:
        normalGenerator = vtk.vtkPolyDataNormals()
        normalGenerator.SetInputData(envPoly)
        normalGenerator.ComputePointNormalsOn()
        normalGenerator.ComputeCellNormalsOff()
        normalGenerator.Update()
        envPoly.DeepCopy(normalGenerator.GetOutput())  # envPoly = normalGenerator.GetOutput()
        envNormals = envPoly.GetPointData().GetNormals()
    
    # Build OBB tree for intersection
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(pialPoly)
    obbTree.BuildLocator()
    
    # Prepare array
    nPoints = envPoly.GetNumberOfPoints()
    sampledScalars = np.zeros(nPoints, dtype=float)
    
    # Point locator for nearest vertex on pial
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(pialPoly)
    locator.BuildLocator()
    
    intersectionPoints = vtk.vtkPoints()
    
    # Loop over envelope points
    for i in range(nPoints):
        pt = np.array(envPoly.GetPoint(i))
        normal = np.array(envNormals.GetTuple(i))
        rayStart = pt
        rayEnd = pt + normal * rayLength
        
        intersectionPoints.Reset()
        code = obbTree.IntersectWithLine(rayStart, rayEnd, intersectionPoints, None)
        
        if intersectionPoints.GetNumberOfPoints() > 0:
            intersectPt = np.array(intersectionPoints.GetPoint(0))
            closestId = locator.FindClosestPoint(intersectPt)
            sampledScalars[i] = pialScalars.GetTuple1(closestId)
        else:
            sampledScalars[i] = np.nan
    
    if attachToEnvelope:
        vtkArray = numpy_support.numpy_to_vtk(sampledScalars.astype("f4"), deep=True)
        vtkArray.SetName(scalarName)
        if envPoly.GetPointData().HasArray(scalarName):
            envPoly.GetPointData().RemoveArray(scalarName)
        envPoly.GetPointData().AddArray(vtkArray)
        envPoly.GetPointData().SetActiveScalars(scalarName)
        envPoly.Modified()
        if envelopeNode.GetDisplayNode():
            envelopeNode.GetDisplayNode().SetActiveScalarName("curv")
            envelopeNode.GetDisplayNode().Modified()
        # envelopeNode.SetAndObservePolyData(envPoly)

    print(f"Completed sampling scalar '{scalarName}'. {np.nansum(np.isnan(sampledScalars))}/{nPoints} points had no intersection.", flush=True)

    return sampledScalars

def ras_to_lps_polydata(polyData):
    """Convert polydata from RAS to LPS coordinate system."""
    transform = vtk.vtkTransform()
    transform.Scale(-1, -1, 1)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(polyData)
    transformFilter.SetTransform(transform)
    transformFilter.Update()
    return transformFilter.GetOutput()

def write_stl_surface(modelNode, stl_path):
    """Write polydata to STL file in LPS coordinate system."""
    polyData = modelNode.GetPolyData()
    transform = vtk.vtkTransform()
    transform.Scale(-1, -1, 1)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(polyData)
    transformFilter.SetTransform(transform)
    transformFilter.Update()
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(stl_path)
    stl_writer.SetInputData(transformFilter.GetOutput())
    stl_writer.Write()
    print(f"Exported model to STL (LPS): {stl_path}")

def load_stl_surface(stl_path, modelNodeName):
    """Load an STL surface into a model node."""
    # Check for file
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    # Add node
    modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", modelNodeName)
    # Read STL
    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(stl_path)
    stl_reader.Update()
    polyData = stl_reader.GetOutput()
    # Set data to node in LPS
    modelNode.SetAndObservePolyData(ras_to_lps_polydata(polyData))
    # Check and return
    if not modelNode:
        raise RuntimeError(f"Failed to load STL model from {stl_path}. Check model path.")
    print(f"STL model loaded: {modelNode.GetName()}")
    return modelNode

def get_poly_normals(polyData):
    """Ensure polydata has point normals and return them."""
    # Get normals
    normals = polyData.GetPointData().GetNormals()
    # Compute if missing
    if normals is None:
        normalsFilter = vtk.vtkPolyDataNormals()
        normalsFilter.SetInputData(polyData)
        normalsFilter.ComputePointNormalsOn()
        normalsFilter.AutoOrientNormalsOn()
        normalsFilter.SplittingOff()
        normalsFilter.ConsistencyOn()
        normalsFilter.Update()
        polyData = normalsFilter.GetOutput()
        normals = polyData.GetPointData().GetNormals()
    return normals

def subdivide_model(modelNode, iterations=1):
    poly = modelNode.GetPolyData()
    subdiv = vtk.vtkLoopSubdivisionFilter()
    subdiv.SetNumberOfSubdivisions(iterations)
    subdiv.SetInputData(poly)
    subdiv.Update()
    modelNode.SetAndObservePolyData(subdiv.GetOutput())

def vtkMatrixToNumpy(vtkMat):
    a = np.zeros((4,4))
    for r in range(4):
        for c in range(4):
            a[r,c] = vtkMat.GetElement(r,c)
    return a

def numpyToVtkMatrix(npMat):
    vtkMat = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            vtkMat.SetElement(r,c, float(npMat[r,c]))
    return vtkMat

def extractRotationScale(mat3):
    U, S, Vt = np.linalg.svd(mat3)
    R = U.dot(Vt)
    if np.linalg.det(R) < 0:
        U[:,-1] *= -1
        R = U.dot(Vt)
    scale = np.mean(S)
    return R, scale

def rotationFromVectors(v_from, v_to):
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)
    if np.allclose(cross,0) and dot > 0.9999:
        return np.eye(3)
    if np.allclose(cross,0) and dot < -0.9999:
        axis = np.array([1,0,0])
        if abs(v_from[0]) > 0.9: axis = np.array([0,1,0])
        axis = axis - np.dot(axis,v_from)*v_from
        axis = axis/np.linalg.norm(axis)
        K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        R = np.eye(3)+2*K.dot(K)
        return R
    K = np.array([[0,-cross[2],cross[1]],[cross[2],0,-cross[0]],[-cross[1],cross[0],0]])
    s = np.linalg.norm(cross)
    c = dot
    R = np.eye(3)+K+K.dot(K)*( (1-c)/(s*s) )
    return R

def load_photo_masks(mask_path, photoVolumeNode,
                     scalar_value_resection=(1, 128, 1),
                     scalar_value_outside=(0, 0, 0)):
    """
    Apply user-drawn resection and outside masks to a photo texture volume (RGB or grayscale).

    - `mask_path`: path to .npz file with 'resection_mask' and 'outside_mask'
    - `photoVolumeNode`: Slicer volume node (e.g., JPG/PNG loaded)
    - Replaces pixel values in masked areas with given RGB or scalar values.
    """

    if not os.path.exists(mask_path):
        print(f"[Mask] File not found: {mask_path}")
        return

    # Load masks
    masks = np.load(mask_path)
    resection_mask = masks.get("resection_mask")
    outside_mask = masks.get("outside_mask")
    if resection_mask is None or outside_mask is None:
        print("[Mask] Missing mask arrays in file.")
        return

    imageData = photoVolumeNode.GetImageData()
    if imageData is None:
        print("[Mask] No image data in photoVolumeNode.")
        return

    dims = imageData.GetDimensions()  # (x, y, z)
    comps = imageData.GetNumberOfScalarComponents()

    vtk_array = imageData.GetPointData().GetScalars()
    np_array = numpy_support.vtk_to_numpy(vtk_array)

    # Reshape properly: (z, y, x, comps)
    arr = np_array.reshape((dims[2], dims[1], dims[0], comps))
    if dims[2] > 1:
        print(f"[Mask] Multi-slice photo (z={dims[2]}), applying to first slice only.")
    img = arr[0]  # (y, x, comps)

    if resection_mask.shape != (dims[1], dims[0]):
        raise ValueError(f"[Mask] Mask shape {resection_mask.shape} != image shape {(dims[1], dims[0])}")

    # Convert scalar → per-component tuple if needed
    def to_vec(val):
        val = np.atleast_1d(val)
        if val.size == 1:
            return np.repeat(val, comps)
        if comps == 4 and val.size == 3:
            # Keep alpha unchanged
            return np.concatenate([val, [img[..., 3].mean()]])
        return np.resize(val, (comps,))

    resec_vals = to_vec(scalar_value_resection)
    outside_vals = to_vec(scalar_value_outside)

    # Apply masks
    modified = img.copy()
    for c in range(comps):
        modified[..., c][resection_mask] = resec_vals[c]
        modified[..., c][outside_mask] = outside_vals[c]

    arr[0] = modified

    # Convert back to VTK
    new_vtk = numpy_support.numpy_to_vtk(num_array=arr.ravel(order='C'), deep=True)
    new_vtk.SetNumberOfComponents(comps)
    imageData.GetPointData().SetScalars(new_vtk)
    imageData.Modified()
    photoVolumeNode.Modified()

    print(f"[Mask] Applied masks to photo volume ({comps} components).")

def center_camera_on_projection(Nodes):
    """Center the 3D view camera on the projection plane normal."""

    threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
    # Get plane normal in world coordinates
    vtkMat = vtk.vtkMatrix4x4()
    Nodes['transformNode'].GetMatrixTransformToWorld(vtkMat)
    R = np.array([[vtkMat.GetElement(i, j) for j in range(3)] for i in range(3)])
    normal = R[:, 2] / np.linalg.norm(R[:, 2])

    # Get plane origin in world coordinates
    origin = np.array([vtkMat.GetElement(i, 3) for i in range(3)])

    # Set camera position and focal point
    camera = threeDView.cameraNode()
    focal_point = origin
    # Move camera to same distance from focal point, but along the plane normal (arc to normal)
    current_position = np.array(camera.GetPosition())
    distance = np.linalg.norm(current_position - focal_point)
    camera_position = focal_point + normal * distance

    camera.SetFocalPoint(*focal_point)
    camera.SetPosition(*camera_position)
    camera.SetViewUp(0, 0, 1)
    threeDView.renderWindow().Render()

class PhotoTransformObserver:
    """
    Enables interactive dragging of a plane along an envelope surface.
    Keeps the plane offset along the normal, allows rotation and scaling, and
    updates smoothly when the transform node changes.

    Usage:
        observer = PhotoTransformObserver(transformNode, planeNode, envelopeModelNode, offset_mm=2.0)
        observer.start()
        ...
        observer.stop()
    """

    def __init__(self, transformNode, planeNode, envelopeModelNode, offset_mm=0.0, initial_scale=0.8):
        self.transformNode = transformNode
        self.planeNode = planeNode
        self.envelopeModelNode = envelopeModelNode
        self.offset_mm = offset_mm
        self.initial_scale = initial_scale

        self._observerTag = None
        self._isUpdating = False
        self._state = {
            'start_point': None,
            'start_translation': None,
            'start_matrix': None,
            'start_normal': None
        }

        # Precompute envelope geometry
        self.envelopePoly = self.envelopeModelNode.GetPolyData()
        if self.envelopePoly is None:
            raise RuntimeError("Envelope polydata is empty")

        self.normals = get_poly_normals(self.envelopePoly)
        self.pointLocator = vtk.vtkPointLocator()
        self.pointLocator.SetDataSet(self.envelopePoly)
        self.pointLocator.BuildLocator()

        # Initialize transform
        self._initialize_transform()

    # ------------------------------------------------------------------
    def _initialize_transform(self):
        bounds = self.envelopePoly.GetBounds()
        lateral_x = bounds[0]
        center_y = (bounds[2] + bounds[3]) / 2
        center_z = (bounds[4] + bounds[5]) / 2
        lateral_edge = np.array([lateral_x, center_y, center_z])
        pid = self.pointLocator.FindClosestPoint(lateral_edge)
        closestPoint = np.array(self.envelopePoly.GetPoint(pid))
        env_normal = np.array(self.normals.GetTuple(pid))
        env_normal /= np.linalg.norm(env_normal)

        # Setup orientation
        z_axis = env_normal
        arbitrary = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.99 else np.array([0, 1, 0])
        x_axis = np.cross(arbitrary, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        initialMat = vtk.vtkMatrix4x4()
        initialMat.Identity()
        for i in range(3):
            initialMat.SetElement(i, 0, x_axis[i] * self.initial_scale)
            initialMat.SetElement(i, 1, y_axis[i] * self.initial_scale)
            initialMat.SetElement(i, 2, z_axis[i] * self.initial_scale)
            initialMat.SetElement(i, 3, closestPoint[i] + env_normal[i] * self.offset_mm)

        self.transformNode.SetMatrixTransformToParent(initialMat)

    # ------------------------------------------------------------------
    def _onModified(self, caller=None, event=None):
        if self._isUpdating:
            return
        self._isUpdating = True
        try:
            vtkMat = vtk.vtkMatrix4x4()
            self.transformNode.GetMatrixTransformToParent(vtkMat)
            mat = vtkMatrixToNumpy(vtkMat)

            trans = mat[:3, 3]
            R_user, scale_user = extractRotationScale(mat[:3, :3])

            # Detect pure scaling
            if (
                self._state["start_translation"] is not None
                and np.allclose(trans, self._state["start_translation"])
                and np.allclose(R_user, extractRotationScale(self._state["start_matrix"][:3, :3])[0])
            ):
                scale_matrix = mat[:3, :3]
                s = np.linalg.norm(scale_matrix[:, 0])
                if not np.allclose(np.linalg.norm(scale_matrix[:, 1]), s) or not np.allclose(np.linalg.norm(scale_matrix[:, 2]), s):
                    # Reset to uniform scale
                    R_user = extractRotationScale(self._state["start_matrix"][:3, :3])[0]
                    new_mat = np.eye(4)
                    new_mat[:3, :3] = R_user * s
                    new_mat[:3, 3] = self._state["start_translation"]
                    vtkNewMat = numpyToVtkMatrix(new_mat)
                    self.transformNode.SetMatrixTransformToParent(vtkNewMat)
                return

            # Envelope lookup
            pid = self.pointLocator.FindClosestPoint(trans)
            closestPoint = np.asarray(self.envelopePoly.GetPoint(pid))
            env_normal = np.asarray(self.normals.GetTuple(pid))
            env_normal /= np.linalg.norm(env_normal)

            # Initialize state
            if self._state["start_point"] is None:
                self._state.update({
                    "start_point": closestPoint,
                    "start_translation": trans.copy(),
                    "start_matrix": mat.copy(),
                    "start_normal": env_normal.copy(),
                })
                return

            new_translation = closestPoint + env_normal * self.offset_mm
            plane_local_z = R_user[:, 2]
            R_align = rotationFromVectors(plane_local_z, env_normal)
            R_final = R_align @ R_user

            new_mat = np.eye(4)
            new_mat[:3, :3] = R_final * scale_user
            new_mat[:3, 3] = new_translation

            self._state.update({
                "start_point": closestPoint,
                "start_translation": new_translation.copy(),
                "start_matrix": new_mat.copy(),
                "start_normal": env_normal.copy(),
            })

            vtkNewMat = numpyToVtkMatrix(new_mat)
            self.transformNode.SetMatrixTransformToParent(vtkNewMat)

        finally:
            self._isUpdating = False

    # ------------------------------------------------------------------
    def start(self):
        """Attach the observer."""
        if self._observerTag is None:
            self._observerTag = self.transformNode.AddObserver(
                slicer.vtkMRMLLinearTransformNode.TransformModifiedEvent, self._onModified
            )
            print(f"PhotoTransformDragger: observing '{self.transformNode.GetName()}'")

    def stop(self):
        """Detach the observer."""
        if self._observerTag is not None:
            self.transformNode.RemoveObserver(self._observerTag)
            print(f"PhotoTransformDragger: stopped observing '{self.transformNode.GetName()}'")
            self._observerTag = None

def setup_interactive_transform(transformNode, visibility=True, limit_to_surf_aligned=True):
    """Setup transform node for interactive manipulation of transform node"""

    # Ensure transform node has a display node
    if not transformNode.GetDisplayNode():
        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformDisplayNode")
        transformNode.SetAndObserveDisplayNodeID(displayNode.GetID())
    else:
        displayNode = transformNode.GetDisplayNode()

    # Enable interactive handles
    if visibility:
        displayNode.SetEditorVisibility(True)
        displayNode.EditorTranslationEnabledOn()
        displayNode.EditorRotationEnabledOn()
        displayNode.EditorScalingEnabledOn()
        displayNode.Visibility3DOn()
        displayNode.Visibility2DOff()

        if limit_to_surf_aligned:
            displayNode.SetRotationHandleComponentVisibility3D(False, False, True, False)       # Z rotation only
            displayNode.SetTranslationHandleComponentVisibility3D(True, True, False, True)      # X/Y/viewing plane translation only
        else:
            displayNode.SetRotationHandleComponentVisibility3D(True, True, True, False)         # XYZ rotations
            displayNode.SetTranslationHandleComponentVisibility3D(True, True, True, True)      # All translations

        displayNode.SetScaleHandleComponentVisibility3D(True, False, False, False)              # X/Y scaling only
        displayNode.SetTranslationHandleComponentVisibilitySlice(False, False, False, False)    # Not in 2D
        displayNode.SetScaleHandleComponentVisibilitySlice(True, False, False, False)           # Not in 2D
        displayNode.SetRotationHandleComponentVisibilitySlice(False, False, False, False)       # Not in 2D
    else:
        displayNode.SetEditorVisibility(False)
        displayNode.Visibility3DOff()
        displayNode.Visibility2DOff()

class Projection:
    """
    Handles projection of a 2D photo volume onto a 3D model surface in 3D Slicer.
    Supports orthographic and perspective projection and automatically updates when
    the associated transform changes.

    Usage:
        proj = Projection(
            modelNode, photoVolumeNode, transformNode,
            plane_dims=(100, 100), cam_dist_mm=100.0,
            rgb=True, visualize_camera=True
        )
        proj.update()  # manually force reproject
        proj.set_cam_distance(120.0)
        proj.remove()  # cleanup observers and visualization
    """

    def __init__(self, modelNode, photoVolumeNode, transformNode, plane_dims,
                 proj_slab_thickness_mm=15.0,
                 rgb=True,
                 upsample_iterations=2,
                 cam_dist_mm=None,
                 plane_to_cortex_distance=0.0,
                 visualize_camera=False):

        self.modelNode = modelNode
        self.photoVolumeNode = photoVolumeNode
        self.transformNode = transformNode
        self.plane_dims = plane_dims
        self.proj_slab_thickness_mm = proj_slab_thickness_mm
        self.rgb = rgb
        self.cam_dist_mm = cam_dist_mm
        self.plane_to_cortex_distance = plane_to_cortex_distance
        self.visualize_camera = visualize_camera
        self.upsample_iterations = upsample_iterations

        # Internal handles
        self._camera_actor = None
        self._rays_actor = None
        self._observer_tag = None

        # Setup
        self._prepare_data()
        self.update()  # initial projection

        # Setup display params
        self.displayNode.SetActiveScalarName("ProjectedPhoto")
        self.displayNode.SetScalarVisibility(True)
        self.displayNode.SetThresholdEnabled(True)
        self.displayNode.SetThresholdRange(1, 255)
        if rgb:
            self.displayNode.SetAutoScalarRange(True)
            self.displayNode.SetScalarRangeFlag(4)
        else:
            self.displayNode.SetAutoScalarRange(False)
            self.displayNode.SetScalarRange(30, 200)
            self.displayNode.SetAndObserveColorNodeID("vtkMRMLColorTableNodeGrey")

        self.displayNode.SetVisibility(False)

        # attach update observer
        self._observer_tag = self.transformNode.AddObserver(
            slicer.vtkMRMLTransformNode.TransformModifiedEvent,
            self._update_projection
        )

        print(f"[Projection] {'Perspective' if self.cam_dist_mm else 'Orthographic'} "
              f"projection set up on '{self.modelNode.GetName()}'.")

    # -------------------------------------------------------------------------
    # Setup helpers
    # -------------------------------------------------------------------------
    def _prepare_data(self):
        """Extract numpy data and precompute model points."""

        # Upsample model if requested
        if self.upsample_iterations is not None:
            subdivide_model(self.modelNode, iterations=self.upsample_iterations)

        # Get model polydata and display node
        self.displayNode = self.modelNode.GetDisplayNode()
        self.envelopePoly = self.modelNode.GetPolyData()
        if self.envelopePoly is None:
            raise RuntimeError("Envelope polydata is empty.")

        # Cache model points
        npts = self.envelopePoly.GetNumberOfPoints()
        self.points_np = np.array(
            [self.envelopePoly.GetPoint(i) for i in range(npts)], dtype=np.float32
        )

        # Load image volume
        arr = slicer.util.arrayFromVolume(self.photoVolumeNode)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        if not self.rgb:
            if arr.shape[-1] >= 3:
                arr = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
            arr = arr.astype(np.uint8)
        else:
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        self.photo_array = arr
        self.img_h, self.img_w = arr.shape[:2]

    # -------------------------------------------------------------------------
    # Core projection logic
    # -------------------------------------------------------------------------
    def _update_projection(self, caller=None, event=None):
        """Observer callback."""
        self.update()

    def update(self):
        """Compute and apply projection to the model."""

        # --- Get transform ---
        vtkMat = vtk.vtkMatrix4x4()
        self.transformNode.GetMatrixTransformToWorld(vtkMat)
        M = np.array([[vtkMat.GetElement(i, j) for j in range(4)] for i in range(4)], dtype=np.float32)
        R = M[:3, :3]
        t = M[:3, 3]
        plane_normal = R[:, 2] / np.linalg.norm(R[:, 2])
        u_axis, v_axis = R[:, 0], R[:, 1]
        plane_origin = t + plane_normal * self.plane_to_cortex_distance

        # --- Camera setup ---
        camera_position = None
        if self.cam_dist_mm is not None:
            camera_position = plane_origin + plane_normal * self.cam_dist_mm

        # --- Projection geometry ---
        pts = self.points_np
        rel = pts - plane_origin
        dist = np.dot(rel, plane_normal)
        mask = np.abs(dist) <= self.proj_slab_thickness_mm
        if not np.any(mask):
            print("\x1b[38;5;208m[Projection] No model points within projection slab thickness.\x1b[0m")
            return

        if camera_position is None:
            proj = pts[mask] - np.outer(dist[mask], plane_normal)
            u = np.dot(proj - plane_origin, u_axis)
            v = np.dot(proj - plane_origin, v_axis)
            valid_points = np.where(mask)[0]
        else:
            mask_points = np.where(mask)[0]
            rays = pts[mask_points] - camera_position
            ray_norms = np.linalg.norm(rays, axis=1)
            rays /= ray_norms[:, None]
            denom = np.dot(rays, plane_normal)
            valid = np.abs(denom) > 1e-6
            valid_points = mask_points[valid]
            t_vals = np.dot(plane_origin - camera_position, plane_normal) / denom[valid]
            hit_points = camera_position + rays[valid] * t_vals[:, None]
            proj = hit_points
            u = np.dot(proj - plane_origin, u_axis)
            v = np.dot(proj - plane_origin, v_axis)

        # --- Convert to image indices ---
        v = -v  # invert vertical axis
        spacing_u = self.plane_dims[0] / self.img_w
        spacing_v = self.plane_dims[1] / self.img_h

        scale_u = np.linalg.norm(R[:, 0]) ** 2
        scale_v = np.linalg.norm(R[:, 1]) ** 2
        photo_size_x_mm = self.img_w * spacing_u * scale_u
        photo_size_y_mm = self.img_h * spacing_v * scale_v

        u_norm = 0.5 + (u / photo_size_x_mm)
        v_norm = 0.5 + (v / photo_size_y_mm)

        valid_mask = (u_norm >= 0) & (u_norm < 1) & (v_norm >= 0) & (v_norm < 1)
        if not np.any(valid_mask):
            raise RuntimeError("No model points project inside the photo image bounds.")
            return

        u_idx = (u_norm[valid_mask] * (self.img_w - 1)).astype(int)
        v_idx = (v_norm[valid_mask] * (self.img_h - 1)).astype(int)
        valid_points = valid_points[valid_mask]

        # --- Sample colors and assign ---
        npts = self.points_np.shape[0]
        if self.rgb:
            sampled = self.photo_array[v_idx, u_idx, :]
            colors = np.zeros((npts, 3), dtype=np.uint8)
            colors[valid_points] = sampled
            vtk_colors = numpy_support.numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            vtk_colors.SetNumberOfComponents(3)
            vtk_colors.SetName("ProjectedPhoto")
            if self.envelopePoly.GetPointData().HasArray("ProjectedPhoto"):
                self.envelopePoly.GetPointData().RemoveArray("ProjectedPhoto")
            self.envelopePoly.GetPointData().AddArray(vtk_colors)
            self.envelopePoly.GetPointData().SetActiveScalars("ProjectedPhoto")
        else:
            sampled = self.photo_array[v_idx, u_idx]
            grays = np.zeros(npts, dtype=np.uint8)
            grays[valid_points] = sampled
            vtk_gray = numpy_support.numpy_to_vtk(grays, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
            vtk_gray.SetNumberOfComponents(1)
            vtk_gray.SetName("ProjectedPhoto")
            if self.envelopePoly.GetPointData().HasArray("ProjectedPhoto"):
                self.envelopePoly.GetPointData().RemoveArray("ProjectedPhoto")
            self.envelopePoly.GetPointData().AddArray(vtk_gray)
            self.envelopePoly.GetPointData().SetActiveScalars("ProjectedPhoto")

        self.envelopePoly.Modified()
        self.displayNode.Modified()
        self.displayNode.SetActiveScalarName("ProjectedPhoto")
        self.displayNode.SetScalarVisibility(True)

        # Optional visualization
        if self.visualize_camera and camera_position is not None:
            self._show_camera_geometry(plane_origin, u_axis, v_axis, plane_normal, camera_position)

    # -------------------------------------------------------------------------
    # Visualization helpers
    # -------------------------------------------------------------------------
    def _show_camera_geometry(self, plane_origin, u_axis, v_axis, plane_normal, camera_position):
        """Draw camera sphere + rays for debugging."""
        self.hide_camera()
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(camera_position)
        sphere.SetRadius(self.cam_dist_mm * 0.02)
        sphere.Update()

        self._camera_actor = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "CameraPosition")
        self._camera_actor.SetAndObservePolyData(sphere.GetOutput())
        disp = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        disp.SetColor(1, 0, 0)
        disp.SetVisibility(True)
        self._camera_actor.SetAndObserveDisplayNodeID(disp.GetID())

        half_w, half_h = self.plane_dims[0] / 2, self.plane_dims[1] / 2
        corners = [
            plane_origin + u_axis * half_w + v_axis * half_h,
            plane_origin - u_axis * half_w + v_axis * half_h,
            plane_origin - u_axis * half_w - v_axis * half_h,
            plane_origin + u_axis * half_w - v_axis * half_h
        ]

        linesPoly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        pid = 0
        for corner in corners:
            points.InsertNextPoint(camera_position)
            points.InsertNextPoint(corner)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, pid)
            line.GetPointIds().SetId(1, pid + 1)
            lines.InsertNextCell(line)
            pid += 2
        linesPoly.SetPoints(points)
        linesPoly.SetLines(lines)

        self._rays_actor = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "CameraRays")
        self._rays_actor.SetAndObservePolyData(linesPoly)
        disp2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        disp2.SetColor(0, 1, 0)
        disp2.SetVisibility(True)
        self._rays_actor.SetAndObserveDisplayNodeID(disp2.GetID())

    def hide_camera(self):
        if self._camera_actor:
            slicer.mrmlScene.RemoveNode(self._camera_actor)
            self._camera_actor = None
        if self._rays_actor:
            slicer.mrmlScene.RemoveNode(self._rays_actor)
            self._rays_actor = None

    # -------------------------------------------------------------------------
    # Parameter control
    # -------------------------------------------------------------------------
    def set_cam_distance(self, dist_mm, update=True):
        self.cam_dist_mm = dist_mm
        if update:
            self.update()

    def set_plane_offset(self, offset_mm, update=True):
        self.plane_to_cortex_distance = offset_mm
        if update:
            self.update()

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    def remove(self):
        """Remove observers and visualization nodes."""
        if self._observer_tag is not None:
            self.transformNode.RemoveObserver(self._observer_tag)
            self._observer_tag = None
        self.hide_camera()
        print("[Projection] Removed observers and visualization.")

def setup_ui_widgets(MainProjection, transformObserver, transformNode,
                     min_mm=1.0, max_mm=300.0, initial_mm=150.0):
    """
    Create a 'Projector Control' dock widget safely attached to the main window.
    Contains camera distance slider + snap toggle, and survives layout changes.
    """

    mainWindow = slicer.util.mainWindow()

    # --- Create a collapsible section as the inner content ---
    collapsibleButton = ctk.ctkCollapsibleButton()
    collapsibleButton.text = "Projector Control"
    collapsibleButton.collapsed = False

    formLayout = qt.QFormLayout(collapsibleButton)
    formLayout.setContentsMargins(8, 4, 8, 8)

    # --- Slider ---
    slider = ctk.ctkSliderWidget()
    slider.minimum = min_mm
    slider.maximum = max_mm
    slider.value = initial_mm
    slider.singleStep = 1
    slider.decimals = 1
    slider.setToolTip("Adjust projection camera distance (mm)")
    slider.setFixedWidth(300)
    formLayout.addRow("Camera distance (mm):", slider)

    # --- Toggle ---
    toggle = ctk.ctkCheckBox()
    toggle.text = "Snap to surface"
    toggle.setChecked(True)
    toggle.setToolTip("Enable/disable dragging constrained to surface")
    formLayout.addRow("Mode:", toggle)

    # --- Make a dock widget ---
    dockWidget = qt.QDockWidget("Projector Control")
    dockWidget.setObjectName("ProjectorControlDock")
    dockWidget.setWidget(collapsibleButton)
    dockWidget.setFeatures(qt.QDockWidget.DockWidgetFloatable | qt.QDockWidget.DockWidgetMovable)

    # --- Add dock widget below the 3D view ---
    mainWindow.addDockWidget(qt.Qt.BottomDockWidgetArea, dockWidget)

    # --- Slider connection ---
    def onValueChanged(value):
        MainProjection.set_cam_distance(value)
    slider.connect('valueChanged(double)', onValueChanged)

    # --- External update timer ---
    timer = qt.QTimer()
    timer.setInterval(200)
    timer.start()

    last_value = MainProjection.cam_dist_mm

    def update_slider():
        nonlocal last_value
        current = MainProjection.cam_dist_mm
        if current != last_value:
            slider.blockSignals(True)
            slider.value = current
            slider.blockSignals(False)
            last_value = current

    timer.timeout.connect(update_slider)

    # --- Toggle observer ---
    def onToggle(state):
        if state:
            transformObserver.start()
            setup_interactive_transform(transformNode, visibility=True, limit_to_surf_aligned=True)
        else:
            transformObserver.stop()
            setup_interactive_transform(transformNode, visibility=True, limit_to_surf_aligned=False)

    toggle.connect('toggled(bool)', onToggle)

    print("[UI] Added 'Projector Control' dock below 3D view")

    return slider, toggle, dockWidget

def setup_interactor(Nodes, plane_dims, photo_mask_path, MainProjection, transformObserver, output_dir):
    """Install keypress handlers on all 3D and slice view interactors.

    This prepares the application to respond to keyboard shortcuts (1,2, space, Return,
    d, s, q) that control layout, camera centering, projection alignment, curve drawing,
    saving, and quitting. Expects `Nodes` to contain at least "transformNode" and
    "brain_envelopeNode".
    """

    # Get app
    app = slicer.app

    # Set up keypress observer
    def onKeyPress(interactor):

        # Get key
        key = interactor.GetKeySym()

        # Switch views with 1, 2 keys
        if key == "1":
            slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
        elif key == "2":
            slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)

        # Center camera on plane with spacebar
        elif key == "space":
            center_camera_on_projection(Nodes)
        
        # Align projection plane to camera view and make projection visible with Enter key
        elif key == "Return":
            # Get camera position
            threeDView = slicer.app.layoutManager().threeDWidget(0).threeDView()
            camera = threeDView.cameraNode()
            cam_pos = np.array(camera.GetPosition())

            # find closest point on model to camera position
            poly = Nodes['brain_envelopeNode'].GetPolyData()
            if poly is None:
                raise RuntimeError("Model polydata is empty")

            locator = vtk.vtkPointLocator()
            locator.SetDataSet(poly)
            locator.BuildLocator()
            pid = locator.FindClosestPoint(tuple(cam_pos))
            closest_pt = np.array(poly.GetPoint(pid))

            # Move transformNode translation to the closest point (preserve rotation/scale)
            vtkMatParent = vtk.vtkMatrix4x4()
            Nodes['transformNode'].GetMatrixTransformToParent(vtkMatParent)
            mat_np = np.array([[vtkMatParent.GetElement(i, j) for j in range(4)] for i in range(4)], dtype=float)
            mat_np[:3, 3] = closest_pt
            vtkNew = numpyToVtkMatrix(mat_np)
            Nodes['transformNode'].SetMatrixTransformToParent(vtkNew)
            # Trigger transform observer once
            transformObserver._onModified()

            # Make projection and transformation interaction visible
            Nodes['brain_envelopeNode'].GetDisplayNode().SetVisibility(True)
            setup_interactive_transform(Nodes['transformNode'], visibility=True, limit_to_surf_aligned=True)

        # Auto-align projection plane with "a" key
        # TODO: Perform check if lh or rh based on coords
        elif key == "a":
            optimizer.auto_align_photo_to_brain(
                Nodes['photoVolumeNode'], Nodes['transformNode'], Nodes['rh_pialNode'],
                Nodes['rh_envelopeNode'], plane_dims, photo_mask_path, MainProjection
                )

        # Make volumetric resection mask from aligned surfaces.
        elif key == "v":
            # Generate segmentation from ribbon and brain envelope
            segmentationNode = surf2vol.project_surface_to_volume_mask(
                Nodes["brain_envelopeNode"], Nodes["ribbonNode"]
            )
            # Store globally for access
            globals()['segmentationNode'] = segmentationNode
            Nodes['segmentationNode'] = segmentationNode
            # Set opacity for visualization
            Nodes['lh_pialNode'].GetDisplayNode().SetOpacity(0.3)
            Nodes['rh_pialNode'].GetDisplayNode().SetOpacity(0.3)
            Nodes['brain_envelopeNode'].GetDisplayNode().SetOpacity(0.4)
        # Save resection curve with "s" key
        elif key == "s":
            segmentationNode = Nodes.get('segmentationNode', None)
            if segmentationNode:
                # Save resection mask to output directory
                output_nifti_path = os.path.join(output_dir, "pic2mri_resection_mask.nii.gz")
                surf2vol.save_resection_mask(segmentationNode, output_nifti_path)
                # Save scene
                scene_path = os.path.join(output_dir, "scene")
                save_scene_to_directory(scene_path, Nodes)

            else:
                print("No segmentationNode found to save.")
        # Exit program with "q" key
        elif key == "q":
            # Check whether to save before quitting
            segmentationNode = Nodes.get('segmentationNode', None)
            if segmentationNode and not (
                os.path.exists(os.path.join(output_dir, "resection_mask.nii.gz")) \
                and os.path.exists(os.path.join(output_dir, "scene"))
            ):
                # Save resection mask to output directory
                output_nifti_path = os.path.join(output_dir, "resection_mask.nii.gz")
                surf2vol.save_resection_mask(segmentationNode, output_nifti_path)
                # Save scene
                scene_path = os.path.join(output_dir, "scene")
                save_scene_to_directory(scene_path, Nodes)
            # Quit application
            app.quit()
        # Ignore other keys
        else:
            return

    # Also install filter on all 3D and slice view interactors
    lm = app.layoutManager()
    for viewIndex in range(lm.threeDViewCount):
        interactor = lm.threeDWidget(viewIndex).threeDView().interactor()
        interactor.AddObserver("KeyPressEvent", lambda obj, evt: onKeyPress(obj))

    for name in lm.sliceViewNames():
        interactor = lm.sliceWidget(name).sliceView().interactor()
        interactor.AddObserver("KeyPressEvent", lambda obj, evt: onKeyPress(obj))