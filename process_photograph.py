import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import tkinter as tk
from tkinter import filedialog
import subprocess, platform
from tqdm import tqdm

def open_image_viewer(path):
    system = platform.system()

    if system == "Windows":
        # Use 'start' through cmd so we get a handle
        return subprocess.Popen(["explorer", path])

    elif system == "Darwin":  # macOS
        # Preview stays open, but we can kill it later by name
        return subprocess.Popen(["open", "-a", "Preview", path])

    else:  # Linux
        # Use eog (Eye of GNOME) or fallback viewer
        return subprocess.Popen(["eog", path])

def close_image_viewer(process):
    system = platform.system()

    if system == "Windows":
        process.terminate()  # Try to close process (doesn't always work, needs manual close)

    elif system == "Darwin":  # macOS
        subprocess.run(["pkill", "Preview"])  # Close Preview

    else:  # Linux
        process.terminate()  # Close eog

def find_pic_path(patient_id, picture_root, copy_dir=None, tqdm_handle=None):
    """Find the path to the intraoperative photograph for a given patient ID. If copy_dir is provided, copy the file there."""

    # Define logging function
    log = tqdm_handle.write if tqdm_handle else print

    # Define patient root
    patient_root = os.path.join(picture_root, patient_id)
    if not os.path.exists(patient_root):
        raise FileNotFoundError(f"Patient directory not found: {patient_root}")

    # Check for "raw" subdirectory
    if os.path.exists(os.path.join(patient_root, "raw")):
        patient_root = os.path.join(patient_root, "raw")

    # Check expected path, take that one if exists
    expected_paths = [
        os.path.join(patient_root, f"{patient_id}_postresection.jpg"),
        os.path.join(patient_root, f"{patient_id}_Postresection.jpg"),
        os.path.join(patient_root, f"{patient_id}_post.jpg"),
        os.path.join(patient_root, f"{patient_id}_Post.jpg")
    ]
    for expected_path in expected_paths:
        if os.path.exists(expected_path):
            # Optionally copy
            if copy_dir:
                dest_path = os.path.join(copy_dir, f"{patient_id}_resection_photo.jpg")
                shutil.copy2(expected_path, dest_path)
                return dest_path
            # Else, just return path
            return expected_path

    # Else, search further: Define search pattern and find files
    search_pattern = os.path.join(patient_root, f"*post*.jpg")
    search_pattern_alt = os.path.join(patient_root, f"*Post*.jpg")
    matching_files = glob.glob(search_pattern) + glob.glob(search_pattern_alt)
    if len(matching_files) == 1:
        matching_file = matching_files[0]
    else:
        # Prompt user to select image if multiple or none found
        log(f"Could not uniquely identify photograph for patient ID {patient_id}.")
        log(f"Please select the correct photograph file from the dialog.")
        # Open dialog
        root = tk.Tk()
        root.withdraw()  # hide main window
        root.attributes("-topmost", True)  # bring dialog to front
        # Ask user to select file
        matching_file = filedialog.askopenfilename(
            title=f"Select post-resection photograph for patient {patient_id}",
            initialdir=patient_root,
            filetypes=[("JPEG files", "*.jpg *.jpeg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        if not matching_file:
            log("No file selected. Aborting patient.")
            return None

    # Optionally copy
    if copy_dir:
        dest_path = os.path.join(copy_dir, f"{patient_id}_resection_photo.jpg")
        shutil.copy2(matching_file, dest_path)
        return dest_path
    # Else, just return path
    else:
        return matching_file

def draw_photo_masks(photo_path, save_path=None, tqdm_handle=None):
    """
    Draw two masks (resection & outside) on the photograph and save to .npz.
    Double-click closes the polygon. Press ENTER to confirm each selection.
    """

    # Define logging function
    log = tqdm_handle.write if tqdm_handle else print

    img = np.array(Image.open(photo_path).convert("RGB"))
    h, w, _ = img.shape
    resection_mask = np.zeros((h, w), bool)
    outside_mask = np.zeros((h, w), bool)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.set_title("Draw resection area (double-click to close, ENTER to confirm)")
    plt.tight_layout()
    plt.ion()
    plt.show()

    mask_done = [False]   # mutable flag for closure

    def polygon_to_mask(verts):
        path = Path(verts)
        y, x = np.mgrid[:h, :w]
        coords = np.stack((x.ravel(), y.ravel()), axis=-1)
        return path.contains_points(coords).reshape(h, w)

    def onselect_1(verts):
        resection_mask[:] = polygon_to_mask(verts)
        ax.imshow(np.dstack((img/255.0, np.where(resection_mask, 0.4, 0.0))))
        ax.set_title("Resection drawn. Press ENTER to continue.")
        fig.canvas.draw_idle()

    selector = PolygonSelector(ax, onselect_1, useblit=True)

    def on_key(event):
        if event.key == "enter":
            mask_done[0] = True

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Wait for ENTER after first polygon
    log("Draw resection area → double-click to close → press ENTER to continue.")
    while not mask_done[0]:
        plt.pause(0.1)

    selector.disconnect_events()
    mask_done[0] = False

    # Draw outside region
    ax.imshow(img)
    ax.set_title("Draw OUTSIDE area (double-click to close, ENTER to confirm)")
    selector = PolygonSelector(ax,
                               lambda v: ax.imshow(np.dstack((img/255.0, np.where(polygon_to_mask(v), 0.4, 0.0)))),
                               useblit=True)
    def onselect_2(verts):
        outside_mask[:] = np.logical_not(polygon_to_mask(verts))
        ax.imshow(np.dstack((img/255.0, np.where(outside_mask, 0.4, 0.0))))
        ax.set_title("Outside area drawn. Press ENTER to finish.")
        fig.canvas.draw_idle()
    selector.onselect = onselect_2  # attach proper handler

    log("Draw outside area → double-click to close → press ENTER to finish.")
    while not mask_done[0]:
        plt.pause(0.1)

    selector.disconnect_events()
    plt.close(fig)

    # Save
    if save_path is None:
        base = os.path.splitext(photo_path)[0]
        save_path = base + "_masks.npz"
    np.savez_compressed(save_path,
                        resection_mask=resection_mask,
                        outside_mask=outside_mask)
    log(f"Masks saved to {save_path}")
    return True

def show_photo_with_masks(photo_path, mask_path, save_path=None,
                          resection_color=(0, 0.7, 0, 0.4),
                          outside_color=(0, 0, 0, 0.6),
                          figsize=(10, 10), tqdm_handle=None):
    """
    Display the photograph with resection and outside masks overlayed.

    Parameters
    ----------
    photo_path : str
        Path to the photograph file (e.g., .jpg, .png).
    mask_path : str
        Path to the .npz file containing 'resection_mask' and 'outside_mask'.
    resection_color : tuple
        RGBA color for the resection mask (default: semi-transparent red).
    outside_color : tuple
        RGBA color for the outside mask (default: semi-transparent blue).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    (fig, ax)
        Matplotlib figure and axes handles for further customization.
    """

    # Define logging function
    log = tqdm_handle.write if tqdm_handle else print

    with plt.ioff():
        # Load image and masks
        img = np.array(Image.open(photo_path).convert("RGB"))
        masks = np.load(mask_path)
        resection_mask = masks.get("resection_mask", None)
        outside_mask = masks.get("outside_mask", None)

        # Safety check
        if resection_mask is None or outside_mask is None:
            raise ValueError("Mask file must contain 'resection_mask' and 'outside_mask' arrays.")

        # Plot base image
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        ax.set_title("Pre-operative Photograph (+ masked areas)")
        ax.axis("off")

        # Overlay resection mask (semi-transparent red)
        if resection_mask.any():
            overlay_r = np.zeros((*resection_mask.shape, 4))
            overlay_r[..., 0] = resection_color[0]
            overlay_r[..., 1] = resection_color[1]
            overlay_r[..., 2] = resection_color[2]
            overlay_r[..., 3] = resection_mask.astype(float) * resection_color[3]
            ax.imshow(overlay_r)

        # Overlay outside mask (semi-transparent blue)
        if outside_mask.any():
            overlay_o = np.zeros((*outside_mask.shape, 4))
            overlay_o[..., 0] = outside_color[0]
            overlay_o[..., 1] = outside_color[1]
            overlay_o[..., 2] = outside_color[2]
            overlay_o[..., 3] = outside_mask.astype(float) * outside_color[3]
            ax.imshow(overlay_o)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            log(f"Figure saved to {save_path}")
            plt.close()
        else:
            plt.show()

    return fig, ax