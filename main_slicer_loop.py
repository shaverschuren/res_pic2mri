"""
Main loop to open each patient in 3D Slicer for manual photo to MRI registration.
To use, make sure to set the correct paths in the __main__ section.
- `slicer_path`: Path to the Slicer executable.
- `root`: Root directory containing patient subdirectories.
"""

import os
import glob
import subprocess
import process_photograph as pic
import yaml

def load_config(config_path=os.path.join(os.path.dirname(__file__), "config.yaml")):
    """
    Loads configuration from YAML.
    If the config file does not exist, creates a template and exits.
    """
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Software expects config file at '{os.path.abspath(config_path)}'")
        print("Creating template config file. Please edit the paths accordingly and re-run.")
        with open(config_path, "w") as f:
            yaml.dump({
            "slicer_exe_path": "C:\\Path\\To\\Slicer.exe",
            "data_dir": "C:\\Path\\To\\Data\\Root\\Directory"
            }, f)
        exit(0)  # Exit so user can edit paths
    
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Check if paths exist
    if not os.path.exists(cfg["slicer_exe_path"]):
        raise FileNotFoundError(f"Slicer executable not found at '{cfg['slicer_exe_path']}'")
    if not os.path.exists(cfg["data_dir"]):
        raise FileNotFoundError(f"Root directory not found at '{cfg['data_dir']}'")

    return cfg

def main_slicer_loop(data_dir, slicer_executable, patient_dir_regex="RESP*", reprocess=False):
    """Open each patient in 3D slicer for manual photo to MRI registration."""

    # Get patient dirs
    patient_dirs = sorted(glob.glob(os.path.join(data_dir, patient_dir_regex)))

    # Loop
    for patient_dir in patient_dirs[45:]:
        # Get paths
        patient_id = os.path.split(os.path.basename(patient_dir))[-1]
        # t1 = os.path.join(patient_dir, "mri", "orig", "001.mgz")
        t1 = os.path.join(patient_dir, "mri", "T1.mgz")
        ribbon = os.path.join(patient_dir, "mri", "ribbon.mgz")
        lh_pial = os.path.join(patient_dir, "surf", "lh.pial")
        rh_pial = os.path.join(patient_dir, "surf", "rh.pial")
        lh_envelope = os.path.join(patient_dir, "surf", "lh_envelope.stl")
        rh_envelope = os.path.join(patient_dir, "surf", "rh_envelope.stl")
        brain_envelope = os.path.join(patient_dir, "surf", "brain_envelope.stl")
        output_dir = os.path.join(patient_dir, f"pic2mri_output")
        mask_path = os.path.join(output_dir, f"{patient_id}_photo_masks.npz")
        figure_path = os.path.join(output_dir, f"{patient_id}_photo_with_masks.png")
        resection_mask_path = os.path.join(output_dir, f"{patient_id}_resection_mask.nii.gz")

        # Pass if resection mask already exists
        if os.path.exists(resection_mask_path) and not reprocess:
            print(f"Resection mask already exists for {patient_id}, skipping patient.")
            continue

        # Create output dir if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Draw masks
        if not os.path.exists(mask_path):
            # Find photograph path and copy to output dir
            pic_path = pic.find_pic_path(patient_id, copy_dir=output_dir)
            if pic_path:
                print(f"Drawing masks for {patient_id}...")
                masks_drawn = pic.draw_photo_masks(pic_path, save_path=mask_path)
            else:
                masks_drawn = False
        else:
            print(f"Masks already exist for {patient_id}, skipping drawing.")
            pic_path = os.path.join(output_dir, f"{patient_id}_resection_photo.jpg")
            masks_drawn = True

        if not masks_drawn:
            print(f"Skipping {patient_id} (no masks drawn).")
            continue

        # Plot photo with masks
        if not os.path.exists(figure_path):
            pic.show_photo_with_masks(pic_path, mask_path, save_path=figure_path)
        # Open in viewer
        viewer_pid = pic.open_image_viewer(figure_path)

        # Open Slicer and wait
        print(f"Processing {patient_dir} in 3D Slicer...", end=' ', flush=True)
        subprocess.run([
            slicer_executable,
            "--python-script", os.path.join(os.path.dirname(__file__), "slicer_script.py"),
            "--t1_path", t1, "--ribbon_path", ribbon, "--lh_pial_path", lh_pial, "--rh_pial_path", rh_pial,
            "--lh_envelope_path", lh_envelope, "--rh_envelope_path", rh_envelope,
            "--brain_envelope_path", brain_envelope, "--photo_path", pic_path,
            "--mask_path", mask_path, "--output_dir", output_dir
        ])
        # Finished with Slicer, close photo
        pic.close_image_viewer(viewer_pid)
        # OK
        print("\033[92mOK\033[0m", flush=True)

        # TODO: Enable loop, break now for testing
        break

if __name__ == "__main__":
    # Create or load config, set path variables
    cfg = load_config()
    slicer_exe_path = cfg["slicer_exe_path"]
    data_dir = cfg["data_dir"]
    # Run main function
    main_slicer_loop(data_dir, slicer_exe_path)