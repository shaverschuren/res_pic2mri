"""
Main loop to open each patient in 3D Slicer for manual photo to MRI registration.
To use, make sure to set the correct paths in the __main__ section.
- `slicer_path`: Path to the Slicer executable.
- `root`: Root directory containing patient subdirectories.
"""

import os
import glob
import subprocess
import process_photograph
import yaml
from tqdm import tqdm

def load_config(config_path=os.path.join(os.path.dirname(__file__), "config.yaml")):
    """
    Load configuration from YAML file.
    
    If the config file does not exist, creates a template and exits.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the configuration YAML file. Defaults to 'config.yaml' in the package directory.
    
    Returns
    -------
    dict
        Configuration dictionary with keys: 'slicer_exe_path', 'mri_data_dir', 'pic_data_dir'
    
    Raises
    ------
    FileNotFoundError
        If any of the paths specified in the config file do not exist.
    """
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Software expects config file at '{os.path.abspath(config_path)}'")
        print("Creating template config file. Please edit the paths accordingly and re-run.")
        with open(config_path, "w") as f:
            yaml.dump({
            "slicer_exe_path": "C:\\Path\\To\\Slicer.exe",
            "mri_data_dir": "C:\\Path\\To\\MRI\\Data\\Directory",
            "pic_data_dir": "C:\\Path\\To\\Photograph\\Data\\Directory"
            }, f)
        exit(0)  # Exit so user can edit paths
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Check if paths exist
    if not os.path.exists(config["slicer_exe_path"]):
        raise FileNotFoundError(f"Slicer executable not found at '{config['slicer_exe_path']}'")
    if not os.path.exists(config["mri_data_dir"]):
        raise FileNotFoundError(f"Root directory not found at '{config['mri_data_dir']}'")
    if not os.path.exists(config["pic_data_dir"]):
        raise FileNotFoundError(f"Photograph root directory not found at '{config['pic_data_dir']}'")

    return config

def main_slicer_loop(mri_data_dir, pic_data_dir, slicer_executable, patient_dir_regex="RESP*", reprocess=False):
    """
    Open each patient in 3D Slicer for manual photo to MRI registration.
    
    Parameters
    ----------
    mri_data_dir : str
        Root directory containing FreeSurfer patient subdirectories (e.g., RESP001, RESP002, etc.)
    pic_data_dir : str
        Root directory containing photograph subdirectories matching patient IDs
    slicer_executable : str
        Path to the 3D Slicer executable
    patient_dir_regex : str, optional
        Glob pattern for patient directories. Defaults to "RESP*"
    reprocess : bool, optional
        If True, reprocess patients even if resection mask already exists. Defaults to False
    """

    # Get patient dirs
    patient_dirs = sorted(glob.glob(os.path.join(mri_data_dir, patient_dir_regex)))

    # Loop
    print(f"Found {len(patient_dirs)} patient directories to process.")
    for patient_dir in tqdm(patient_dirs, desc="Processing patients:", unit="pt"):
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
        resection_mask_path = os.path.join(output_dir, f"pic2mri_resection_mask.nii.gz")
        atlas_based_flag_path = os.path.join(output_dir, "atlas_based.txt")

        # Pass if missing photo directory
        if not os.path.exists(os.path.join(pic_data_dir, patient_id)):
            tqdm.write(f"No photograph data found for {patient_id}, skipping patient.")
            continue

        # Pass if resection mask already exists
        if os.path.exists(resection_mask_path) and not reprocess:
            tqdm.write(f"Resection mask already exists for {patient_id}, skipping patient.")
            continue
    
        # Pass if atlas-based flag exists
        if os.path.exists(atlas_based_flag_path) and not reprocess:
            tqdm.write(f"Atlas-based resection mask flagged for {patient_id}, skipping patient.")
            continue

        # Create output dir if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Draw masks
        if not os.path.exists(mask_path):
            # Find photograph path and copy to output dir
            photo_path = process_photograph.find_pic_path(patient_id, picture_root=pic_data_dir, copy_dir=output_dir, tqdm_handle=tqdm)
            if photo_path:
                tqdm.write(f"Drawing masks for {patient_id}...")
                masks_drawn = process_photograph.draw_photo_masks(photo_path, save_path=mask_path, tqdm_handle=tqdm)
            else:
                masks_drawn = False
        else:
            tqdm.write(f"Masks already exist for {patient_id}, skipping drawing.")
            photo_path = os.path.join(output_dir, f"{patient_id}_resection_photo.jpg")
            masks_drawn = True

        if not masks_drawn:
            tqdm.write(f"Skipping {patient_id} (no masks drawn).")
            continue

        # Plot photo with masks
        if not os.path.exists(figure_path):
            process_photograph.show_photo_with_masks(photo_path, mask_path, save_path=figure_path)
        # Open in viewer
        viewer_process_id = process_photograph.open_image_viewer(figure_path)

        # Open Slicer and wait
        tqdm.write(f"Processing {patient_dir} in 3D Slicer...")
        subprocess.run([
            slicer_executable,
            "--python-script", os.path.join(os.path.dirname(__file__), "slicer_script.py"),
            "--t1_path", t1, "--ribbon_path", ribbon, "--lh_pial_path", lh_pial, "--rh_pial_path", rh_pial,
            "--lh_envelope_path", lh_envelope, "--rh_envelope_path", rh_envelope,
            "--brain_envelope_path", brain_envelope, "--photo_path", photo_path,
            "--mask_path", mask_path, "--output_dir", output_dir
        ])
        # Finished with Slicer, close photo
        process_photograph.close_image_viewer(viewer_process_id)
        # OK
        tqdm.write("\033[92mDONE\033[0m")

if __name__ == "__main__":
    # Create or load config, set path variables
    config = load_config()
    slicer_exe_path = config["slicer_exe_path"]
    mri_data_dir = config["mri_data_dir"]
    pic_data_dir = config["pic_data_dir"]
    # Run main function
    main_slicer_loop(mri_data_dir, pic_data_dir, slicer_exe_path)