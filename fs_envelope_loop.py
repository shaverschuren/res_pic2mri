import os
import glob
import subprocess
from tqdm import tqdm
from main_slicer_loop import load_config

def slicer_loop(root_dir, slicer_executable):
    """
    Open each patient in 3D Slicer for creating envelope STL files if they don't exist.

    This is a modified version of main_slicer_loop.py focusing on envelope creation only.
    You can run this script first to create missing envelopes before running main_slicer_loop.py 
    for photo to MRI registration. Otherwise you'll have to wait for envelope creation for each 
    patient in main_slicer_loop.py.
    
    Parameters
    ----------
    root_dir : str
        Root directory containing FreeSurfer patient subdirectories
    slicer_executable : str
        Path to the 3D Slicer executable
    """

    # Get patient dirs
    patient_dirs = sorted(glob.glob(os.path.join(root_dir, "RESP*")))

    # Print some info
    print(
        "Running custom Slicer loop for envelope creation only.\n",
        "Runs Slicer in no-main-window mode to enable FreeSurfer import",
        " and creates envelope STL files if they don't exist yet via external MATLAB script\n",
        f"Found {len(patient_dirs)} patient directories in {root_dir}."
    )

    # Loop
    for patient_dir in tqdm(patient_dirs, desc="Processing patients", unit="pt"):
        # Get paths
        patient_id = os.path.split(os.path.basename(patient_dir))[-1]
        t1 = os.path.join(patient_dir, "mri", "T1.nii")
        lh_pial = os.path.join(patient_dir, "surf", "lh.pial")
        rh_pial = os.path.join(patient_dir, "surf", "rh.pial")
        lh_envelope = os.path.join(patient_dir, "surf", "lh_envelope.stl")
        rh_envelope = os.path.join(patient_dir, "surf", "rh_envelope.stl")
        brain_envelope = os.path.join(patient_dir, "surf", "brain_envelope.stl")

        if not os.path.exists(t1) or not os.path.exists(lh_pial) or not os.path.exists(rh_pial):
            tqdm.write(f"Missing required MRI/surface files for {patient_id}, skipping patient.")
            continue

        if not os.path.exists(lh_envelope) or not os.path.exists(rh_envelope) or not os.path.exists(brain_envelope):
            # Open Slicer and wait
            tqdm.write(f"Processing {patient_dir} in 3D Slicer...")
            result = subprocess.run([
                slicer_executable, "--no-main-window",
                "--python-script", os.path.join(os.path.dirname(__file__), "slicer_script.py"),
                "--t1_path", t1, "--lh_pial_path", lh_pial, "--rh_pial_path", rh_pial,
                "--lh_envelope_path", lh_envelope, "--rh_envelope_path", rh_envelope,
                "--brain_envelope_path", brain_envelope, "--create_envelope_mode"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                tqdm.write(f"Error processing {patient_dir}: {result.stderr}")
        else:
            tqdm.write(f"Envelopes already exist for {patient_id}, skipping.")

if __name__ == "__main__":
    # Create or load config, set path variables
    config = load_config()
    slicer_exe_path = config["slicer_exe_path"]
    mri_data_dir = config["mri_data_dir"]
    # Run function
    slicer_loop(mri_data_dir, slicer_exe_path)