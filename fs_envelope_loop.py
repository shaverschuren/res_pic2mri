import os
import glob
import subprocess

def slicer_loop(root_dir, slicer_executable):
    """
    Opens each patient in 3D slicer for creating envelope STL files if they don't exist.

    This is a modified version of main_slicer_loop.py focusing on envelope creation only.
    You can run this script first to create missing envelopes before running main_slicer_loop.py for photo to MRI registration.
    Otherwise you'll have to wait for envelope creation for each patient in main_slicer_loop.py.
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
    for patient_dir in patient_dirs:
        # Get paths
        patient_id = os.path.split(os.path.basename(patient_dir))[-1]
        t1 = os.path.join(patient_dir, "mri", "T1.nii")
        lh_pial = os.path.join(patient_dir, "surf", "lh.pial")
        rh_pial = os.path.join(patient_dir, "surf", "rh.pial")
        lh_envelope = os.path.join(patient_dir, "surf", "lh_envelope.stl")
        rh_envelope = os.path.join(patient_dir, "surf", "rh_envelope.stl")
        brain_envelope = os.path.join(patient_dir, "surf", "brain_envelope.stl")

        if not os.path.exists(lh_envelope) or not os.path.exists(rh_envelope) or not os.path.exists(brain_envelope):
            # Open Slicer and wait
            print(f"Processing {patient_dir} in 3D Slicer...", end=' ', flush=True)
            subprocess.run([
                slicer_executable, "--no-main-window",
                "--python-script", os.path.join(os.path.dirname(__file__), "slicer_script.py"),
                "--t1_path", t1, "--lh_pial_path", lh_pial, "--rh_pial_path", rh_pial,
                "--lh_envelope_path", lh_envelope, "--rh_envelope_path", rh_envelope,
                "--brain_envelope_path", brain_envelope, "--create_envelope_mode"
            ])
            # Finished with Slicer
            print("\033[92mOK\033[0m", flush=True)
        else:
            print(f"Envelopes already exist for {patient_id}, skipping.")

if __name__ == "__main__":
    # Set paths
    slicer_path = "C:\\Users\\sversch6\\AppData\\Local\\slicer.org\\Slicer 5.8.1\\Slicer.exe"
    root = "L:\\her_knf_golf\\Wetenschap\\newtransport\\Sjors\\data\\mri_search\\freesurfer"
    # Run function
    slicer_loop(root, slicer_path)