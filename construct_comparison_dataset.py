import os
import shutil

# Define dataset directories
dataset_dirs = [
    "dataset_giant_square_100hp_varyK_1000dp inputs_ixydk+s_outer outputs_t",
    "dataset_giant_square_100hp_varyK_1000dp inputs_pki outputs_t",
    "dataset_giant_square_100hp_varyK_1000dp inputs_pki outputs_xy",
    "dataset_giant_square_100hp_varyK_1000dp inputs_xy outputs_s+s_outer"
]

# Define label files to copy explicitly
selected_files = [
    "RUN_952.pt", "RUN_502.pt", "RUN_560.pt", "RUN_541.pt", "RUN_604.pt",
    "RUN_753.pt", "RUN_503.pt", "RUN_561.pt", "RUN_520.pt", "RUN_103.pt", "RUN_23.pt"
]

# Root directory where original data is stored
data_root = "data"

# Destination directory for copied files
destination_root = "comparison_datasets"

for dataset in dataset_dirs:
    # Construct original data paths
    dataset_path = os.path.join(data_root, dataset)
    input_folder = os.path.join(dataset_path, "Inputs")
    label_folder = os.path.join(dataset_path, "Labels")

    # Construct destination paths
    new_dataset_folder = os.path.join(destination_root, dataset)
    new_inputs_folder = os.path.join(new_dataset_folder, "Inputs")
    new_labels_folder = os.path.join(new_dataset_folder, "Labels")

    # Create destination directories
    os.makedirs(new_inputs_folder, exist_ok=True)
    os.makedirs(new_labels_folder, exist_ok=True)

    # Copy specified files from Inputs and Labels
    for folder, dest_folder in [(input_folder, new_inputs_folder), (label_folder, new_labels_folder)]:
        for file in selected_files:
            src_file = os.path.join(folder, file)
            dest_file = os.path.join(dest_folder, file)
            if os.path.exists(src_file):
                shutil.copy(src_file, dest_file)

    # Copy metadata files
    for meta_file in ["args.yaml", "info.yaml"]:
        src_meta = os.path.join(dataset_path, meta_file)
        dest_meta = os.path.join(new_dataset_folder, meta_file)
        if os.path.exists(src_meta):
            shutil.copy(src_meta, dest_meta)

print("Specified input and label files copied successfully!")
