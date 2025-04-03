import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configure Logging
logging.basicConfig(
    filename="merge_datasets.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Paths
DATASET_1_PATH = "video-datasets/dataset-1"
PROCESSED_DATASETS_PATH = "processed-datasets"
MOVE_FILES = False  # Set to True to move instead of copying
DRY_RUN = False  # Set to True to simulate the merging process

def copy_or_move_file(source_file, target_file):
    """Copy or move a file while avoiding overwrites."""
    if os.path.exists(target_file):  
        base_name, ext = os.path.splitext(target_file)
        counter = 1
        while os.path.exists(target_file):
            target_file = f"{base_name}_{counter}{ext}"
            counter += 1

    if DRY_RUN:
        logging.info(f"DRY RUN: Would copy/move {source_file} to {target_file}")
    else:
        try:
            if MOVE_FILES:
                shutil.move(source_file, target_file)
            else:
                shutil.copy2(source_file, target_file)
            logging.info(f"Successfully copied/moved {source_file} to {target_file}")
        except Exception as e:
            logging.error(f"Error copying/moving {source_file} to {target_file}: {e}")

    return target_file

def process_files(source_label_path, target_label_path, label, split):
    """Process files in parallel using ThreadPoolExecutor."""
    os.makedirs(target_label_path, exist_ok=True)

    if os.path.exists(source_label_path):
        files = os.listdir(source_label_path)
        total_files = len(files)

        if total_files == 0:
            logging.warning(f"No files found in {source_label_path}. Skipping...")
            return

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(
                tqdm(
                    executor.map(
                        lambda file_name: copy_or_move_file(
                            os.path.join(source_label_path, file_name),
                            os.path.join(target_label_path, file_name),
                        ),
                        files,
                    ),
                    total=total_files,
                    desc=f"Merging {split}/{label}",
                )
            )

def merge_datasets(source_path, target_path):
    """Merge dataset-1 into processed-datasets in parallel."""
    for split in ["train", "val", "test"]:
        for label in ["Accident", "Non Accident"]:
            source_label_path = os.path.join(source_path, split, label)
            target_label_path = os.path.join(target_path, split, label)

            process_files(source_label_path, target_label_path, label, split)

    logging.info("✅ Merging complete! Dataset-1 successfully merged into processed-datasets.")

# Run merging process
merge_datasets(DATASET_1_PATH, PROCESSED_DATASETS_PATH)