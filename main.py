import json
import os
from data_load import read_csv
from create_output_folder import output_dir
from model_selection import dataAugmented
from results_save import csv_save
import shutil
def load_config(config_path="indicators.json"):
    """Load configuration from a JSON file."""
    with open(config_path, "r") as file:
        return json.load(file)

def main():
    # Load configuration
    config = load_config()

    if os.path.exists(config['ResultDir']):
        shutil.rmtree(config['ResultDir'])  # Recursively delete the folder and its contents
        print(f"Deleted existing folder: {config['ResultDir']}")
    # Create a new output directory
    os.makedirs(config['ResultDir'], exist_ok=True)
    # Prepare directories and paths
    train_csv, valid_csv, img_path, train_dir, valid_dir = output_dir(config['ResultDir'])

    # Process each dataset
    for n in range(len(train_csv)):
        # Load data
        train_x, train_y, valid_x, valid_y, train, valid = read_csv(
            os.path.join(train_dir, train_csv[n]),
            os.path.join(valid_dir, valid_csv[n])
        )

        # Perform data augmentation
        dataAugmented(train_x, train_y, valid_x, valid_y, config['SynthtaticDataSize'], config, img_path)

        # Save results
        csv_save(config, valid, valid_y)

if __name__ == "__main__":
    main()