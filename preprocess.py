import os
import json
import pandas as pd
import fiftyone as fo
import fiftyone.zoo as foz
from contextlib import contextmanager

# ==========================================
# CONFIGURATION
# ==========================================
# Get the absolute path of the directory containing this script
REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) 
# Define the target root for ALL FiftyOne data *inside* your repository
LOCAL_FIFTYONE_ROOT = os.path.join(REPO_ROOT, "data", "fiftyone")
COCO_DOWNLOAD_DIR = os.path.join(LOCAL_FIFTYONE_ROOT, "coco-2017") 

INPUT_CSV = 'images_val2014.csv' 
TRAIN_OUTPUT = 'MASTER_TRAIN.csv'
VAL_OUTPUT = 'MASTER_VAL.csv'

# ==========================================
# Context Manager to Temporarily Set Environment Variable
# ==========================================
@contextmanager
def set_fiftyone_dir(path):
    # Temporarily sets the global FiftyOne data directory
    original_dir = os.environ.get("FIFTYONE_DATASET_DIR")
    os.environ["FIFTYONE_DATASET_DIR"] = path
    fo.config.dataset_zoo_dir = path
    try:
        yield
    finally:
        if original_dir is not None:
            os.environ["FIFTYONE_DATASET_DIR"] = original_dir
        else:
            del os.environ["FIFTYONE_DATASET_DIR"]
        fo.config.dataset_zoo_dir = original_dir if original_dir else os.path.expanduser("~/fiftyone")

def main():
    print("Step 1: Downloading COCO-2017 Raw Files...")
    
    os.makedirs(LOCAL_FIFTYONE_ROOT, exist_ok=True)
    
    # Use the context manager to force the download location
    with set_fiftyone_dir(LOCAL_FIFTYONE_ROOT):
        foz.download_zoo_dataset(
            "coco-2017",
            splits=["train", "validation"],
            max_samples=None, 
        )
    
    # 2. Path construction now uses the fixed, local directory
    coco_dir = COCO_DOWNLOAD_DIR 
    print(f"Data directory located at: {coco_dir}")

    # ==========================================
    # Step 1.5: Locate Files Dynamically
    # ==========================================
    
    # Standard FiftyOne structure
    train_img_dir = os.path.join(coco_dir, "train", "data")
    val_img_dir = os.path.join(coco_dir, "validation", "data")
    
    # Attempt to locate the caption JSONs in common locations
    possible_train_json = [
        os.path.join(coco_dir, "train", "labels", "captions_train2017.json"),
        os.path.join(coco_dir, "raw", "annotations", "captions_train2017.json"),
        os.path.join(coco_dir, "raw", "captions_train2017.json") 
    ]
    
    possible_val_json = [
        os.path.join(coco_dir, "validation", "labels", "captions_val2017.json"),
        os.path.join(coco_dir, "raw", "annotations", "captions_val2017.json"),
        os.path.join(coco_dir, "raw", "captions_val2017.json")
    ]

    def find_file(options):
        for path in options:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not find caption JSON. Checked: {options}")

    train_json = find_file(possible_train_json)
    val_json = find_file(possible_val_json)
    
    print(f"Found Training Captions at: {train_json}")
    print(f"Found Validation Captions at: {val_json}")
    
    # ==========================================
    # Step 2: Process Race Labels
    # ==========================================
    print("\nStep 2: Processing Race Labels...")
    df = pd.read_csv(INPUT_CSV)
    
    # Filter: Keep only images with clear 'Light' or 'Dark' labels
    df_clean = df[df['bb_skin'].isin(['Light', 'Dark'])].copy()
    
    # Create Binary Target: Light=0, Dark=1
    df_clean['race_label'] = df_clean['bb_skin'].map({'Light': 0, 'Dark': 1})
    
    # Fix Filenames: Convert '136' -> '000000000136.jpg'
    df_clean['filename'] = df_clean['id'].apply(lambda x: f"{int(x):012d}.jpg")
    
    def get_abs_path(row):
        if row['split'] == 'train':
            # Path construction must use the local directory
            return os.path.join(COCO_DOWNLOAD_DIR, "train", "data", row['filename'])
        else:
            return os.path.join(COCO_DOWNLOAD_DIR, "validation", "data", row['filename'])

    df_clean['image_path'] = df_clean.apply(get_abs_path, axis=1)

    # ==========================================
    # Step 3: Load COCO Captions
    # ==========================================
    print("\nStep 3: Loading COCO Captions into Memory...")
    
    def load_coco_captions(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        mapping = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            caption = ann['caption']
            if img_id not in mapping:
                mapping[img_id] = []
            mapping[img_id].append(caption)
        return mapping

    train_caps_map = load_coco_captions(train_json)
    val_caps_map = load_coco_captions(val_json)

    # ==========================================
    # Step 4: Merge and Explode
    # ==========================================
    print("\nStep 4: Merging Race Labels with Captions...")

    def get_captions_for_row(row):
        img_id = int(row['id'])
        if img_id in train_caps_map:
            return train_caps_map[img_id]
        elif img_id in val_caps_map:
            return val_caps_map[img_id]
        else:
            return [] 

    df_clean['captions_list'] = df_clean.apply(get_captions_for_row, axis=1)
    
    # Remove images with no captions
    df_clean = df_clean[df_clean['captions_list'].map(len) > 0]

    # EXPLODE: Turn 1 image into 5 training rows (one for each caption)
    df_exploded = df_clean.explode('captions_list').rename(columns={'captions_list': 'caption_text'})

    # ==========================================
    # Step 5: Save Final Splits
    # ==========================================
    print("\nStep 5: Saving Final CSVs...")
    
    final_train = df_exploded[df_exploded['split'] == 'train']
    final_val = df_exploded[df_exploded['split'] == 'val']
    
    cols_to_keep = ['image_path', 'caption_text', 'race_label', 'bb_skin', 'id']
    
    final_train[cols_to_keep].to_csv(TRAIN_OUTPUT, index=False)
    final_val[cols_to_keep].to_csv(VAL_OUTPUT, index=False)
    
    print(f"\nSUCCESS!")
    print(f"Created {TRAIN_OUTPUT} with {len(final_train)} rows.")
    print(f"Created {VAL_OUTPUT} with {len(final_val)} rows.")

if __name__ == "__main__":
    main()
