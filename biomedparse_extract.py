import os
import shutil
import argparse
from tqdm import tqdm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

def is_image(file_name):
    ext = os.path.splitext(file_name)[1].lower()
    return ext in IMAGE_EXTENSIONS

def collect_images(src_dataset_root, dst_dataset_root):
    os.makedirs(dst_dataset_root, exist_ok=True)

    walk = list(os.walk(src_dataset_root))
    for root, dirs, files in tqdm(walk, desc=f"Processing folders in {src_dataset_root}"):
        if "mask" in root.lower():
            continue  # skip entire branch

        dirs[:] = [d for d in dirs if "_mask" not in d]  # skip mask folders
        
        for file in tqdm(files, desc="Copying images", leave=False, colour="yellow"):
            if "_mask" in file:
                continue
            if not is_image(file):
                continue

            src = os.path.join(root, file)

            dst = os.path.join(dst_dataset_root, file)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

def main():
    parser = argparse.ArgumentParser(description="BiomedParse Dataset Extractor")
    parser.add_argument("--input_path", type=str,  default='../BiomedParseData')
    parser.add_argument("--output_path", type=str, default='../BiomedParseDataRAE')

    args = parser.parse_args()

    # Train output 
    train_output = os.path.join(args.output_path, 'train')
    os.makedirs(train_output, exist_ok=True)

    # Test output 
    test_output = os.path.join(args.output_path, 'test')
    os.makedirs(test_output, exist_ok=True)

    for item in os.listdir(args.input_path):
        if "_mask" in item:
            continue

        src = os.path.join(args.input_path, item)

        if os.path.isdir(src):
            if "test" in src.lower():
                print(f"Processing test folder {src}, saving to {test_output}")
                collect_images(src, os.path.join(test_output, item))
            else:
                print(f"Processing folder {src}  saving to  {train_output}")
                collect_images(src, os.path.join(train_output, item))

    print("\nCompleted.")

if __name__ == "__main__":
    main()
