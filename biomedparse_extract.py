import os
import shutil
import argparse
from tqdm import tqdm


"""
This script reorganizes BiomedParseData in the format expected for RAE. Only transfers non-mask files.

Usage:
    python biomedparse_extract.py --input_path /path/to/BiomedParseData --output_path /path/to/output_path

Output will look like:
    output_path/
        └── train/
            ├── ACDC
            ├── amos22
            ├── CDD-CESM
            ├── kits23
            ├── LGG
            ...

        └── test/
            ├── ACDC
            ├── amos22
            ├── CDD-CESM
            ├── kits23
            ├── LGG
            ...
"""



IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
def is_image(f): 
    return os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS


def collect_images(src_dataset_root, train_output, test_output):
    walk = list(os.walk(src_dataset_root))

    for root, dirs, files in tqdm(walk, desc=f"Scanning {src_dataset_root}"):

        if "mask" in root.lower():
            continue
        
        dirs[:] = [d for d in dirs if "mask" not in d.lower()]

        for file in tqdm(files, desc=f"Processing {root}", dynamic_ncols=True, leave=False, colour="#6F37D7"):
            if "mask" in file.lower() or not is_image(file):
                continue

            src = os.path.join(root, file)

            rel = os.path.relpath(root, src_dataset_root)
            top_level_dir = rel.split(os.sep)[0]  # only first folder
            dst_root = test_output if "test" in root.lower() else train_output
            dst_dir = os.path.join(dst_root, top_level_dir)
            os.makedirs(dst_dir, exist_ok=True)

            dst = os.path.join(dst_dir, file)

            if not os.path.exists(dst):
                shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='../BiomedParseData')
    parser.add_argument("--output_path", type=str, default='../BiomedParseDataRAE')
    args = parser.parse_args()

    train_output = os.path.join(args.output_path, 'train')
    test_output  = os.path.join(args.output_path, 'test')
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(test_output, exist_ok=True)

    collect_images(args.input_path, train_output, test_output)
    print("\nCompleted.\n")
    

if __name__ == "__main__":
    main()



