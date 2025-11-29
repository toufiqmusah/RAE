import os
import shutil
import argparse

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

def is_image(file_name):
    ext = os.path.splitext(file_name)[1].lower()
    return ext in IMAGE_EXTENSIONS


def collect_images(src_dataset_root, dst_dataset_root):
    """
    src_dataset_root = BiomedParseData/GlS
    dst_dataset_root = BiomedParseDataset/GlS
    """
    os.makedirs(dst_dataset_root, exist_ok=True)

    for root, _, files in os.walk(src_dataset_root):
        for file in files:
            if not is_image(file):
                continue

            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_dataset_root, file)

            if os.path.exists(dst_file):
                continue

            shutil.copy2(src_file, dst_file)


def main():
    parser = argparse.ArgumentParser(description="BiomedParse Dataset Extractor")
    parser.add_argument("--input_path", type=str, required=True, help="Path to BiomedParseData")
    parser.add_argument("--output_path", type=str, required=True, help="Where to create BiomedParseDataset")

    args = parser.parse_args()

    final_output_dir = os.path.join(args.output_path, "BiomedParseDataset")
    os.makedirs(final_output_dir, exist_ok=True)

    for item in os.listdir(args.input_path):
        src_item_path = os.path.join(args.input_path, item)

        if os.path.isdir(src_item_path):
            dst_item_path = os.path.join(final_output_dir, item)
            # print(f"Processing: {item}")
            collect_images(src_item_path, dst_item_path)

    print("Completed.")


if __name__ == "__main__":
    main()
