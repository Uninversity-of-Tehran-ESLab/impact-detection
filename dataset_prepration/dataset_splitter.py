import os
import random
import shutil

def split_dataset(
    image_dir: str, 
    label_dir: str, 
    output_dir: str, 
    split_ratios: tuple =(0.7, 0.15, 0.15)
) -> None:
    """
    Splits a dataset of images and labels into train, validation, and test sets.

    This function takes directories for images and labels, shuffles them,
    and then splits them according to the provided ratios. It creates the
    standard YOLO directory structure required for training.

    Args:
        image_dir (str): Path to the folder containing all images.
        label_dir (str): Path to the folder containing all .txt label files.
        output_dir (str): The root directory where the split dataset will be created.
        split_ratios (tuple): A tuple containing the (train, val, test) split ratios.
                              Must sum to 1.0.
    """
    if sum(split_ratios) != 1.0:
        print("Error: Split ratios must sum to 1.0")
        return

    sets = ['train', 'val', 'test']
    for s in sets:
        os.makedirs(os.path.join(output_dir, 'images', s), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', s), exist_ok=True)

    image_filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
    random.shuffle(image_filenames)

    total_files = len(image_filenames)
    train_end = int(total_files * split_ratios[0])
    val_end = train_end + int(total_files * split_ratios[1])

    splits = {
        'train': image_filenames[:train_end],
        'val': image_filenames[train_end:val_end],
        'test': image_filenames[val_end:]
    }

    for set_name, filenames in splits.items():
        print(f"Processing {set_name} set ({len(filenames)} files)...")
        for image_filename in filenames:
            base_name = os.path.splitext(image_filename)[0]

            source_image = os.path.join(image_dir, image_filename)
            source_label = os.path.join(label_dir, base_name + '.txt')

            destination_image = os.path.join(output_dir, 'images', set_name, image_filename)
            destination_label = os.path.join(output_dir, 'labels', set_name, base_name + '.txt')

            shutil.move(source_image, destination_image)

            if os.path.exists(source_label):
                shutil.move(source_label, destination_label)
            else:
                print(f"Warning: Label file not found for {image_filename}")

    print("Dataset split successfully!")


if __name__ == '__main__':
    SOURCE_IMAGES_FOLDER = "../raw_dataset/all_frames"
    SOURCE_LABELS_FOLDER = "../raw_dataset/all_labels"
    OUTPUT_DATASET_FOLDER = "../clean_dataset"

    split_dataset(SOURCE_IMAGES_FOLDER, SOURCE_LABELS_FOLDER, OUTPUT_DATASET_FOLDER)