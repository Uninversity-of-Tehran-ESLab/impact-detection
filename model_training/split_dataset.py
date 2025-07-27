import os
import random
import shutil

def split_dataset(image_dir, label_dir, output_dir, split_ratios=(0.7, 0.15, 0.15)):
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

    image_names = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    random.shuffle(image_names)

    total_files = len(image_names)
    train_end = int(total_files * split_ratios[0])
    val_end = train_end + int(total_files * split_ratios[1])

    splits = {
        'train': image_names[:train_end],
        'val': image_names[train_end:val_end],
        'test': image_names[val_end:]
    }

    for set_name, filenames in splits.items():
        print(f"Processing {set_name} set ({len(filenames)} files)...")
        for name in filenames:
            # Source paths
            src_img = os.path.join(image_dir, name + '.jpg')
            src_lbl = os.path.join(label_dir, name + '.txt')

            # Destination paths
            dst_img = os.path.join(output_dir, 'images', set_name, name + '.jpg')
            dst_lbl = os.path.join(output_dir, 'labels', set_name, name + '.txt')

            # Move files
            shutil.move(src_img, dst_img)
            shutil.move(src_lbl, dst_lbl)
    
    print("Dataset split successfully!")


if __name__ == '__main__':
    SOURCE_IMAGES_FOLDER = "path/to/all_your/images"
    SOURCE_LABELS_FOLDER = "path/to/all_your/labels"
    OUTPUT_DATASET_FOLDER = "tennis_ball_dataset"

    split_dataset(SOURCE_IMAGES_FOLDER, SOURCE_LABELS_FOLDER, OUTPUT_DATASET_FOLDER)