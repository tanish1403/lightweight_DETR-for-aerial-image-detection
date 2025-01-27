import random
import os
import xml.etree.ElementTree as ET
from utils import extract_classes, get_classes
import math

# Configuration
ANN_MODE = 0
TRAIN_PERCENT = 0.8
VAL_PERCENT = 0.1
TEST_PERCENT = 0.1
VOC_ANN_PATH = "..\\dataset\\SCCOS"
OUTPUT_PATH = "organized_dataset"
CLASSES_FILE = "classes.txt"

# Load or extract classes
if not os.path.exists(CLASSES_FILE):
    extract_classes(VOC_ANN_PATH, CLASSES_FILE)
    CLASSES = get_classes(CLASSES_FILE)
else:
    CLASSES, _ = get_classes(CLASSES_FILE)

print(CLASSES)

def parse_annotation(img_id):
    """Parse XML annotations for a given image ID."""
    annotation_file = os.path.join(VOC_ANN_PATH, f'{img_id}.xml')
    if not os.path.exists(annotation_file):
        return []
    
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    annotations = []

    for obj in root.iter('object'):
        difficulty = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        cls = obj.find('name').text

        if cls not in CLASSES or difficulty == 1:
            continue

        cls_id = CLASSES.index(cls)
        bndbox = obj.find('robndbox')
        if bndbox is not None:
            cx = float(bndbox.find('cx').text)
            cy = float(bndbox.find('cy').text)
            w = float(bndbox.find('w').text)
            h = float(bndbox.find('h').text)
            angle = float(bndbox.find('angle').text)

            # Calculate rotated vertices
            hw, hh = w / 2, h / 2
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]

            vertices = [(cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a) for x, y in corners]
            annotations.append((cls_id, vertices))

    return annotations

def write_image_and_annotations(img_id, split_dir):
    """Copy image and write annotations to the split directory."""
    # Copy the image file
    image_src = os.path.join(VOC_ANN_PATH, f"{img_id}.jpg")
    image_dest = os.path.join(split_dir, f"{img_id}.jpg")
    if os.path.exists(image_src):
        os.makedirs(split_dir, exist_ok=True)
        with open(image_dest, "wb") as out_file:
            with open(image_src, "rb") as in_file:
                out_file.write(in_file.read())
    
    # Write annotation file
    annotations = parse_annotation(img_id)
    annotation_file = os.path.join(split_dir, f"{img_id}.txt")
    with open(annotation_file, "w") as file:
        for cls_id, vertices in annotations:
            vertex_str = " ".join(f"{x} {y}" for x, y in vertices)
            file.write(f"{cls_id} {vertex_str}\n")

def split_dataset(train_percent, val_percent, test_percent):
    """Split dataset into train, validation, and test sets."""
    img_ids = [img_id.split('.')[0] for img_id in os.listdir(VOC_ANN_PATH) if img_id.endswith('.xml')]
    random.shuffle(img_ids)

    total = len(img_ids)
    train_lim = int(total * train_percent)
    val_lim = train_lim + int(total * val_percent)

    train_ids = img_ids[:train_lim]
    val_ids = img_ids[train_lim:val_lim]
    test_ids = img_ids[val_lim:]

    return train_ids, val_ids, test_ids

def organize_dataset():
    """Organize dataset into directories and create label files."""
    # Create output directories
    train_dir = os.path.join(OUTPUT_PATH, "train")
    val_dir = os.path.join(OUTPUT_PATH, "val")
    test_dir = os.path.join(OUTPUT_PATH, "test")

    train_ids, val_ids, test_ids = split_dataset(TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT)

    # Process each split
    for split_dir, img_ids in zip([train_dir, val_dir, test_dir], [train_ids, val_ids, test_ids]):
        os.makedirs(split_dir, exist_ok=True)
        for img_id in img_ids:
            write_image_and_annotations(img_id, split_dir)
        print(f"{split_dir} organized with {len(img_ids)} images.")

if __name__ == "__main__":
    random.seed(0)
    organize_dataset()
