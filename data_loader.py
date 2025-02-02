import os
import torch
import xml.etree.ElementTree as ET
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

class ObjectDetectionDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image file
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Get corresponding annotation file
        annotation_path = os.path.join(self.annotations_dir, img_name.replace('.png', '.xml'))
        boxes, labels = self.parse_annotation(annotation_path)

        if self.transform:
            image = self.transform(image)

        return image, {'boxes': boxes, 'labels': labels}

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            label = obj.find('name').text
            labels.append(label)

            bbox = obj.find('robndbox')
            cx = float(bbox.find('cx').text)
            cy = float(bbox.find('cy').text)
            w = float(bbox.find('w').text)
            h = float(bbox.find('h').text)
            angle = float(bbox.find('angle').text)

            # xmin = cx - w / 2
            # ymin = cy - h / 2
            # xmax = cx + w / 2
            # ymax = cy + h / 2

            #boxes.append([xmin, ymin, xmax, ymax])
            boxes.append([cx, cy, w, h, angle])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor([self.label_to_index(label) for label in labels], dtype=torch.int64)

        return boxes, labels

    def label_to_index(self, label):
        # Convert label name to numeric index (extend as needed)
        label_map = {'ship': 0, 'No ship': 1}  # Example labels
        return label_map.get(label, -1)  # Default to -1 for unknown labels
    
# Dataset directory paths
images_dir = r'D:\myfolder\small_object\code\dataset\images'
annotations_dir = r'D:\myfolder\small_object\code\dataset\annotations'

# Define transforms
transform = transforms.Compose([
    transforms.Resize((300, 300)),  # Resize images to a fixed size
    transforms.ToTensor()
])

# Create dataset and dataloader
Dataset = ObjectDetectionDataset(images_dir, annotations_dir, transform=transform)

#Calculate sizes for splits
total_size = len(Dataset)
train_size = int(0.7 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size  # Ensure no data is left out due to rounding

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(Dataset, [train_size, val_size, test_size])

# Create dataloaders for each split
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
#dataloader = DataLoader(Dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Example usage
'''for images, targets in dataloader:
    print(f"Batch size: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Target boxes: {targets[0]['boxes']}")
    print(f"Target labels: {targets[0]['labels']}")
    break'''

# Example usage
for images, targets in train_loader:
    print(f"Training Batch - Batch size: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Target boxes: {targets[0]['boxes']}")
    print(f"Target labels: {targets[0]['labels']}")
    break

for images, targets in val_loader:
    print(f"Validation Batch - Batch size: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Target boxes: {targets[0]['boxes']}")
    print(f"Target labels: {targets[0]['labels']}")
    break

for images, targets in test_loader:
    print(f"Test Batch - Batch size: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Target boxes: {targets[0]['boxes']}")
    print(f"Target labels: {targets[0]['labels']}")
    break

