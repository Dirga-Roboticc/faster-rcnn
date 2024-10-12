import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import time
import json

# Dataset class for VOC format
class VOCDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
        self.classes = set()
        self._load_classes()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root, "images", self.images[idx])
        img = Image.open(img_path).convert("RGB")

        # Load annotation
        annotation_path = os.path.join(self.root, "Annotations", self.annotations[idx])
        boxes, labels = self.parse_voc_annotation(annotation_path)

        # Convert everything to torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def parse_voc_annotation(self, annotation_path):
        # Parse the XML annotation file
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            label = obj.find('name').text
            labels.append(self.class_to_index(label))  # Convert class name to index
            
            # Get the bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        return np.array(boxes), np.array(labels)

    def _load_classes(self):
        for annotation in self.annotations:
            annotation_path = os.path.join(self.root, "Annotations", annotation)
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                self.classes.add(obj.find('name').text)
        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(sorted(self.classes))}
        self.class_to_idx['background'] = 0
        self.num_classes = len(self.class_to_idx)

    def class_to_index(self, label):
        return self.class_to_idx.get(label, 0)  # Default to background (0)

# Data transformations (basic example)
def get_transform():
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())  # Convert PIL image to Tensor
    return torchvision.transforms.Compose(transforms)

# Load a pre-trained model and modify it for training
def get_model(num_classes):
    # Load a pre-trained model on COCO dataset
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the head with a new one (adjust the number of classes)
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model

# Training function
def train_model(model, data_loader, optimizer, device, epoch, num_epochs, prev_epoch_loss=None):
    model.train()
    epoch_loss = 0
    epoch_loss_classifier = 0
    epoch_loss_box_reg = 0
    epoch_loss_objectness = 0
    epoch_loss_rpn_box_reg = 0
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)

        # Loss calculation
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()
        epoch_loss_classifier += loss_dict['loss_classifier'].item()
        epoch_loss_box_reg += loss_dict['loss_box_reg'].item()
        epoch_loss_objectness += loss_dict['loss_objectness'].item()
        epoch_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {losses.item():.4f}")

    num_batches = len(data_loader)
    epoch_time = time.time() - start_time
    avg_loss = epoch_loss / num_batches

    # Calculate percentage change in loss
    loss_change = ""
    if prev_epoch_loss is not None:
        change = (avg_loss - prev_epoch_loss) / prev_epoch_loss * 100
        loss_change = f" ({change:.2f}%)" if change < 0 else f" (+{change:.2f}%)"

    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']

    # Estimate remaining time
    time_per_epoch = epoch_time
    remaining_epochs = num_epochs - (epoch + 1)
    estimated_time_remaining = time_per_epoch * remaining_epochs

    print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
    print(f"  Total Loss: {epoch_loss:.4f} (Avg: {avg_loss:.4f}){loss_change}")
    print(f"  Classifier Loss: {epoch_loss_classifier:.4f} (Avg: {epoch_loss_classifier/num_batches:.4f})")
    print(f"  Box Reg Loss: {epoch_loss_box_reg:.4f} (Avg: {epoch_loss_box_reg/num_batches:.4f})")
    print(f"  Objectness Loss: {epoch_loss_objectness:.4f} (Avg: {epoch_loss_objectness/num_batches:.4f})")
    print(f"  RPN Box Reg Loss: {epoch_loss_rpn_box_reg:.4f} (Avg: {epoch_loss_rpn_box_reg/num_batches:.4f})")
    print(f"  Learning Rate: {current_lr:.6f}")
    print(f"  Time Taken: {epoch_time:.2f}s")
    print(f"  Estimated Time Remaining: {estimated_time_remaining:.2f}s")

    return avg_loss

# Main training script
def main():
    # Define training device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset = VOCDataset(root="voc-fontawsome-dataset", transforms=get_transform())
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    
    # Number of classes (including background)
    num_classes = dataset.num_classes
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {', '.join(dataset.class_to_idx.keys())}")

    # Load the model
    model = get_model(num_classes)
    print(f"Model: Faster R-CNN with ResNet-50 FPN backbone")

    # Move model to the device
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    print(f"Optimizer: SGD (lr=0.005, momentum=0.9, weight_decay=0.0005)")

    # Train the model for a few epochs
    num_epochs = 10
    print(f"\nStarting training for {num_epochs} epochs...")
    prev_epoch_loss = None
    for epoch in range(num_epochs):
        prev_epoch_loss = train_model(model, data_loader, optimizer, device, epoch, num_epochs, prev_epoch_loss)

    # Save the trained model
    torch.save(model.state_dict(), 'faster_rcnn_model.pth')
    print("Model saved as 'faster_rcnn_model.pth'")

    # Save class information to class_info.json
    class_info = {
        "num_classes": num_classes,
        "classes": [class_name for class_name in sorted(dataset.class_to_idx.keys()) if class_name != 'background']
    }
    with open('class_info.json', 'w') as f:
        json.dump(class_info, f, indent=4)
    print("Class information saved to 'class_info.json'")

if __name__ == "__main__":
    main()
