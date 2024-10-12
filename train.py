#Train Faster R-CNN
import os
import json
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class IconDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root_dir, "annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        ann_path = os.path.join(self.root_dir, "annotations", self.annotations[idx])
        
        img = Image.open(img_path).convert("RGB")
        img = F.to_tensor(img)  # Convert PIL Image to PyTorch tensor
        
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
        
        boxes = []
        labels = []
        for ann in annotation:
            bbox = ann['bbox']
            boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            labels.append(ann['category_id'] + 1)  # Add 1 because 0 is background
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train_model(data_dir, num_epochs=10, batch_size=1):
    device = torch.device('cuda')
    
    dataset = IconDataset(data_dir)
    
    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    
    num_classes = max(max(ann['category_id'] for ann in json.load(open(os.path.join(data_dir, "annotations", img_anns)))) for img_anns in dataset.annotations) + 2  # Add 1 for background and 1 because category_id starts from 0
    
    model = get_model(num_classes)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
            elif isinstance(loss_dict, list):
                losses = sum(loss for loss in loss_dict if loss.numel() > 0)
            else:
                losses = loss_dict if isinstance(loss_dict, torch.Tensor) else torch.tensor(loss_dict)
        
            if losses.numel() > 0:
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                total_loss += losses.item()
            else:
                print("Warning: Empty loss tensor encountered. Skipping backward pass.")
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {losses.item():.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                elif isinstance(loss_dict, list):
                    losses = sum(loss for loss in loss_dict if loss.numel() > 0)
                else:
                    losses = loss_dict if isinstance(loss_dict, torch.Tensor) else torch.tensor(loss_dict)
                
                if losses.numel() > 0:
                    total_val_loss += losses.item()
                else:
                    print("Warning: Empty loss tensor encountered during validation.")
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}")
    
    torch.save(model.state_dict(), "faster-rcnn-model.pth")
    print("Model saved as 'faster-rcnn-model.pth'")

if __name__ == "__main__":
    data_dir = "faster_rcnn_dataset"
    train_model(data_dir, num_epochs=10, batch_size=20)
