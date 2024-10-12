import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_model(model_path, num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, image_path, confidence_threshold=0.1):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)
    
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    
    # Filter predictions based on confidence threshold
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Apply non-maximum suppression
    keep = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou_threshold=0.5)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # Ensure boxes, scores, and labels are always 2D arrays
    if len(boxes.shape) == 1:
        boxes = boxes.reshape(1, -1)
    if len(scores.shape) == 0:
        scores = scores.reshape(1)
    if len(labels.shape) == 0:
        labels = labels.reshape(1)
    
    return boxes, scores, labels, image

def visualize_prediction(image, boxes, scores, labels, class_names):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for box, score, label in zip(boxes, scores, labels):
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        class_name = class_names[label - 1]  # Subtract 1 because label 0 is background
        ax.text(box[0], box[1], f'{class_name}: {score:.2f}', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.axis('off')
    plt.show()

import json

def load_class_info(class_info_path):
    with open(class_info_path, 'r') as f:
        class_info = json.load(f)
    return class_info['num_classes'], class_info['class_names']

if __name__ == "__main__":
    model_path = "faster-rcnn-model.pth"
    image_path = "test1.jpg"
    class_info_path = "faster_rcnn_dataset/class_info.json"
    
    num_classes, class_names = load_class_info(class_info_path)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    model = load_model(model_path, num_classes)
    boxes, scores, labels, image = predict(model, image_path)
    
    print(f"Number of detections: {len(boxes)}")
    print(f"Scores: {scores}")
    print(f"Labels: {labels}")
    
    if len(boxes) > 0:
        visualize_prediction(image, boxes, scores, labels, class_names)
    else:
        print("No objects detected in the image.")
        plt.imshow(image)
        plt.axis('off')
        plt.show()
