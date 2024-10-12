import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def load_classes(class_file):
    with open(class_file, 'r') as f:
        classes = f.read().splitlines()
    return ['background'] + classes

def detect_icons(image_path, model, device, classes, confidence_threshold=0.5):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)

    # Process predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    # Filter predictions based on confidence threshold
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    return image, boxes, scores, labels

def draw_boxes(image, boxes, scores, labels, classes):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box.astype(int)
        class_name = classes[label]
        label_text = f"{class_name}: {score:.2f}"

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        draw.text((xmin, ymin - 10), label_text, fill="red", font=font)

    return image

def main():
    # Load class information
    with open('class_info.json', 'r') as f:
        class_info = json.load(f)
    
    num_classes = class_info['num_classes']
    classes = ['background'] + class_info['classes']

    # Set up the device and model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes)
    model.load_state_dict(torch.load('faster_rcnn_model.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Perform detection on a sample image
    image_path = 'test1.png'  # Updated image path
    image, boxes, scores, labels = detect_icons(image_path, model, device, classes)

    # Draw bounding boxes on the image
    result_image = draw_boxes(image, boxes, scores, labels, classes)

    # Save the result
    result_image.save('detection_result.jpg')
    print("Detection result saved as 'detection_result.jpg'")

if __name__ == "__main__":
    main()
