#Generate dataset for Faster R-CNN from icon images
import os
import random
from PIL import Image, ImageDraw
import numpy as np
import json
import matplotlib.pyplot as plt

def save_class_info(output_folder):
    annotations_folder = os.path.join(output_folder, "annotations")
    class_set = set()

    for annotation_file in os.listdir(annotations_folder):
        if annotation_file.endswith('.json'):
            with open(os.path.join(annotations_folder, annotation_file), 'r') as f:
                annotations = json.load(f)
                for ann in annotations:
                    class_set.add(ann['category_name'])

    class_names = sorted(list(class_set))
    num_classes = len(class_names) + 1  # Add 1 for background class

    class_info = {
        "num_classes": num_classes,
        "class_names": ["background"] + class_names
    }

    with open(os.path.join(output_folder, "class_info.json"), 'w') as f:
        json.dump(class_info, f, indent=2)

    print(f"Class information saved to {os.path.join(output_folder, 'class_info.json')}")

def create_random_background(width, height):
    color = tuple(np.random.randint(0, 256, 3))
    background = Image.new('RGB', (width, height), color)
    draw = ImageDraw.Draw(background)
    for _ in range(random.randint(3, 10)):
        shape_color = tuple(np.random.randint(0, 256, 3))
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        draw.rectangle([x1, y1, x2, y2], fill=shape_color)
    return background

def resize_icon(icon, max_size):
    icon_width, icon_height = icon.size
    scale = min(max_size[0] / icon_width, max_size[1] / icon_height, 1)
    new_size = (int(icon_width * scale), int(icon_height * scale))
    return icon.resize(new_size, Image.LANCZOS)

def find_empty_space(background, icon_size, occupied_spaces):
    bg_width, bg_height = background.size
    icon_width, icon_height = icon_size
    max_attempts = 100

    for _ in range(max_attempts):
        x = random.randint(0, bg_width - icon_width)
        y = random.randint(0, bg_height - icon_height)
        new_space = (x, y, x + icon_width, y + icon_height)
        
        if not any(overlaps(new_space, space) for space in occupied_spaces):
            return new_space
    
    return None

def overlaps(space1, space2):
    return not (space1[2] <= space2[0] or space1[0] >= space2[2] or
                space1[3] <= space2[1] or space1[1] >= space2[3])

def place_icon_on_background(background, icon, occupied_spaces):
    space = find_empty_space(background, icon.size, occupied_spaces)
    if space is None:
        return None

    x, y, _, _ = space
    background.paste(icon, (x, y), icon)
    return space

def generate_dataset(input_folder, output_folder, num_images=100, bg_size=(800, 600), num_icons=5):
    images_folder = os.path.join(output_folder, "images")
    annotations_folder = os.path.join(output_folder, "annotations")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(annotations_folder, exist_ok=True)
    
    classes = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    
    for i in range(num_images):
        background = create_random_background(*bg_size)
        annotations = []
        occupied_spaces = []
        
        for _ in range(num_icons):
            cls = random.choice(classes)
            cls_folder = os.path.join(input_folder, cls)
            icon_files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            if not icon_files:
                continue
            icon_file = random.choice(icon_files)
            icon_path = os.path.join(cls_folder, icon_file)
            
            try:
                with Image.open(icon_path) as icon:
                    icon = icon.convert("RGBA")
                    icon = resize_icon(icon, (bg_size[0]//2, bg_size[1]//2))
                    space = place_icon_on_background(background, icon, occupied_spaces)
                    
                    if space is not None:
                        occupied_spaces.append(space)
                        annotations.append({
                            "bbox": space,
                            "category_id": class_to_id[cls],
                            "category_name": cls
                        })
            except Exception as e:
                print(f"Error processing {icon_path}: {str(e)}")
        
        image_filename = f"image_{i:04d}.png"
        image_path = os.path.join(images_folder, image_filename)
        background.save(image_path)
        
        annotation_filename = f"annotation_{i:04d}.json"
        annotation_path = os.path.join(annotations_folder, annotation_filename)
        with open(annotation_path, 'w') as f:
            json.dump(annotations, f)

def visualize_dataset(output_folder, num_samples=5):
    images_folder = os.path.join(output_folder, "images")
    annotations_folder = os.path.join(output_folder, "annotations")
    
    image_files = os.listdir(images_folder)
    random.shuffle(image_files)
    
    fig, axs = plt.subplots(1, num_samples, figsize=(20, 4))
    
    for i, image_file in enumerate(image_files[:num_samples]):
        image_path = os.path.join(images_folder, image_file)
        annotation_file = f"annotation_{image_file[6:-4]}.json"
        annotation_path = os.path.join(annotations_folder, annotation_file)
        
        image = Image.open(image_path)
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        
        axs[i].imshow(image)
        axs[i].axis('off')
        
        for ann in annotations:
            bbox = ann['bbox']
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                 fill=False, edgecolor='red', linewidth=2)
            axs[i].add_patch(rect)
            axs[i].text(bbox[0], bbox[1], ann['category_name'], color='red',
                        fontsize=8, backgroundcolor='white')
    
    plt.tight_layout()
    plt.show()

def generate_dataset(input_folder, output_folder, num_images=100, bg_size=(800, 600), num_icons=5):
    # ... (existing code remains unchanged)

    # Add this line at the end of the function
    save_class_info(output_folder)

# Usage
input_folder = "dataset"
output_folder = "faster_rcnn_dataset"
num_images = 1000
num_icons_per_image = 10

generate_dataset(input_folder, output_folder, num_images=num_images, num_icons=num_icons_per_image)
visualize_dataset(output_folder)

print("\nDataset generation complete. Class information has been saved.")
