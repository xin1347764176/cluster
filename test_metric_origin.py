import json
import os
import cv2
import numpy as np
from typing import List, Tuple
from sklearn.linear_model import LinearRegression
from getboxes import process_interface

def read_labelme_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    boxes = []
    for shape in data['shapes']:
        label_str = shape['label']
        try:
            # 处理标签字符串，允许负数和边界情况
            if '--' in label_str:
                label_parts = label_str.split('--')
                label = (int(label_parts[0]), int('-' + label_parts[1]))
            else:
                label_parts = label_str.split('-')
                if len(label_parts) == 2:
                    label = (int(label_parts[0]), int(label_parts[1]))
                else:
                    raise ValueError(f"Invalid label format: {label_str} in file {json_path}")
        except ValueError:
            print(f"Invalid label format: {label_str} in file {json_path}")
            continue
        points = shape['points']
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        boxes.append((label, (x_min, y_min, x_max, y_max)))
    return data, boxes

def generate_colors(n):
    predefined_colors = [
        (0, 0, 0),       # Black
        (255, 255, 255), # White
        (255, 0, 0),     # Red
        (255, 255, 0),   # Yellow
        (0, 0, 255),     # Blue
        (0, 255, 0),     # Green
        (255, 192, 203)  # Pink
    ]
    colors = predefined_colors[:]
    while len(colors) < n:
        color = tuple(int(x) for x in np.random.choice(range(256), size=3))
        if color not in colors:
            colors.append(color)
    return colors

def draw_boxes_on_image(image_path: str, boxes: List[Tuple[Tuple[int, int], Tuple[int, int, int, int]]], output_path: str):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Unable to load image file: {image_path}")
        return
    
    # Generate colors for each unique category
    unique_categories = set(label[0] for label, _ in boxes)
    colors = generate_colors(len(unique_categories))
    color_map = {category: color for category, color in zip(unique_categories, colors)}

    for (category, idx), (x_min, y_min, x_max, y_max) in boxes:
        color = color_map.get(category, (0, 255, 0))  # Default to green if not found
        label_str = f"{category}-{idx}"
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        cv2.putText(image, label_str, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(output_path, image)
    print(f"Saved annotated image: {output_path}")

def calculate_accuracy(directory_path, error_image_output_directory):
    total_boxes = 0
    correct_boxes = 0
    total_images = 0
    correct_images = 0

    if not os.path.exists(error_image_output_directory):
        os.makedirs(error_image_output_directory)

    for file_name in os.listdir(directory_path):
        if file_name.endswith('.json'):
            json_path = os.path.join(directory_path, file_name)
            
            data, boxes = read_labelme_json(json_path)
            result = process_interface(boxes)
            print(f"Processing {json_path}: {result}")
            
            image_correct = True
            for shape, (cluster_id, position_id) in zip(data['shapes'], result):
                try:
                    label_str = shape['label']
                    if '--' in label_str:
                        label_parts = label_str.split('--')
                        true_label = (int(label_parts[0]), int('-' + label_parts[1]))
                    else:
                        label_parts = label_str.split('-')
                        if len(label_parts) == 2:
                            true_label = (int(label_parts[0]), int(label_parts[1]))
                        else:
                            raise ValueError(f"Invalid label format: {label_str} in file {json_path}")
                except ValueError:
                    print(f"Invalid label format: {label_str} in file {json_path}")
                    continue
                predicted_label = (cluster_id, position_id)
                if true_label == predicted_label:
                    correct_boxes += 1
                else:
                    image_correct = False
                total_boxes += 1
            
            if image_correct:
                correct_images += 1
            else:
                image_file_name = file_name.replace('.json', '.jpg')
                image_path = os.path.join(directory_path, image_file_name)
                output_path = os.path.join(error_image_output_directory, image_file_name)
                draw_boxes_on_image(image_path, boxes, output_path)
            total_images += 1

    box_accuracy = correct_boxes / total_boxes if total_boxes > 0 else 0
    image_accuracy = correct_images / total_images if total_images > 0 else 0
    print(f"Boxes Accuracy: {box_accuracy:.2%}")
    print(f"Images Accuracy: {image_accuracy:.2%}")

# Example usage
directory_path = './pic_metric'
error_image_output_directory = './pic_metric_origin'
calculate_accuracy(directory_path, error_image_output_directory)