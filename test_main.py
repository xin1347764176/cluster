import json
import os
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from getboxes import process_interface


def read_labelme_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    boxes = []
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        boxes.append((label, (x_min, y_min, x_max, y_max)))
    return boxes

# Example usage
json_path = './pic_single/test1.json'

boxes = read_labelme_json(json_path)
result = process_interface(boxes)
# new_tuples = [(box[0], res) for box, res in zip(boxes, result)]
print(result)
