import json
from multiprocessing.spawn import _main
import os
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List, Tuple
from numbers import Real

def D_P2P(p1: Tuple[float, float], p2: Tuple[float, float], k: float) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.sqrt(dx ** 2 + k * dy ** 2)

def D_C2C(C1: List[Tuple[float, float]], C2: List[Tuple[float, float]], k: float) -> float:
    return np.min([D_P2P(p1, p2, k) for p1 in C1 for p2 in C2])

def hierarchical_clustering(B: List[Tuple[float, float]], d: float, k: float) -> List[List[Tuple[float, float]]]:
    clusters = [[p] for p in B]
    while True:
        distances = np.array([[D_C2C(C1, C2, k) for C2 in clusters] for C1 in clusters])
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        if min_dist >= d:
            break
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        clusters[i].extend(clusters[j])
        del clusters[j]
    return clusters

def merge_clusters_based_on_line_proximity(boxes,clusters: List[List[Tuple[float, float]]], slope_threshold: float, intercept_threshold: float) -> List[List[Tuple[float, float]]]:
    # 存储每个簇的线性回归模型参数
    lines = []

    for cluster in clusters:
        if len(cluster) < 3:     # 修改条件以过滤掉长度小于3的簇
            continue
        X = []
        Y = []
        updated_cluster=[]
        for center in cluster:
            for box in boxes:
                label, (x_min, y_min, x_max, y_max) = box
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                if (center_x, center_y) == center:
                    X.append([center_x])
                    Y.append(y_min)
                    updated_cluster.append((center_x, y_min))
        model = LinearRegression().fit(X, Y)
        slope = model.coef_[0]
        intercept = model.intercept_
        lines.append((slope, intercept, updated_cluster,cluster))  
    
    # 合并基于直线参数接近的簇
    i = 0
    while i < len(lines):
        if i==-1:
            i=0
            continue
        j = i + 1
        while j < len(lines):
            slope_i, intercept_i, cluster_i,cluster_ii = lines[i]
            slope_j, intercept_j, cluster_j,cluster_jj = lines[j]
            slope_diff = abs(slope_i - slope_j)
            intercept_diff = abs(intercept_i - intercept_j)
            
            if slope_diff < slope_threshold and intercept_diff < intercept_threshold:
                # 合并簇
                cluster_i.extend(cluster_j)
                cluster_ii.extend(cluster_jj)
                # 更新直线参数
                X = np.array([[p[0]] for p in cluster_i])
                Y = np.array([p[1] for p in cluster_i])
                new_model = LinearRegression().fit(X, Y)
                new_slope = new_model.coef_[0]
                new_intercept = new_model.intercept_
                lines[i] = (new_slope, new_intercept, cluster_i,cluster_ii)
                # 移除已经合并的簇
                lines.pop(j)
                i=-1
                break
            else:
                j += 1
        i += 1
    
    # 提取合并后的簇，并再次过滤掉长度小于3的簇
    merged_clusters = [line[3] for line in lines if len(line[3]) >= 3]
    return merged_clusters

def assign_clusters_to_boxes(boxes: List[Tuple[str, Tuple[float, float, float, float]]], clusters: List[List[Tuple[float, float]]], k_line: float) -> Tuple[List[Tuple[int, int]], List[List[Tuple[float, float]]]]:
    filtered_clusters = []
    # clusters_with_avg_y = [(cluster, np.mean([p[1] for p in cluster])) for cluster in clusters]
    # clusters_with_avg_y.sort(key=lambda x: x[1])
    # sorted_clusters = [cluster for cluster, _ in clusters_with_avg_y]
    for cluster in clusters:
        if len(cluster) < 3:
            continue
        X = []
        Y = []
        heights = []
        for center in cluster:
            for box in boxes:
                label, (x_min, y_min, x_max, y_max) = box
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                if (center_x, center_y) == center:
                    X.append([center_x])
                    Y.append(y_min)
                    heights.append(y_max - y_min)

        if len(X) > 1:
            model = LinearRegression().fit(X, Y)
            distance_ratios = []
            for i, (x, y) in enumerate(zip(X, Y)):
                predicted_y = model.predict([x])[0]
                distance = abs(predicted_y - y)
                ratio = distance / heights[i]
                distance_ratios.append(ratio)

            if distance_ratios:
                avg_ratio = np.mean(distance_ratios)
                if avg_ratio > k_line:
                    continue

        filtered_clusters.append(sorted(cluster, key=lambda p: p[0]))
    filtered_clusters = sorted (filtered_clusters, key=lambda c: np.mean([y for x,y in c]))

    ans = [(-1, -1)] * len(boxes)
    for new_cluster_idx, cluster in enumerate(filtered_clusters):
        sorted_cluster = sorted(cluster, key=lambda p: p[0])

        for position_idx, center in enumerate(sorted_cluster):
            for box_idx, box in enumerate(boxes):
                label, (x_min, y_min, x_max, y_max) = box
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                if (center_x, center_y) == center:
                    ans[box_idx] = (new_cluster_idx, position_idx)
    return ans, filtered_clusters

def process_boxes(boxes: List[Tuple[str, Tuple[float, float, float, float]]], k: float = 10, k_line: float = 10000000.0) -> Tuple[List[Tuple[int, int]], List[List[Tuple[float, float]]]]:
    B = []
    total_area = 0
    for box in boxes:
        label, (x_min, y_min, x_max, y_max) = box
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        B.append((center_x, center_y))
        area = (x_max - x_min) * (y_max - y_min)
        total_area += area
    num_boxes = len(boxes)
    average_area = total_area / num_boxes if num_boxes > 0 else 0

    d = 0.002233 * average_area + 173.80

    clusters = hierarchical_clustering(B, d, k)
    clusters_merge=merge_clusters_based_on_line_proximity(boxes,clusters,slope_threshold=0.1,intercept_threshold=80)        #slope_threshold (斜率阈值)，intercept_threshold (截距阈值)

    ans, filtered_clusters = assign_clusters_to_boxes(boxes, clusters_merge, k_line)
    return ans, filtered_clusters

def process_interface(boxes: List[Tuple[str, Tuple[float, float, float, float]]]) -> List[Tuple[int, int]]:
    ans, filtered_clusters = process_boxes(boxes, k=10, k_line=1000000000)
    return ans

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

def draw_clusters_on_image(image_path: str, boxes: List[Tuple[str, Tuple[float, float, float, float]]], clusters: List[List[Tuple[float, float]]], output_path: str):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Unable to load image file: {image_path}")
        return
    colors = generate_colors(len(clusters))
    for new_cluster_idx, cluster in enumerate(clusters):
        color = colors[new_cluster_idx % len(colors)]
        X = []
        Y = []
        heights=[]
        for position_idx, center in enumerate(cluster):
            for box in boxes:
                label, (x_min, y_min, x_max, y_max) = box
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                if (center_x, center_y) == center:
                    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
                    cv2.circle(image, (int(center_x), int(center_y)), 5, color, -1)
                    text = f"({new_cluster_idx},{position_idx})"
                    cv2.putText(image, text, (int(center_x), int(center_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    X.append([center_x])
                    Y.append(y_min)
                    heights.append(y_max - y_min)

        if len(X) > 1:
            model = LinearRegression().fit(np.array(X).reshape(-1, 1), Y)
            slope = model.coef_[0]
            intercept = model.intercept_

            line_y_start = model.predict([[0]])[0]
            line_y_end = model.predict([[image.shape[1]]])[0]
            cv2.line(image, (0, int(line_y_start)), (image.shape[1], int(line_y_end)), color, 2)
            
            distance_ratios = []
            for i, (x, y) in enumerate(zip(X, Y)):
                predicted_y = model.predict([x])[0]
                distance = abs(predicted_y - y)
                ratio = distance / heights[i]
                distance_ratios.append(ratio)
            if distance_ratios:
                avg_ratio = np.mean(distance_ratios)
                # avg_ratio_text = f"{new_cluster_idx}: {avg_ratio:.2f}"
                avg_ratio_text = f"{new_cluster_idx}: {avg_ratio:.2f} Slope: {slope:.2f}, Intercept: {intercept:.2f}"
                cv2.putText(image, avg_ratio_text, (10, 30 + 30 * new_cluster_idx), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite(output_path, image)
    print(f"Saved annotated image: {output_path}")

if __name__ == "__main__":

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
    
    json_path = './pic_single/test1.json'
    image_path = "./pic_single/test1.jpg"
    output_path = "./pic_single/output_image.jpg"

    boxes = read_labelme_json(json_path)

    ans, clusters = process_boxes(boxes, k=10, k_line=1000000000)                   #
    print('boxes_len=',len(boxes),'ans_len=',len(ans))
    print(ans)
    # 绘制聚类结果
    draw_clusters_on_image(image_path, boxes, clusters, output_path)


