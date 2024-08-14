import json
import os
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List, Tuple

def D_P2P(p1: Tuple[float, float], p2: Tuple[float, float], k: float) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.sqrt(dx ** 2 + k * dy ** 2)

def calculate_distances(clusters: List[List[Tuple[float, float]]], k: float) -> List[List[np.ndarray]]:
    n = len(clusters)
    distances = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix = np.array([[D_P2P(p1, p2, k) for p2 in clusters[j]] for p1 in clusters[i]])
            distances[i][j] = dist_matrix
            distances[j][i] = dist_matrix.T
    
    return distances


def calculate_areas(clusters: List[List[Tuple[float, float]]]) -> List[List[np.ndarray]]:
    n = len(clusters)
    area_matrix = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            avg_area_matrix = np.array([[np.linalg.norm(np.array(p1) - np.array(p2)) for p2 in clusters[j]] for p1 in clusters[i]]) / 2
            area_matrix[i][j] = avg_area_matrix
            area_matrix[j][i] = avg_area_matrix.T
    return area_matrix

def calculate_thresholds(average_areas: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    n = len(average_areas)
    thresholds = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            threshold_matrix = 0.002233 * average_areas[i][j] +273.80                  #
            thresholds[i][j] = threshold_matrix
            thresholds[j][i] = threshold_matrix.T
    return thresholds

def hierarchical_clustering(B: List[Tuple[float, float]],k: float) -> List[List[Tuple[float, float]]]:
    clusters = [[p] for p in B]
    areas = calculate_areas(clusters)

    areas_clusters = [[a] for a in areas]
    while True:
        distances = calculate_distances(clusters, k)
        average_areas = calculate_areas(clusters)
        thresholds = calculate_thresholds(average_areas)
        # import pdb;pdb.set_trace()                                  #
        found_merge = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # print(min(distances[i][j]))
                if np.any(distances[i][j] <= thresholds[i][j]):
                    clusters[i].extend(clusters[j])
                    areas_clusters[i].extend(areas_clusters[j])
                    del clusters[j]
                    del areas_clusters[j]
                    found_merge = True
                    break
            if found_merge:
                break

        if not found_merge:
            break

    return clusters

def assign_clusters_to_boxes(boxes: List[Tuple[str, Tuple[float, float, float, float]]], clusters: List[List[Tuple[float, float]]], k_line: float) -> Tuple[List[Tuple[int, int]], List[List[Tuple[float, float]]]]:
    filtered_clusters = []
    clusters_with_avg_y = [(cluster, np.mean([p[1] for p in cluster])) for cluster in clusters]
    clusters_with_avg_y.sort(key=lambda x: x[1])
    sorted_clusters = [cluster for cluster, _ in clusters_with_avg_y]
    for cluster in sorted_clusters:
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
    


    clusters = hierarchical_clustering(B, k)

    ans, filtered_clusters = assign_clusters_to_boxes(boxes, clusters, k_line)
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
            model = LinearRegression().fit(X, Y)
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
                avg_ratio_text = f"{new_cluster_idx}: {avg_ratio:.2f}"
                cv2.putText(image, avg_ratio_text, (10, 30 + 30 * new_cluster_idx), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite(output_path, image)
    print(f"Saved annotated image: {output_path}")

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

if __name__ == "__main__":
    # json_dir = '/svap_intern/allenzzeng/code/work3/single'
    # image_dir = '/svap_intern/allenzzeng/code/work3/single'
    # output_dir = '/svap_intern/allenzzeng/code/work3/single'
    json_dir = './pic_single'
    image_dir = './pic_single'
    output_dir = './pic_single'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        image_path = os.path.join(image_dir, json_file.replace('.json', '.jpg'))
        output_path = os.path.join(output_dir, json_file.replace('.json', '_output.jpg'))

        if not os.path.exists(image_path):
            print(f"Image file {image_path} does not exist, skipping.")
            continue

        boxes = read_labelme_json(json_path)

        ans, clusters = process_boxes(boxes, k=10, k_line=1000000000)
        print(f'Processing {json_file}: boxes_len={len(boxes)}, ans_len={len(ans)}')
        print(ans)

        # 绘制聚类结果
        draw_clusters_on_image(image_path, boxes, clusters, output_path)



# 考虑根据图内每个检测框的面积，d = 0.002233 * average_area + 173.80
# 定义每个框的代表的值，2个框之间的值取平均作为阈值决定是否形成一个簇

# 初始化阶段：
# 对于每个边界框，计算其中心点 (center_x, center_y) 和面积 area。
# 将每个边界框初始化为一个单独的聚类。

# 计算簇之间的距离和面积：
# 构建一个矩阵 distances，其中 distances[i][j] 表示簇 i 和簇 j 之间的所有点对的距离。
# 构建一个矩阵 areas，其中 areas[i][j] 表示簇 i 和簇 j 之间的所有点对的面积平均值。
# 对于每对簇 (C1, C2)，计算簇内所有点对之间的距离 D_P2P 和面积平均值 average_area。

# 计算簇之间的阈值：
# 构建一个矩阵 thresholds，其中 thresholds[i][j] 表示簇 i 和簇 j 之间的阈值。
# 对于每对簇 (C1, C2)，使用公式 d = 0.002233 * average_area + 173.80 计算阈值 d。

# 合并簇阶段：
# 对于每对簇 (C1, C2)，遍历所有点对 (p1, p2)，计算 distances[p1][p2] - thresholds[p1][p2]。
# 如果存在任意一个点对 (p1, p2) 满足 distances[p1][p2] - thresholds[p1][p2] <= 0，则合并簇 C1 和簇 C2：

# 更新新的簇的点和面积。

# 重新计算更新后的簇与其他簇之间的距离和阈值。
# 删除已合并的簇，并更新 distances 和 thresholds 矩阵。
# 循环合并阶段：
# 重复步骤 ，直到所有 distances[p1][p2] - thresholds[p1][p2] > 0，即没有更多的簇可以合并。
# 输出阶段：
# 最终的聚类结果即为所有边界框的聚类集合。