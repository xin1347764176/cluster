# Sheet1_摆台照片1_09.jpg 对应json中  images/Sheet1_\u6446\u53f0\u7167\u72471_09.jpg    images

import json
import os
import shutil
from sklearn.model_selection import train_test_split

# 路径设置
json_path = '/svap_intern/allenzzeng/dataset/tiandiyihao/labelme/货架/huojia_coco.json'
image_dir = '/svap_intern/allenzzeng/dataset/tiandiyihao'
train_json_dir = '/svap_intern/allenzzeng/dataset/tiandiyihao/labelme/货架/huojia90'
test_json_dir = '/svap_intern/allenzzeng/dataset/tiandiyihao/labelme/货架/huojia10'
train_json_path = '/svap_intern/allenzzeng/dataset/tiandiyihao/labelme/货架/huojia90.json'
test_json_path = '/svap_intern/allenzzeng/dataset/tiandiyihao/labelme/货架/huojia10.json'

# 读取JSON文件
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 获取所有图像信息和对应的标注
images = data['images']
annotations = data['annotations']

# 将图像ID分为训练集和测试集
image_ids = [image['id'] for image in images]
train_ids, test_ids = train_test_split(image_ids, test_size=0.1, random_state=42)

# 按照ID分配图像和标注
train_images = [img for img in images if img['id'] in train_ids]
test_images = [img for img in images if img['id'] in test_ids]
train_annotations = [ann for ann in annotations if ann['image_id'] in train_ids]
test_annotations = [ann for ann in annotations if ann['image_id'] in test_ids]

# 创建输出字典
train_data = {'images': train_images, 'annotations': train_annotations, 'categories': data['categories']}
test_data = {'images': test_images, 'annotations': test_annotations, 'categories': data['categories']}

# 保存JSON文件
os.makedirs(train_json_dir, exist_ok=True)
os.makedirs(test_json_dir, exist_ok=True)

with open(train_json_path, 'w') as file:
    json.dump(train_data, file)

with open(test_json_path, 'w') as file:
    json.dump(test_data, file)

# 复制图像文件到相应的目录
def copy_images(images, dest_dir):
    for img in images:
        src_path = os.path.join(image_dir, img['file_name'])
        dst_path = os.path.join(dest_dir, img['file_name'])
        # 确保目标文件的目录存在
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(src_path, dst_path)

copy_images(train_images, train_json_dir)
copy_images(test_images, test_json_dir)

print("数据划分完成。")
