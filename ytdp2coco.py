import os
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

def tilabel_shapes2coco(shapes):
    # shape_format = {"frame_type":"rect",
    #             "pts":[{"x":int(b[0]),"y":int(b[1])},
    #                 {"x":int(b[2]),"y":int(b[1])},
    #                 {"x":int(b[2]),"y":int(b[3])},
    #                 {"x":int(b[0]),"y":int(b[3])}
    #                 ],
    #             "tag":{"分类":{"label_value_id":str(c[0]),
    #                         "label_value_name":'_'.join(n[0].split(' '))}}
    #             }
    lbls = []
    for s in shapes:
        assert s['frame_type'] in ('rect','polyline'),'Only support rect and polyline'
        xs = [int(p['x']) for p in s['pts']]
        ys = [int(p['y']) for p in s['pts']]
        x1,y1,x2,y2 = min(xs),min(ys),max(xs),max(ys)
        if 'tag' not in s:
            print('Without tag:',s)
            cls,name = '-1','unknown'
            lbls.append([x1,y1,x2-x1,y2-y1,cls,name])
        else:
            if isinstance(s['tag']['商品陈列'],list):
                for clsinfo in s['tag']['商品陈列']:
                    lbls.append([x1,y1,x2-x1,y2-y1,clsinfo['label_value_id'],clsinfo['label_value_name']])
            elif isinstance(s['tag']['商品陈列'],dict):
                cls,name = s['tag']['商品陈列']['label_value_id'],s['tag']['商品陈列']['label_value_name']
                lbls.append([x1,y1,x2-x1,y2-y1,cls,name])
    return np.array(lbls)

def YTDPlabel2COCO(ytdp_json,coco_json):
    coco_output = {
        "info": {
            "description": "YTDPlabel to COCO",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "date_created": ""
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    with open(ytdp_json,'r') as f:
        lines = [l.strip() for l in f.readlines()]

    img_dir = os.path.dirname(ytdp_json)

    category_set = {}
    category_item_id = 1
    image_id = 1
    annotation_id = 1
    for line in tqdm(lines):
        unneeded_flag = False
        #label = ast.literal_eval(line)
        label = json.loads(line)
        tags = label['tags']
        if tags['annotated'] == False:
            continue
        img_name = os.path.basename(label['info']['path'])

        image_path = os.path.join(img_dir,img_name)
        if not os.path.exists(image_path):
            print(f'{image_path} not exists')
            continue

        image = Image.open(image_path)
        width, height = image.size

        image_info = {
            "id": image_id,
            "file_name": os.path.join('images',img_name),
            "width": width,
            "height": height
        }
        coco_output['images'].append(image_info)
        
        shapes = tags['shapes']
        labels = tilabel_shapes2coco(shapes)
        bbox,cls,name = np.split(labels,[4,5],1)
        
        bbox=bbox.astype(np.float64)
        #cls = cls.astype(np.int32)
        cls = cls.reshape(-1).tolist()
        name = name.reshape(-1).tolist()
        
        for bb,cl,na in zip(bbox,cls,name):
            if cl not in category_set:
                category_set[cl] = category_item_id
                category_info = {
                    "supercategory": None,
                    "id": category_item_id,
                    "name": na
                }
                coco_output['categories'].append(category_info)
                category_item_id += 1
            
            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_set[cl],
                "area": bb[2] * bb[3],
                "bbox": bb.tolist(),
                "iscrowd": 0
            }
            coco_output['annotations'].append(annotation_info)
            annotation_id += 1
        image_id += 1

    print(f'Save coco json on {coco_json}')
    with open(coco_json, 'w',encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=True, indent=4)


if __name__ == '__main__':
    ydtp_json = './labels_ytdp.json'
    coco_json = './labels_coco.json'
    YTDPlabel2COCO(ydtp_json,coco_json)
