import os
import json
import io


ytdp_label_json = "../origin/ytdp_json/垃圾满溢_2584774720_2788054664_teamlabel_ti.json"
image_root = "../origin/images"
out_labelme_json_root = "../origin/labelme_json"


def to_json_format(fin):
    formated_string = ""
    formated_string += "["
    lines = fin.readlines()
    for line in lines:
        if len(line) > 1:
            line = line.strip() + ",\n"
            formated_string += line
    formated_string = formated_string.strip(",\n")
    formated_string += "]"

    return io.StringIO(formated_string)


with open(ytdp_label_json) as fin:
    fin = to_json_format(fin)
    ytdp_label = json.load(fin)

for image_label in ytdp_label:
    labelme_json = {
        "version": "4.6.0",
        "flags": {},
        "shapes": [],
        "imageData": None,
    }
    image_path = os.path.join(image_root, image_label["info"]["path"])
    image_name = os.path.basename(image_path)
    image_h, image_w = image_label["info"]["image_size"]
    labelme_json["imagePath"] = image_name
    labelme_json["imageHeight"] = image_h
    labelme_json["imageWidth"] = image_w

    if image_label["tags"]["annotated"]:
        for ytdp_shape in image_label["tags"]["shapes"]:
            pts4 = ytdp_shape["pts"]
            x1 = min([p["x"] for p in pts4])
            y1 = min([p["y"] for p in pts4])
            x2 = max([p["x"] for p in pts4])
            y2 = max([p["y"] for p in pts4])
            tag_root = list(ytdp_shape["tag"].keys())[0]
            labelme_shape = {
                "label": ytdp_shape["tag"][tag_root]["label_value_id"],
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            labelme_json["shapes"].append(labelme_shape)

    out_labelme_json_fp = image_path.replace(image_root, out_labelme_json_root)
    ext = os.path.splitext(out_labelme_json_fp)[1]
    out_labelme_json_fp = out_labelme_json_fp.replace(ext, '.json')
    print(out_labelme_json_fp)
    os.makedirs(os.path.dirname(out_labelme_json_fp), exist_ok=True)
    with open(out_labelme_json_fp, "w") as fout:
        json.dump(labelme_json, fout)
