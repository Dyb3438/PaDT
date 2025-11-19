import os
import re
import numpy as np
from PIL import Image
import random
from pycocotools import mask
from pycocotools.coco import COCO
from refer import REFER
import matplotlib.pyplot as plt
from copy import deepcopy
import cv2
import json

os.makedirs('../../dataset/llava_mix665k/processed', exist_ok=True)
data_path = "../../dataset/llava_mix665k/"
output_json = '../../dataset/llava_mix665k/processed/llava_v1_5_mix665k.json'



bbox_pattern = r'(\[([01]\.\d+),\s*([01]\.\d+),\s*([01]\.\d+),\s*([01]\.\d+)\])'
bbox_pattern_split = r'\[[01]\.\d+,\s*[01]\.\d+,\s*[01]\.\d+,\s*[01]\.\d+\]'

with open(output_json, 'w') as f:
    f.write("")

with open(os.path.join(data_path, 'sampled_data.json'), 'r') as f:
    data = json.loads(f.read())

    for sample in data:
        sample_pass = False

        item = {
            "id": sample['id'],
            "image": sample['image'],
            "task": "llava_mix665k",
            "answer_template": None,
            "conversations": [],
            "objects": []
        }

        image = Image.open(os.path.join(data_path, sample['image']))
        ori_w, ori_h = image.size

        object_idx = 0

        for conv in sample['conversations']:
            conv['value'] = conv['value'].replace("<image>\n", "").replace("\n<image>", "")
            box_split_parts = re.split(bbox_pattern_split, conv['value'])
            box_strs = re.findall(bbox_pattern, conv['value'])

            new_value = box_split_parts[0]
            for box_str, box_split_part in zip(box_strs, box_split_parts[1:]):
                x1_n, y1_n, x2_n, y2_n = float(box_str[1]), float(box_str[2]), float(box_str[3]), float(box_str[4])
                x1_n, y1_n, x2_n, y2_n = max(0, x1_n), max(0, y1_n), min(x2_n, 1), min(y2_n, 1)

                mask = np.zeros((ori_h, ori_w)).astype(np.uint8)
                x1, y1, x2, y2 = round(x1_n * ori_w), round(y1_n * ori_h), round(x2_n * ori_w), round(y2_n * ori_h)
                mask[y1: y2, x1: x2] = 1

                resized_h, resized_w = int(round(ori_h / 28) * 28), int(round(ori_w / 28) * 28)
                resized_mask = cv2.resize(mask * 255, (resized_w, resized_h))
                patch_mask = resized_mask.reshape(resized_h // 28, 28, resized_w//28, 28).transpose(0, 2, 1, 3).mean(axis=-1).mean(axis=-1) >= 255 / 28

                if patch_mask.sum() < 1:
                    sample_pass = True
                    break

                item['objects'].append(
                    {
                        'patches': np.where(patch_mask.reshape(-1))[0].tolist(),
                        'bbox': [x1_n, y1_n, x2_n, y2_n],
                        'iscrowd': 1,
                        'area': (x2 - x1) * (y2 - y1),
                        'label': 'Obj_%d' % object_idx,
                        'selecting_stategy': 'border' if conv['from'] != 'gpt' else 'random'
                    }
                )

                new_value += '<|Obj_%d|>' % object_idx + box_split_part
                object_idx += 1            
            
            if sample_pass:
                break
            item['conversations'].append({
                'from': conv['from'],
                'value': new_value
            })
        
        if sample_pass is False:
            with open(output_json, 'a+') as f:
                f.write(json.dumps(item) + '\n')
