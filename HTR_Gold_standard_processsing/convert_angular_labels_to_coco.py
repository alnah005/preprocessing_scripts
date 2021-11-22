# -*- coding: utf-8 -*-
"""
file: convert_angular_labels_to_coco.py

@author: Suhail.Alnahari

@description: This file generates the split up jsons from angular_labels_<degree>.csv

@created: 2021-05-25T09:27:28.391Z-05:00

@last-modified: 2021-05-25T10:27:23.182Z-05:00
"""

# standard library
import os
from typing import Dict, List, Any
# 3rd party packages
import json
import numpy as np
# local source


ImageDir = "Original_Images_rotated_10/"
dataset = "/ASM/"
labels = "angular_labels_10.csv"
validation_percentage = 0.0
test_percentage = 0.0
try:
    root =  os.getcwd()
    assert os.path.exists(root+"/data/")
    assert os.path.exists(root+"/data"+dataset)
    dataset_path = root+"/data"+dataset
    assert os.path.isfile(root+"/../../aggregation_for_caesar/htr/"+labels)
    box_labels_path = root+"/../../aggregation_for_caesar/htr/"+labels
    box_labels_parent_path = root+"/../../aggregation_for_caesar/htr/"
except:
    root = os.getcwd()+"/../../text_recognition"
    assert os.path.exists(root+"/data/")
    assert os.path.exists(root+"/data"+dataset)
    dataset_path = root+"/data"+dataset
    assert os.path.isfile(os.getcwd()+"/"+labels)
    box_labels_path = os.getcwd()+"/"+labels
    box_labels_parent_path = os.getcwd()+"/"

def get_row_with_text_containing_commas(columnNames,row_split,text_columns):
    assert len(text_columns) == 1
    assignment = {}
    # get columns from the left of the text column
    for i in range(text_columns[0]):
        assignment[columnNames[i]] = row_split[i]

    # get columns from the right of the text columns
    for i in range(len(columnNames) - text_columns[0] - 1):
        assignment[columnNames[-1*(i+1)]] = row_split[-1*(i+1)]

    # get text column
    assignment[columnNames[text_columns[0]]] = ','.join(row_split[text_columns[0]:len(row_split)-(len(columnNames)-text_columns[0]-1)])
    return assignment

f = open(box_labels_path, 'r')
final_labels_train: Dict[str, List[Dict[str, Any]]] = {
    "images": [], "annotations": [], "categories": []}
final_labels_val: Dict[str, List[Dict[str, Any]]] = {
    "images": [], "annotations": [], "categories": []}
final_labels_test: Dict[str, List[Dict[str, Any]]] = {
    "images": [], "annotations": [], "categories": []}
images_dict: Dict[str, Dict[str, Any]] = {}
annotations_id = 0
image_id = 0
columnNames = ['img_name', 'x', 'y', 'w', 'h', 'a_radian']
columnNames +=  ['x1_orig', 'y1_orig', 'x2_orig', 'y2_orig','x3_orig', 'y3_orig', 'x4_orig', 'y4_orig']
columnNames += ['text_orig']
columnNames += ['angle_transform']
text_columns = sorted([columnNames.index('text_orig')])
for row in f:
    try:
        row_split = row.replace('\n', '').split(',')
        useful_cols = get_row_with_text_containing_commas(columnNames,row_split,text_columns)
        assert os.path.isfile(dataset_path+ImageDir+useful_cols['img_name'])
        isVal = np.random.uniform() < validation_percentage+test_percentage
        isTest = np.random.uniform() < (test_percentage*1.0/(validation_percentage+test_percentage))
        if images_dict.get(useful_cols['img_name']) is None:
            images_dict[useful_cols['img_name']] = {
                'id': image_id, "isVal": isVal, "isTest": isTest}
            image_id += 1
        xmin = float(useful_cols['x'])
        ymin = float(useful_cols['y'])
        w = float(useful_cols['w'])
        h = float(useful_cols['h'])
        theta = float(useful_cols['a_radian'])
        centre = np.array([xmin + w / 2.0, ymin + h / 2.0])
        original_points = np.array([[xmin, ymin],                # This would be the box if theta = 0
                                    [xmin + w, ymin],
                                    [xmin + w, ymin + h],
                                    [xmin, ymin + h]])
        rotation = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
        corners = np.matmul(original_points - centre, rotation) + centre
        annot = {
            "id": annotations_id,
            "image_id": images_dict[useful_cols['img_name']]['id'],
            "category_id": 1,
            # all floats, where theta is measured in radians anti-clockwise from the x-axis.
            "bbox": [xmin, ymin, w, h, theta],
            # Required for validation scores.
            "segmentation": [[corners[0][0], corners[0][1], corners[1][0], corners[1][1], corners[2][0], corners[2][1], corners[3][0], corners[3][1]]],
            "area": w*h,  # w * h. Required for validation scores
            "iscrowd": 0,  # Required for validation scores
            "text": useful_cols['text_orig']
        }
        if images_dict[useful_cols['img_name']]['isVal']:
            if images_dict[useful_cols['img_name']]['isTest']:
                final_labels_test["annotations"].append(annot)
            else:
                final_labels_val["annotations"].append(annot)
        else:
            final_labels_train["annotations"].append(annot)
        annotations_id += 1
    except:
        print("Error encountered", row)

for i in images_dict:
    image_dict = {
        "id": images_dict[i]['id'],
        "file_name": i
    }
    if images_dict[i]['isVal']:
        if images_dict[i]['isTest']:
            final_labels_test["images"].append(image_dict)
        else:
            final_labels_val["images"].append(image_dict)
    else:
        final_labels_train["images"].append(image_dict)

categs = {
    "id": 1,
    "name": "textline"
}

final_labels_val["categories"].append(categs)
final_labels_train["categories"].append(categs)
final_labels_test["categories"].append(categs)

with open(box_labels_parent_path+'final_annotations_10_train.json', 'w') as fp:
    json.dump(final_labels_train, fp, indent=4)

with open(box_labels_parent_path+'final_annotations_10_val.json', 'w') as fp:
    json.dump(final_labels_val, fp, indent=4)

with open(box_labels_parent_path+'final_annotations_10_test.json', 'w') as fp:
    json.dump(final_labels_test, fp, indent=4)

num_images_train, num_annotation_train = len(
    final_labels_train["images"]), len(final_labels_train["annotations"])
num_images_val, num_annotation_val = len(
    final_labels_val["images"]), len(final_labels_val["annotations"])
num_images_test, num_annotation_test = len(
    final_labels_test["images"]), len(final_labels_test["annotations"])
print(
    f"Num of Images,Annotations for train: {num_images_train},{num_annotation_train}\nNum of Images,Annotations for val: {num_images_val},{num_annotation_val}\nNum of Images,Annotations for test: {num_images_test},{num_annotation_test}")
