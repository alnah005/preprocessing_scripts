# -*- coding: utf-8 -*-
"""
file: add_image_height_width_to_json.py

@author: Suhail.Alnahari

@description: This file doesn't generate any files but modifies the split up jsons

@created: 2021-05-28T15:47:41.812Z-05:00

@last-modified: 2021-05-28T15:58:18.041Z-05:00
"""

# standard library
import os
from typing import Dict, List, Any
# 3rd party packages
import json
import numpy as np
import shutil
import cv2
# local source


ImageDir = "Original_Images_rotated_10/"
dataset = "/ASM/"
labels_val = "final_annotations_10_val.json"
labels_train = "final_annotations_10_train.json"
labels_test = "final_annotations_10_test.json"
try:
    root =  os.getcwd()
    assert os.path.exists(root+"/data/")
    assert os.path.exists(root+"/data"+dataset)
    assert os.path.isfile(root+"/../../aggregation_for_caesar/htr/"+labels_val)
    assert os.path.isfile(root+"/../../aggregation_for_caesar/htr/"+labels_train)
    assert os.path.isfile(root+"/../../aggregation_for_caesar/htr/"+labels_test)
    box_labels_parent_path = root+"/../../aggregation_for_caesar/htr/"
except:
    root = os.getcwd()+"/../../text_recognition"
    assert os.path.exists(root+"/data/")
    assert os.path.exists(root+"/data"+dataset)
    assert os.path.isfile(os.getcwd()+"/"+labels_val)
    assert os.path.isfile(os.getcwd()+"/"+labels_train)
    assert os.path.isfile(os.getcwd()+"/"+labels_test)
    box_labels_parent_path = os.getcwd()+"/"

dataset_path = root+"/data"+dataset

with open(box_labels_parent_path+labels_val) as json_file:
    final_labels = json.load(json_file)

for i in range(len(final_labels["images"])):
    im = cv2.imread(dataset_path+ImageDir +
                    final_labels["images"][i]["file_name"])
    assert len(im.shape) > 1
    if len(im.shape) == 3:
        height, width, _ = im.shape
    elif len(im.shape) == 2:
        height, width = im.shape
    else:
        print("error at", dataset_path+ImageDir +
              final_labels["images"][i]["file_name"])
        continue
    final_labels["images"][i]["height"] = int(height)
    final_labels["images"][i]["width"] = int(width)

with open(box_labels_parent_path+labels_val, 'w') as fp:
    json.dump(final_labels, fp, indent=4)

with open(box_labels_parent_path+labels_train) as json_file:
    final_labels = json.load(json_file)

for i in range(len(final_labels["images"])):
    im = cv2.imread(dataset_path+ImageDir +
                    final_labels["images"][i]["file_name"])
    assert len(im.shape) > 1
    if len(im.shape) == 3:
        height, width, _ = im.shape
    elif len(im.shape) == 2:
        height, width = im.shape
    else:
        print("error at", dataset_path+ImageDir +
              final_labels["images"][i]["file_name"])
        continue
    final_labels["images"][i]["height"] = int(height)
    final_labels["images"][i]["width"] = int(width)

with open(box_labels_parent_path+labels_train, 'w') as fp:
    json.dump(final_labels, fp, indent=4)

with open(box_labels_parent_path+labels_test) as json_file:
    final_labels = json.load(json_file)

for i in range(len(final_labels["images"])):
    im = cv2.imread(dataset_path+ImageDir +
                    final_labels["images"][i]["file_name"])
    assert len(im.shape) > 1
    if len(im.shape) == 3:
        height, width, _ = im.shape
    elif len(im.shape) == 2:
        height, width = im.shape
    else:
        print("error at", dataset_path+ImageDir +
              final_labels["images"][i]["file_name"])
        continue
    final_labels["images"][i]["height"] = int(height)
    final_labels["images"][i]["width"] = int(width)

with open(box_labels_parent_path+labels_test, 'w') as fp:
    json.dump(final_labels, fp, indent=4)
