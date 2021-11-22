# -*- coding: utf-8 -*-
"""
file: split_train_test_per_json.py

@author: Suhail.Alnahari

@description: This file moves actual images based on the json splits

@created: 2021-05-28T10:00:54.493Z-05:00

@last-modified: 2021-05-28T10:16:13.328Z-05:00
"""

# standard library
import os
from typing import Dict, List, Any
# 3rd party packages
import json
import numpy as np
import shutil
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

try:
    shutil.rmtree(dataset_path+ImageDir[:-1]+"_val")
    print("overwriting existing output dir")
except:
    print("first time creating dir")

try:
    shutil.rmtree(dataset_path+ImageDir[:-1]+"_train")
    print("overwriting existing output dir")
except:
    print("first time creating dir")

try:
    shutil.rmtree(dataset_path+ImageDir[:-1]+"_test")
    print("overwriting existing output dir")
except:
    print("first time creating dir")


os.mkdir(dataset_path+ImageDir[:-1]+"_val")
assert os.path.exists(dataset_path+ImageDir[:-1]+"_val")

os.mkdir(dataset_path+ImageDir[:-1]+"_train")
assert os.path.exists(dataset_path+ImageDir[:-1]+"_train")

os.mkdir(dataset_path+ImageDir[:-1]+"_test")
assert os.path.exists(dataset_path+ImageDir[:-1]+"_test")

with open(box_labels_parent_path+labels_val) as json_file:
    angular_labels_dict = json.load(json_file)

print(f"moving {len(angular_labels_dict['images'])} images to validation dir")

for i in angular_labels_dict['images']:
    try:
        shutil.copy(dataset_path+ImageDir +
                    i['file_name'], dataset_path+ImageDir[:-1]+"_val/"+i['file_name'])
    except:
        print("error",dataset_path ,i['file_name'])

with open(box_labels_parent_path+labels_train) as json_file:
    angular_labels_dict = json.load(json_file)

print(f"moving {len(angular_labels_dict['images'])} images to train dir")

for i in angular_labels_dict['images']:
    try:
        shutil.copy(dataset_path+ImageDir +
                    i['file_name'], dataset_path+ImageDir[:-1]+"_train/"+i['file_name'])
    except:
        print("error", i['file_name'])


with open(box_labels_parent_path+labels_test) as json_file:
    angular_labels_dict = json.load(json_file)

print(f"moving {len(angular_labels_dict['images'])} images to test dir")

for i in angular_labels_dict['images']:
    try:
        shutil.copy(dataset_path+ImageDir +
                    i['file_name'], dataset_path+ImageDir[:-1]+"_test/"+i['file_name'])
    except:
        print("error", i['file_name'])