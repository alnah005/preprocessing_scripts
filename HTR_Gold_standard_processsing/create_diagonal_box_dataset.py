# -*- coding: utf-8 -*-
"""
file: create_diagonal_box_dataset.py

@author: Suhail.Alnahari

@description: This file generates angular_labels_<degree>.csv and moves images from the locations specified by all_boxes_with_locations.csv into outputFolder

@created: 2021-05-24T11:29:35.906Z-05:00

@last-modified: 2021-05-24T16:01:51.235Z-05:00
"""

# standard library
import os
from ast import literal_eval as make_tuple
from typing import Dict
# 3rd party packages
import pandas as pd
# local source
try:
    from preprocess.diagonal_box import get_rotated_sample
except:
    import diagonal_box
    get_rotated_sample = diagonal_box.get_rotated_sample

dataset = "/ASM/"
labels = "all_boxes_with_locations.csv"
outputFolder = "Original_Images_rotated_10/"
chunksize = 10 ** 6
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

outputFolderPath = dataset_path + outputFolder

try:
    shutil.rmtree(outputFolderPath)
    print("overwriting existing output dir")
except:
    print("first time creating dir")

os.mkdir(outputFolderPath)
assert os.path.exists(outputFolderPath)

f = open(box_labels_parent_path+"angular_labels_10.csv", 'w')
old_points_label = ['x1_orig', 'y1_orig', 'x2_orig', 'y2_orig','x3_orig', 'y3_orig', 'x4_orig', 'y4_orig']
passing_labels = ['text_orig']
f.write(','.join(['img_name', 'x', 'y', 'w', 'h', 'a_radian']+old_points_label+passing_labels+['angle_transform'])+'\n')
multiplier = 1
angles: Dict[str, int] = {}
hits = 0
reruns = 0
misses = 0
for chunk in pd.read_csv(box_labels_path, chunksize=chunksize):
    chunk['image_id_path'] = chunk['original_image_path'].apply(
        lambda loc: loc.split('/')[-1])

    for index, row in chunk.iterrows():
        fn = row['original_image_path']
        if fn in angles.keys():
            angle_rotate = angles[fn]
        else:
            angle_rotate = (index % 10)*multiplier
            multiplier *= -1
            angles[fn] = angle_rotate

        coords = {'x1': row['x_0'], 'y1': row['y_0'], 'x2': row['x_1'], 'y2': row['y_1'],
                  'x3': row['x_2'], 'y3': row['y_2'], 'x4': row['x_3'], 'y4': row['y_3']}
        for k,v in coords.items():
            coords[k] = round(v)
        try:
            tries = 0
            rotated_labels = [-1, -1]
            while (rotated_labels[0] <= 0 or rotated_labels[1] <= 0 or rotated_labels[2] < 0 or rotated_labels[3] < 0) and tries < 3:
                img, rotated_labels = get_rotated_sample(
                    coords, image_path=fn, angle=angle_rotate)
                tries += 1
            assert rotated_labels[0] >= 0 and rotated_labels[1] >= 0 and rotated_labels[2] > 0 and rotated_labels[3] > 0
            if not os.path.exists(outputFolderPath+row['image_id_path']):
                img.save(outputFolderPath+row['image_id_path'])
            f.write(
                ','.join([row['image_id_path']]+[str(j) for j in rotated_labels]+[str(coords[k[:-5]]) for k in old_points_label]+[str(row[k[:-5]]) for k in passing_labels]+[str(angle_rotate)])+'\n')
            hits += 1
        except:
            misses += 1
            co = [coords['x1'], coords['y1'], coords['x2'], coords['y2'], coords['x3'],
                  coords['y3'], coords['x4'], coords['y4']]
            # errors.write(','.join(
            #     [fn]+[str(c) for c in co]+[str(angle_rotate)]+[str(k) for k in line_box])+'\n')
            print(','.join(
                [fn]+[str(c) for c in co]+[str(angle_rotate)]+[str(coords[k[:-5]]) for k in old_points_label]+[str(row[k[:-5]]) for k in passing_labels]))
f.close()
# errors.close()


print(
    f"number of converted with no problems = {hits}\nnumber of converted with some problems {reruns}\nnumber of errors {misses}")
