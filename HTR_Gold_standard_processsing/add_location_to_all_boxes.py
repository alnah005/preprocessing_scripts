# -*- coding: utf-8 -*-
"""
file: add_location_to_all_boxes.py

@author: Suhail.Alnahari

@description: This file generates all_boxes_with_locations.csv

@created: 2021-05-24T11:29:35.906Z-05:00

@last-modified: 2021-05-24T16:01:51.235Z-05:00
"""

import pandas as pd

all_boxes = pd.read_csv("all_boxes.csv")
subjects_to_names = pd.read_csv("subject_to_name.csv")

all_boxes["id"] = all_boxes["subjectId_frame_classification"].apply(lambda x: '_'.join(x.split("_")[:-1]))

merged = pd.merge(all_boxes, subjects_to_names, on="id", how="left")

merged = merged.drop(['id'], axis=1)

merged = merged.rename(columns={"name":"original_image_path"})

merged[~merged["original_image_path"].isna()].to_csv("all_boxes_with_locations.csv",index=False)