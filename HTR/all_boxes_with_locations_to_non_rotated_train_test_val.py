import pandas as pd
from dataclasses import dataclass, asdict
import os
import json


cwd = os.getcwd()


@dataclass
class Corners:
    x_0: float
    y_0: float
    x_1: float
    y_1: float
    x_2: float
    y_2: float
    x_3: float
    y_3: float
    height: float
    text: str
    subjectId_frame_classification: str
    original_image_path: str

boxes = pd.read_csv(cwd+"/all_boxes_with_locations.csv")

partition_files = {
    'annotations_0':  cwd+"/final_annotations_10_test.json",
    'annotations_1': cwd+"/final_annotations_10_train.json",
    'annotations_2': cwd+"/final_annotations_10_val.json",
}
for i in partition_files.values():
    assert os.path.exists(i)

annotations = {}
for k in partition_files.keys():
    if 'annotations_' in k[:12]:
        assert '.json' in partition_files[k][-5:]
        assert not('.json' in partition_files[k][:-5])
        with open(partition_files[k]) as json_file:
            annotations[k] = json.load(json_file)


image_name_to_boxes = {}
for index, row in boxes.iterrows():
    image_name = row['original_image_path'].split('/')[-1]
    corners = Corners(
        x_0=row['x_0'],y_0=row['y_0'],
        x_1=row['x_1'],y_1=row['y_1'],
        x_2=row['x_2'],y_2=row['y_2'],
        x_3=row['x_3'],y_3=row['y_3'],
        height=row['height'], text=row['text'],
        subjectId_frame_classification= row['subjectId_frame_classification'],
        original_image_path=row['original_image_path'],
    )
    if image_name_to_boxes.get(image_name,None) is not None:
        image_name_to_boxes[image_name].append(corners)
    else:
        image_name_to_boxes[image_name] = [corners]


partitions = {k:[] for k in annotations.keys()}
for k in annotations.keys():
    images = [i['file_name'] for i in annotations[k]['images']]
    for image in images:
        assert image in image_name_to_boxes
        partitions[k].extend(image_name_to_boxes[image])
        del image_name_to_boxes[image]

print({len(v):len(annotations[k]['annotations']) for k,v in partitions.items()})


for k in annotations.keys():
    new_path = partition_files[k].split('.json')[0]
    new_path += '_non_rotated.json'
    with open(new_path, "w") as outfile:    
        list_of_dicts = [asdict(corners) for corners in partitions[k]]
        json.dump(list_of_dicts, outfile,indent=1)