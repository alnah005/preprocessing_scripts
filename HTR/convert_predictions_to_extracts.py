import os
import json
import numpy as np
import copy

prediction_json_path = "detections_cropped_with_text_stats_filled.json"
output_file_extracts = "detections_cropped_with_text_stats_filled_extracts.json"
enable_graphs = False
number_of_frames = 1
path_delimiter = "/"
assert number_of_frames == 1, "number of frames in each subject must be one for now"
if enable_graphs:
    import matplotlib.pyplot as plt
assert os.path.isfile(prediction_json_path)
with open(prediction_json_path, 'r') as openfile:
    detections = json.load(openfile)

extract_template = {
    "frame0": {"text": [], "slope": [], "points": {"x": [], "y": []}, "gold_standard": False,"file_name":None},
    # "aggregation_version": "3.6.0"
}

result = {i["im_id"]:copy.deepcopy(extract_template) for i in detections}

for det in detections:
    result[det["im_id"]]["frame0"]["text"].append([" ".join([t for t in det["text"].split(' ') if len(t) > 0])])
    result[det["im_id"]]["frame0"]["slope"].append(det["slope_deg"])
    result[det["im_id"]]["frame0"]["points"]["x"].append(det["x"])
    result[det["im_id"]]["frame0"]["points"]["y"].append(det["y"])
    cropped_file_name = det['im_path'].split(path_delimiter)[-1]
    format = cropped_file_name.split('.')[-1]
    parent_file_name = '_'.join(cropped_file_name.split('_')[:-1])+'.'+format
    result[det["im_id"]]["frame0"]["file_name"] =  parent_file_name
    assert result[det["im_id"]]["frame0"]["file_name"] is not None and len(result[det["im_id"]]["frame0"]["file_name"]) > 0 and '.' in result[det["im_id"]]["frame0"]["file_name"]
    

with open(output_file_extracts, "w") as outfile:
    json.dump(result,outfile,indent=3)