import os
import json
import pandas as pd
import csv

extracts_file = "detections_cropped_with_text_stats_filled_extracts.json"
ids_to_subjects_file = "imageid_to_subjectid.json"
output_file = "final_extracts.csv"

assert os.path.isfile(extracts_file)
with open(extracts_file, 'r') as openfile:
    extracts = json.load(openfile)

assert os.path.isfile(ids_to_subjects_file)
with open(ids_to_subjects_file, 'r') as openfile:
    ids_to_subjects = json.load(openfile)

final_result = {
    "extractor_key":[],
    "subject_id": [],
    "data":[]
}

for k,v in extracts.items():
    assert k in ids_to_subjects
    subject_id, ids_file_name = ids_to_subjects[k]["subject_id"], ids_to_subjects[k]["file_name"]
    assert ids_file_name == extracts[k]["frame0"]["file_name"]
    final_result["extractor_key"].append("alice")
    final_result["subject_id"].append(subject_id)
    final_result["data"].append(json.dumps(extracts[k]))

pd.DataFrame.from_dict(final_result).to_csv(output_file,index=False)