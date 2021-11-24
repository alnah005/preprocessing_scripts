import pandas as pd
from collections import defaultdict
import os
from typing import Dict, Any, List
from extracts_to_cluster_dataclasses import (
    Extractable,
    ExtractRotatedRectangle,
    ExtractPoint,
    PointClustersWithMembers,
    RotatedRectangleClusters,
    PointClusters,
    Cluster,
)
import json
import copy
from dataclasses import asdict
import numpy as np

extracts = {
    "point_clusters": "point_reducer_hdbscan_beta2_proposed.csv",
}

extracts_dataclasses: Dict[str, Extractable] = {
    "point_clusters": PointClustersWithMembers,
}

suffix = "space_clustering"

verbose = True
graphics = True

if graphics:
    import matplotlib.pyplot as plt

for k, v in extracts.items():
    assert os.path.isfile(v)
    assert k in extracts_dataclasses

extract_files: Dict[str, pd.DataFrame] = {
    k: pd.read_csv(v) for k, v in extracts.items()
}

for v in extract_files.values():
    assert "task" in v.columns
    assert len([i for i in v.columns if "data.frame" in i]) > 0

extractObjects: Dict[int, Extractable] = {}
element = 0
num_empty_labels = 0
for extract_k, v in extract_files.items():
    available_frame_columns = [i for i in v.columns if "data.frame" in i]
    tools = list(
        set([i.split("tool")[1].split("_")[0] for i in available_frame_columns])
    )
    unrelated_frame_columns = [i for i in v.columns if "data.frame" not in i]
    frames = list(
        set(
            [
                int(col.split("data.frame")[1].split(".")[0])
                for col in available_frame_columns
            ]
        )
    )
    for f in frames:
        frame_string = f"data.frame{f}."
        frame_columns = [c for c in available_frame_columns if frame_string in c]
        remaining_info = []
        for k in frame_columns:
            assert len(k.split(frame_string)) > 1
            field_split = k.split(frame_string)[1].split("_")
            if len(field_split) > 3:
                field_split[2] = "_".join(field_split[2:])
                field_split = field_split[:3]
            remaining_info.append(field_split)
        tasks: Dict[str, Dict[str, Any]] = {}
        for r in remaining_info:
            task, tool, field = r
            if task not in tasks:
                tasks[task] = {}
            if tool not in tasks[task]:
                tasks[task][tool] = []
            tasks[task][tool].append(field)
        for index, row in v.iterrows():
            for task in tasks:
                extract = {k: row[k] for k in unrelated_frame_columns}
                for tool, fieldNames in tasks[task].items():
                    string_fields: Dict[str, List[Any]] = {}
                    for field in fieldNames:
                        row_val = row[f"{frame_string}{task}_{tool}_{field}"]
                        if (
                            isinstance(row_val, str)
                            and field in extracts_dataclasses[extract_k].field_names()
                        ):
                            try:
                                string_fields[field] = json.loads(row_val)
                            except AttributeError:
                                pass
                        if field not in string_fields:
                            extract[field] = row_val
                    extract["frame"] = f
                    extract["tool"] = tool
                    string_field_lens = list(
                        set(
                            [
                                len(v)
                                for k, v in string_fields.items()
                                if k
                                in extracts_dataclasses[
                                    extract_k
                                ].extract_defining_columns()
                            ]
                        )
                    )
                    if len(string_field_lens) != 1:
                        num_empty_labels += 1
                        continue
                    for i in range(string_field_lens[0]):
                        new_extract = copy.deepcopy(extract)
                        for string_field in string_fields:
                            new_extract[string_field] = string_fields[string_field]
                            if (
                                string_field
                                in extracts_dataclasses[
                                    extract_k
                                ].extract_defining_columns()
                            ):
                                new_extract[string_field] = string_fields[string_field][
                                    i
                                ]
                        new_extract["id"] = element
                        new_extract["list_order"] = i
                        try:
                            extractObjects[element] = extracts_dataclasses[
                                extract_k
                            ].from_json(new_extract)
                            element += 1
                        except AssertionError:
                            if verbose:
                                print(
                                    new_extract,
                                    "contains required null fields or json does not contain all fields",
                                )

distances_start = [
    dis
    for obj in extractObjects.values()
    for dis in obj.get_distances()
    if obj.tool == "tool0"
]
print(len(distances_start))
plt.hist(distances_start, bins=100)
plt.xlabel("Distance start members to their clusters")
plt.ylabel("Frequency")
plt.axvline(
    x=np.quantile(distances_start, 0.5),
    label="mean = {}".format(np.quantile(distances_start, 0.5)),
    color="k",
    linestyle="--",
)
plt.axvline(
    x=np.quantile(distances_start, 0.5) + np.std(distances_start),
    label="mean + sigma = {}".format(
        np.quantile(distances_start, 0.5) + np.std(distances_start)
    ),
    color="k",
    linestyle="-",
)
plt.legend()
plt.show()


distances_end = [
    dis
    for obj in extractObjects.values()
    for dis in obj.get_distances()
    if obj.tool != "tool0"
]
print(len(distances_end))
plt.hist(distances_end, bins=100)
plt.xlabel("Distance end members to their clusters")
plt.ylabel("Frequency")
plt.axvline(
    x=np.quantile(distances_end, 0.5),
    label="mean = {}".format(np.quantile(distances_end, 0.5)),
    color="k",
    linestyle="--",
)
plt.axvline(
    x=np.quantile(distances_end, 0.5) + np.std(distances_end),
    label="mean + sigma = {}".format(
        np.quantile(distances_end, 0.5) + np.std(distances_end)
    ),
    color="k",
    linestyle="-",
)
plt.legend()
plt.show()
