import pandas as pd
from collections import defaultdict
import os
from typing import Dict, Any, List
from extracts_to_cluster_dataclasses import (
    Extractable,
    ExtractRotatedRectangle,
    ExtractPoint,
    RotatedRectangleClusters,
    PointClusters,
    Cluster,
)
import json
import copy
from dataclasses import asdict
import numpy as np

extracts = {
    "point": "point_extractor_by_frame_beta2.csv",
    "rectangle": "shape_extractor_rotateRectangle_beta2.csv",
    "rectangle_clusters": "shape_reducer_hdbscan_beta2_proposed.csv",
    "point_clusters": "point_reducer_hdbscan_beta2_proposed.csv",
    # "rectangle_clusters": "shape_reducer_hdbscan_beta2.csv",
    # "point_clusters": "point_reducer_hdbscan_beta2.csv",
}

extracts_dataclasses: Dict[str, Extractable] = {
    "point": ExtractPoint,
    "rectangle": ExtractRotatedRectangle,
    "rectangle_clusters": RotatedRectangleClusters,
    "point_clusters": PointClusters,
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

if verbose:
    print("number of labels that were found but had empty fields:", num_empty_labels)


def getNearestObjects(dic: Dict[int, Extractable]) -> Dict[int, List]:
    obj_to_nearest_objects = {o: [] for o in dic.keys()}
    types = list(set([type(o) for o in dic.values()]))
    types_to_object = [[o for o in dic.values() if type(o) == t] for t in types]
    for i in obj_to_nearest_objects:
        for j in range(len(types)):
            elements_to_compare = types_to_object[j]
            type_elements = []
            for k in range(len(elements_to_compare)):
                distance = dic[i].distance_between_classes(elements_to_compare[k])
                if distance is not None:
                    type_elements.append((elements_to_compare[k].id, distance))
            obj_to_nearest_objects[i].append(type_elements)
    return obj_to_nearest_objects


def seperateObjects(dic: Dict[int, Extractable]) -> Dict:
    obj = {o.id: asdict(o) for o in dic.values()}
    object_to_nearest_objects = getNearestObjects(dic)
    for i in obj:
        obj[i]["type"] = type(dic[i]).__name__
        obj[i]["clusters"] = []
        obj[i]["volunteer_labels"] = []
        for j in object_to_nearest_objects[i]:
            for w in j:
                if w[0] in dic and w[0] != i and w[1] is not None:
                    new_dict = asdict(dic[w[0]])
                    new_dict["type"] = type(dic[w[0]]).__name__
                    new_dict["distance_away"] = w[1]
                    if isinstance(dic[w[0]], Cluster):
                        obj[i]["clusters"].append(new_dict)
                    else:
                        obj[i]["volunteer_labels"].append(new_dict)
    return obj


objects = seperateObjects(extractObjects)
with open("objects_to_closest_" + suffix + ".json", "w") as outfile:
    json.dump(list(objects.values()), outfile, indent=3)


volunteer_freqs = {
    i: [0, []]
    for i in set(
        [
            len(w["volunteer_labels"])
            for w in [k for k in objects.values() if "cluster" in k["type"].lower()]
        ]
    )
}
for i in [k for k in objects.values() if "cluster" in k["type"].lower()]:
    volunteer_freqs[len(i["volunteer_labels"])][0] += 1
    volunteer_freqs[len(i["volunteer_labels"])][1].append(i["id"])
if verbose:
    print(
        "\n\nfrequency count of number of close volunteer labels for each cluster\n",
        {k: v[0] for k, v in volunteer_freqs.items()},
    )
    print(
        "ids of the maximum clusters:", volunteer_freqs[max(volunteer_freqs.keys())][1]
    )
    print(
        "ids of the minimum clusters:", volunteer_freqs[min(volunteer_freqs.keys())][1]
    )


def getClusterFreqs(dic: Dict):
    cluster_freqs = {
        i: [0, []]
        for i in set(
            [
                len(w["clusters"])
                for w in [k for k in dic.values() if "cluster" in k["type"].lower()]
            ]
        )
    }
    for i in [k for k in dic.values() if "cluster" in k["type"].lower()]:
        cluster_freqs[len(i["clusters"])][0] += 1
        cluster_freqs[len(i["clusters"])][1].append(i["id"])
    return cluster_freqs


cluster_freqs = getClusterFreqs(objects)
if verbose:
    print(
        "\nfrequency count of number of close clusters for each cluster\n",
        {k: v[0] for k, v in cluster_freqs.items()},
    )
    print("ids of the maximum clusters:", cluster_freqs[max(cluster_freqs.keys())][1])
    print("ids of the minimum clusters:", cluster_freqs[min(cluster_freqs.keys())][1])


##### start of solar jets specific code
def validate(obs: List):
    allO = [b for a in obs for b in a["clusters"]]
    unique_objects = list({i["id"]: i for i in allO}.values())
    start_points = sorted(
        [
            u
            for u in unique_objects
            if u["type"] != "RotatedRectangleClusters" and u["tool"] == "tool0"
        ],
        key=lambda a: min(
            [
                extractObjects[a["id"]].distance_between_classes(
                    extractObjects[endpoint["id"]]
                )
                for endpoint in [
                    w
                    for w in unique_objects
                    if w["type"] != "RotatedRectangleClusters" and w["tool"] != "tool0"
                ]
            ]
        )
        if len(
            [
                w
                for w in unique_objects
                if w["type"] != "RotatedRectangleClusters" and w["tool"] != "tool0"
            ]
        )
        > 0
        else a["distance_away"],
    )
    end_points = sorted(
        [
            u
            for u in unique_objects
            if u["type"] != "RotatedRectangleClusters" and u["tool"] != "tool0"
        ],
        key=lambda a: extractObjects[start_points[0]["id"]].distance_between_classes(
            extractObjects[a["id"]]
        )
        if len(start_points) > 0
        else a["distance_away"],
    )
    rectangles = sorted(
        [u for u in unique_objects if u["type"] == "RotatedRectangleClusters"],
        key=lambda a: extractObjects[start_points[0]["id"]].distance_between_classes(
            extractObjects[a["id"]]
        )
        if len(start_points) > 0
        else a["distance_away"],
    )
    res = {}
    if len(start_points) > 0:
        res["start"] = start_points[0]
    if len(rectangles) > 0:
        res["box"] = rectangles[0]
    if len(end_points) > 0:
        res["end"] = end_points[0]
    if len(res) == 0:
        assert len(obs) == 1
        if obs[0]["type"] == "RotatedRectangleClusters":
            return {"box": obs[0]}
        else:
            if obs[0]["tool"] == "tool0":
                return {"start": obs[0]}
            return {"end": obs[0]}
    return res


current_extractObjects = {k: v for k, v in extractObjects.items()}
current_objects = seperateObjects(extractObjects)
results = []
while len(cluster_freqs) > 0:
    current_key = sorted(cluster_freqs.keys())[0]
    freq, ids = cluster_freqs[current_key]
    valid = True
    k = current_objects[ids[0]]
    allObjects = [k] + [current_objects[c["id"]] for c in k["clusters"]]
    val = validate(allObjects)
    if all([v["id"] in current_extractObjects for v in val.values()]):
        results.append({k: current_objects[v["id"]] for k, v in val.items()})
        for v in val.values():
            del current_extractObjects[v["id"]]
        valid = False
    if not valid:
        current_objects = seperateObjects(current_extractObjects)
        cluster_freqs = getClusterFreqs(current_objects)

cluster_freqs = {len(i): [0, []] for i in results}
for index, i in enumerate(results):
    cluster_freqs[len(i)][0] += 1
    cluster_freqs[len(i)][1].append(index)

if verbose:
    print(
        "Num of clusters matched up:",
        cluster_freqs[3][0] / sum([k[0] for k in cluster_freqs.values()]),
    )

with open("cluster_groups_" + suffix + ".json", "w") as outfile:
    json.dump(results, outfile, indent=3)


def combineGroupChildren(
    ls: Dict, allObj: Dict, extractObjects: Dict[str, Extractable]
) -> List:
    allChildren = {
        w["id"]: w
        for k in ls.values()
        for a in k["volunteer_labels"]
        for w in allObj[a["id"]]["volunteer_labels"]
    }
    result = {}
    if "start" in ls:
        result["start"] = asdict(extractObjects[ls["start"]["id"]])
    if "box" in ls:
        result["box"] = asdict(extractObjects[ls["box"]["id"]])
    if "end" in ls:
        result["end"] = asdict(extractObjects[ls["end"]["id"]])
    result["children"] = {}
    result["children"]["start"] = [
        asdict(extractObjects[k])
        for k, v in allChildren.items()
        if v["type"] == "ExtractPoint" and v["tool"] == "tool0"
    ]
    result["children"]["box"] = [
        asdict(extractObjects[k])
        for k, v in allChildren.items()
        if v["type"] != "ExtractPoint"
    ]
    result["children"]["end"] = [
        asdict(extractObjects[k])
        for k, v in allChildren.items()
        if v["type"] == "ExtractPoint" and v["tool"] != "tool0"
    ]
    return result


for index, l in enumerate(results):
    results[index] = combineGroupChildren(l, objects, extractObjects)

with open("preimpute_result_" + suffix + ".json", "w") as outfile:
    json.dump(results, outfile, indent=3)

final_result = []
for index, l in enumerate(results):
    clss = {
        "start": (extractObjects[l["start"]["id"]], ExtractPoint)
        if "start" in l
        else (None, ExtractPoint),
        "box": (extractObjects[l["box"]["id"]], ExtractRotatedRectangle)
        if "box" in l
        else (None, ExtractRotatedRectangle),
        "end": (extractObjects[l["end"]["id"]], ExtractPoint)
        if "end" in l
        else (None, ExtractPoint),
    }
    final_dict = {}
    assert any([v[0] is not None for v in clss.values()])
    final_dict["children"] = {}
    for k, class_v in clss.items():
        assert k in l["children"]
        others = {kprime: vprime[0] for kprime, vprime in clss.items() if k != kprime}
        child_to_distance = {
            v["id"]: [None, extractObjects[v["id"]]] for v in l["children"][k]
        }
        for child in child_to_distance:
            for kprime, vprime in others.items():
                if vprime is not None:
                    dis = vprime.distance_between_classes(child_to_distance[child][1])
                    assert dis is not None
                    if child_to_distance[child][0] is None or (
                        child_to_distance[child][0] > dis
                    ):
                        child_to_distance[child][0] = dis

        final_dict["children"][k] = []
        for dis, child in sorted(
            child_to_distance.values(),
            key=lambda x: x[0] if x[0] is not None else 10000000,
        ):
            child_dict = asdict(child)
            child_dict["distance_away"] = dis
            final_dict["children"][k].append(child_dict)
        cluster_result = class_v[1].clusterLabels(
            [extractObjects[child["id"]] for child in final_dict["children"][k]],
            distances=[child["distance_away"] for child in final_dict["children"][k]],
        )
        final_dict["num_volunteers_used_for_" + k] = cluster_result["num_predictions"]
        final_dict["time_entropy_" + k] = cluster_result["time_entropy"]
        final_dict["time_mode_" + k] = cluster_result["time_mode"]
        if class_v[0] is None:
            final_dict[k] = asdict(cluster_result["cluster"].toCluster())
        else:
            final_dict[k] = asdict(class_v[0])
    cluster_result = ExtractRotatedRectangle.clusterLabels(
        [extractObjects[child["id"]] for child in final_dict["children"]["box"]],
        distances=[child["distance_away"] for child in final_dict["children"]["box"]],
    )
    final_dict["imputed_box"] = asdict(cluster_result["cluster"].toCluster())
    final_result.append(final_dict)

with open("final_result_" + suffix + ".json", "w") as outfile:
    json.dump(final_result, outfile, indent=3)


if graphics:
    children_start_distances = [
        i["distance_away"]
        for k in final_result
        for i in k["children"]["start"]
        if i["distance_away"] is not None
    ]
    children_end_distances = [
        i["distance_away"]
        for k in final_result
        for i in k["children"]["end"]
        if i["distance_away"] is not None
    ]
    children_box_distances = [
        i["distance_away"]
        for k in final_result
        for i in k["children"]["box"]
        if i["distance_away"] is not None
    ]

    plt.hist([len(i["children"]["start"]) for i in final_result], bins=20)
    plt.xlabel("num of start children")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist([len(i["children"]["end"]) for i in final_result], bins=20)
    plt.xlabel("num of end children")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist([len(i["children"]["box"]) for i in final_result], bins=20)
    plt.xlabel("num of box children")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist(children_start_distances, bins=100)
    plt.xlabel("Distance a start label is away from nearest cluster")
    plt.ylabel("Frequency")
    plt.axvline(
        x=np.quantile(children_start_distances, 0.5),
        label="mean = {}".format(np.quantile(children_start_distances, 0.5)),
        color="k",
        linestyle="--",
    )
    plt.axvline(
        x=np.quantile(children_start_distances, 0.5) + np.std(children_box_distances),
        label="mean + sigma = {}".format(
            np.quantile(children_start_distances, 0.5) + np.std(children_box_distances)
        ),
        color="k",
        linestyle="-",
    )
    plt.legend()
    plt.show()
    plt.hist(children_end_distances, bins=100)
    plt.xlabel("Distance an end label is away from nearest cluster")
    plt.ylabel("Frequency")
    plt.axvline(
        x=np.quantile(children_end_distances, 0.5),
        label="mean = {}".format(np.quantile(children_end_distances, 0.5)),
        color="k",
        linestyle="--",
    )
    plt.axvline(
        x=np.quantile(children_end_distances, 0.5) + np.std(children_box_distances),
        label="mean + sigma = {}".format(
            np.quantile(children_end_distances, 0.5) + np.std(children_box_distances)
        ),
        color="k",
        linestyle="-",
    )
    plt.legend()
    plt.show()
    plt.hist(children_box_distances, bins=100)
    plt.xlabel("Distance a box label is away from nearest cluster")
    plt.ylabel("Frequency")
    plt.axvline(
        x=np.quantile(children_box_distances, 0.5),
        label="mean = {}".format(np.quantile(children_box_distances, 0.5)),
        color="k",
        linestyle="--",
    )
    plt.axvline(
        x=np.quantile(children_box_distances, 0.5) + np.std(children_box_distances),
        label="mean + sigma = {}".format(
            np.quantile(children_box_distances, 0.5) + np.std(children_box_distances)
        ),
        color="k",
        linestyle="-",
    )
    plt.legend()
    plt.show()

    plt.hist(
        [
            i["time_entropy_start"]
            for i in final_result
            if i["num_volunteers_used_for_start"] > 0
        ],
        bins=20,
    )
    plt.xlabel("time entropy of start labels")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist(
        [
            i["time_entropy_end"]
            for i in final_result
            if i["num_volunteers_used_for_end"] > 0
        ],
        bins=20,
    )
    plt.xlabel("time entropy of end labels")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist(
        [
            i["time_entropy_box"]
            for i in final_result
            if i["num_volunteers_used_for_box"] > 0
        ],
        bins=20,
    )
    plt.xlabel("time entropy of box labels")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist(
        [
            i["time_mode_start"]
            for i in final_result
            if i["num_volunteers_used_for_start"] > 0
        ],
        bins=20,
    )
    plt.xlabel("time mode of start labels")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist(
        [
            i["time_mode_end"]
            for i in final_result
            if i["num_volunteers_used_for_end"] > 0
        ],
        bins=20,
    )
    plt.xlabel("time mode of end labels")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist(
        [
            i["time_mode_box"]
            for i in final_result
            if i["num_volunteers_used_for_box"] > 0
        ],
        bins=20,
    )
    plt.xlabel("time mode of box labels")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist(
        [i["num_volunteers_used_for_start"] for i in final_result], bins=20,
    )
    plt.xlabel("num of start labels within 70 pixel threshhold for each cluster set")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist(
        [i["num_volunteers_used_for_end"] for i in final_result], bins=20,
    )
    plt.xlabel("num of end labels within 70 pixel threshhold for each cluster set")
    plt.ylabel("Frequency")
    plt.show()

    plt.hist(
        [i["num_volunteers_used_for_box"] for i in final_result], bins=20,
    )
    plt.xlabel("num of box labels within 70 pixel threshhold for each cluster set")
    plt.ylabel("Frequency")
    plt.show()
