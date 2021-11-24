# Solar jets

This directory assumes that you've run the volunteer transcriptions through the [aggregation_for_caesar](https://aggregation-caesar.zooniverse.org/Scripts.html) repo. The files used in the scripts as inputs won't make sense otherwise.

## Files

Below is a detailed description of each file in this directory

### `extracts_to_cluster_dataclasses.py`

This file contains classes that help encapsulate the tools used in the Solar jets task. If you follow the same structure in creating children classes, there's nothing stopping you from generalizing the work to other tools/tasks.

### `convert_task_and_match_tools.py`

During the beta tests, I proposed the idea that all volunteer labels that are in seperate frames could be squashed onto one frame, then a clustering algorithm can be tested to generate more general clusters. Since aggregation_for_caesar does this robustly using standardized csv file structure, the code in this files converts a multi frame label task into a one frame task. The sole requirement is each extract can have at most **ONE** label in a certain frame

### `connect_extracts_to_clusters.py`

This script utilizes the classes created in `extracts_to_cluster_dataclasses.py` to process extracts and clusters generated from aggregation_for_caesar. Up until line 151, the code might look unclear, but what it's essentially doing is processing the csv's into the `extractObjects` dictionary. You'll notice that there are two dictionaries that are hardcoded (`extracts` and `extracts_dataclasses`). These two dictionaries define which dataclass is associated to which csv file from aggregation_for_caesar. I've included below a view of what each file looks like, so I hope that clarifies things for you. The code has two parts:

1. General code that calculates the distance between each object that was extracted

2. [Solar jets beta specific code that finds the cluster set by aggregating volunteer responses.](https://docs.google.com/presentation/d/1K5JiaPdL2EfD3ys0HmKKduMYQQdTGdO5pOieUNiiOiI/edit?usp=sharing)

## Input/Output files views

### `point_extractor_by_frame_beta2.csv`

|classification_id|user_name|user_id  |workflow_id|task|created_at             |subject_id|extractor               |data.aggregation_version|data.frame2.T1_tool0_x|data.frame2.T1_tool0_y|data.frame14.T1_tool1_x|data.frame14.T1_tool1_y|data.frame14.T1_tool0_x|data.frame14.T1_tool0_y|data.frame1.T1_tool0_x|data.frame1.T1_tool0_y|data.frame13.T1_tool1_x|data.frame13.T1_tool1_y|data.frame11.T1_tool0_x|data.frame11.T1_tool0_y|data.frame11.T1_tool1_x|data.frame11.T1_tool1_y|data.frame11.T5_tool0_x|data.frame11.T5_tool0_y|data.frame11.T5_tool1_x|data.frame11.T5_tool1_y|data.frame0.T1_tool0_x|data.frame0.T1_tool0_y|data.frame4.T1_tool0_x|data.frame4.T1_tool0_y|data.frame4.T1_tool1_x|data.frame4.T1_tool1_y|data.frame3.T1_tool0_x|data.frame3.T1_tool0_y|data.frame3.T1_tool1_x|data.frame3.T1_tool1_y|data.frame2.T5_tool0_x|data.frame2.T5_tool0_y|data.frame2.T5_tool1_x|data.frame2.T5_tool1_y|data.frame0.T5_tool0_x|data.frame0.T5_tool0_y|data.frame14.T5_tool1_x|data.frame14.T5_tool1_y|data.frame6.T1_tool0_x|data.frame6.T1_tool0_y|data.frame8.T1_tool1_x|data.frame8.T1_tool1_y|data.frame5.T1_tool0_x|data.frame5.T1_tool0_y|data.frame10.T1_tool1_x|data.frame10.T1_tool1_y|data.frame6.T5_tool0_x|data.frame6.T5_tool0_y|data.frame9.T1_tool1_x|data.frame9.T1_tool1_y|data.frame13.T5_tool1_x|data.frame13.T5_tool1_y|data.frame5.T1_tool1_x|data.frame5.T1_tool1_y|data.frame6.T5_tool1_x|data.frame6.T5_tool1_y|data.frame9.T1_tool0_x|data.frame9.T1_tool0_y|data.frame10.T5_tool0_x|data.frame10.T5_tool0_y|data.frame10.T5_tool1_x|data.frame10.T5_tool1_y|data.frame7.T1_tool0_x|data.frame7.T1_tool0_y|data.frame12.T1_tool1_x|data.frame12.T1_tool1_y|data.frame0.T1_tool1_x|data.frame0.T1_tool1_y|data.frame0.T5_tool1_x|data.frame0.T5_tool1_y|data.frame4.T5_tool0_x|data.frame4.T5_tool0_y|data.frame4.T5_tool1_x|data.frame4.T5_tool1_y|data.frame8.T5_tool0_x|data.frame8.T5_tool0_y|data.frame8.T5_tool1_x|data.frame8.T5_tool1_y|data.frame9.T5_tool0_x|data.frame9.T5_tool0_y|data.frame7.T1_tool1_x|data.frame7.T1_tool1_y|data.frame10.T1_tool0_x|data.frame10.T1_tool0_y|data.frame13.T5_tool0_x|data.frame13.T5_tool0_y|data.frame1.T1_tool1_x|data.frame1.T1_tool1_y|data.frame8.T1_tool0_x|data.frame8.T1_tool0_y|data.frame9.T5_tool1_x|data.frame9.T5_tool1_y|data.frame7.T5_tool0_x|data.frame7.T5_tool0_y|data.frame5.T5_tool0_x|data.frame5.T5_tool0_y|data.frame5.T5_tool1_x|data.frame5.T5_tool1_y|data.frame14.T5_tool0_x|data.frame14.T5_tool0_y|data.frame7.T5_tool1_x|data.frame7.T5_tool1_y|data.frame12.T5_tool0_x|data.frame12.T5_tool0_y|data.frame6.T1_tool1_x|data.frame6.T1_tool1_y|data.frame13.T1_tool0_x|data.frame13.T1_tool0_y|data.frame3.T5_tool0_x|data.frame3.T5_tool0_y|data.frame1.T5_tool0_x|data.frame1.T5_tool0_y|data.frame1.T5_tool1_x|data.frame1.T5_tool1_y|data.frame12.T1_tool0_x|data.frame12.T1_tool0_y|
|-----------------|---------|---------|-----------|----|-----------------------|----------|------------------------|------------------------|----------------------|----------------------|-----------------------|-----------------------|-----------------------|-----------------------|----------------------|----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|-----------------------|-----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|-----------------------|-----------------------|----------------------|----------------------|----------------------|----------------------|-----------------------|-----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|-----------------------|-----------------------|-----------------------|-----------------------|----------------------|----------------------|-----------------------|-----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|-----------------------|-----------------------|-----------------------|-----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|-----------------------|-----------------------|----------------------|----------------------|-----------------------|-----------------------|----------------------|----------------------|-----------------------|-----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|-----------------------|-----------------------|
|363540732        |SophieMu |2058613.0|19650      |T1  |2021-09-29 09:10:47 UTC|68916589  |point_extractor_by_frame|3.6.0                   |[1198.0238037109375]  |[714.0210571289062]   |[1180.194091796875]    |[710.4551391601562]    |                       |                       |                      |                      |                       |                       |                       |                       |                       |                       |                       |                       |                       |                       |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                       |                       |                      |                      |                      |                      |                      |                      |                       |                       |                      |                      |                      |                      |                       |                       |                      |                      |                      |                      |                      |                      |                       |                       |                       |                       |                      |                      |                       |                       |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                       |                       |                       |                       |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                       |                       |                      |                      |                       |                       |                      |                      |                       |                       |                      |                      |                      |                      |                      |                      |                       |                       |
|363540732        |SophieMu |2058613.0|19650      |T5  |2021-09-29 09:10:47 UTC|68916589  |point_extractor_by_frame|3.6.0                   |                      |                      |                       |                       |                       |                       |                      |                      |                       |                       |                       |                       |                       |                       |                       |                       |                       |                       |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                       |                       |                      |                      |                      |                      |                      |                      |                       |                       |                      |                      |                      |                      |                       |                       |                      |                      |                      |                      |                      |                      |                       |                       |                       |                       |                      |                      |                       |                       |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                       |                       |                       |                       |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                      |                       |                       |                      |                      |                       |                       |                      |                      |                       |                       |                      |                      |                      |                      |                      |                      |                       |                       |

### `shape_extractor_rotateRectangle_beta2.csv`

|classification_id|user_name|user_id  |workflow_id|task|created_at             |subject_id|extractor               |data.aggregation_version|data.frame14.T1_tool2_x|data.frame14.T1_tool2_y|data.frame14.T1_tool2_width|data.frame14.T1_tool2_height|data.frame14.T1_tool2_angle|data.frame13.T1_tool2_x|data.frame13.T1_tool2_y|data.frame13.T1_tool2_width|data.frame13.T1_tool2_height|data.frame13.T1_tool2_angle|data.frame11.T1_tool2_x|data.frame11.T1_tool2_y|data.frame11.T1_tool2_width|data.frame11.T1_tool2_height|data.frame11.T1_tool2_angle|data.frame11.T5_tool2_x|data.frame11.T5_tool2_y|data.frame11.T5_tool2_width|data.frame11.T5_tool2_height|data.frame11.T5_tool2_angle|data.frame4.T1_tool2_x|data.frame4.T1_tool2_y|data.frame4.T1_tool2_width|data.frame4.T1_tool2_height|data.frame4.T1_tool2_angle|data.frame3.T1_tool2_x|data.frame3.T1_tool2_y|data.frame3.T1_tool2_width|data.frame3.T1_tool2_height|data.frame3.T1_tool2_angle|data.frame2.T5_tool2_x|data.frame2.T5_tool2_y|data.frame2.T5_tool2_width|data.frame2.T5_tool2_height|data.frame2.T5_tool2_angle|data.frame14.T5_tool2_x|data.frame14.T5_tool2_y|data.frame14.T5_tool2_width|data.frame14.T5_tool2_height|data.frame14.T5_tool2_angle|data.frame5.T5_tool2_x|data.frame5.T5_tool2_y|data.frame5.T5_tool2_width|data.frame5.T5_tool2_height|data.frame5.T5_tool2_angle|data.frame12.T1_tool2_x|data.frame12.T1_tool2_y|data.frame12.T1_tool2_width|data.frame12.T1_tool2_height|data.frame12.T1_tool2_angle|data.frame7.T1_tool2_x|data.frame7.T1_tool2_y|data.frame7.T1_tool2_width|data.frame7.T1_tool2_height|data.frame7.T1_tool2_angle|data.frame10.T5_tool2_x|data.frame10.T5_tool2_y|data.frame10.T5_tool2_width|data.frame10.T5_tool2_height|data.frame10.T5_tool2_angle|data.frame6.T1_tool2_x|data.frame6.T1_tool2_y|data.frame6.T1_tool2_width|data.frame6.T1_tool2_height|data.frame6.T1_tool2_angle|data.frame8.T1_tool2_x|data.frame8.T1_tool2_y|data.frame8.T1_tool2_width|data.frame8.T1_tool2_height|data.frame8.T1_tool2_angle|data.frame13.T5_tool2_x|data.frame13.T5_tool2_y|data.frame13.T5_tool2_width|data.frame13.T5_tool2_height|data.frame13.T5_tool2_angle|data.frame5.T1_tool2_x|data.frame5.T1_tool2_y|data.frame5.T1_tool2_width|data.frame5.T1_tool2_height|data.frame5.T1_tool2_angle|data.frame0.T5_tool2_x|data.frame0.T5_tool2_y|data.frame0.T5_tool2_width|data.frame0.T5_tool2_height|data.frame0.T5_tool2_angle|data.frame9.T1_tool2_x|data.frame9.T1_tool2_y|data.frame9.T1_tool2_width|data.frame9.T1_tool2_height|data.frame9.T1_tool2_angle|data.frame10.T1_tool2_x|data.frame10.T1_tool2_y|data.frame10.T1_tool2_width|data.frame10.T1_tool2_height|data.frame10.T1_tool2_angle|data.frame0.T1_tool2_x|data.frame0.T1_tool2_y|data.frame0.T1_tool2_width|data.frame0.T1_tool2_height|data.frame0.T1_tool2_angle|data.frame2.T1_tool2_x|data.frame2.T1_tool2_y|data.frame2.T1_tool2_width|data.frame2.T1_tool2_height|data.frame2.T1_tool2_angle|data.frame12.T5_tool2_x|data.frame12.T5_tool2_y|data.frame12.T5_tool2_width|data.frame12.T5_tool2_height|data.frame12.T5_tool2_angle|data.frame1.T5_tool2_x|data.frame1.T5_tool2_y|data.frame1.T5_tool2_width|data.frame1.T5_tool2_height|data.frame1.T5_tool2_angle|data.frame4.T5_tool2_x|data.frame4.T5_tool2_y|data.frame4.T5_tool2_width|data.frame4.T5_tool2_height|data.frame4.T5_tool2_angle|data.frame9.T5_tool2_x|data.frame9.T5_tool2_y|data.frame9.T5_tool2_width|data.frame9.T5_tool2_height|data.frame9.T5_tool2_angle|data.frame1.T1_tool2_x|data.frame1.T1_tool2_y|data.frame1.T1_tool2_width|data.frame1.T1_tool2_height|data.frame1.T1_tool2_angle|data.frame6.T5_tool2_x|data.frame6.T5_tool2_y|data.frame6.T5_tool2_width|data.frame6.T5_tool2_height|data.frame6.T5_tool2_angle|data.frame3.T5_tool2_x|data.frame3.T5_tool2_y|data.frame3.T5_tool2_width|data.frame3.T5_tool2_height|data.frame3.T5_tool2_angle|data.frame8.T5_tool2_x|data.frame8.T5_tool2_y|data.frame8.T5_tool2_width|data.frame8.T5_tool2_height|data.frame8.T5_tool2_angle|
|-----------------|---------|---------|-----------|----|-----------------------|----------|------------------------|------------------------|-----------------------|-----------------------|---------------------------|----------------------------|---------------------------|-----------------------|-----------------------|---------------------------|----------------------------|---------------------------|-----------------------|-----------------------|---------------------------|----------------------------|---------------------------|-----------------------|-----------------------|---------------------------|----------------------------|---------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|-----------------------|-----------------------|---------------------------|----------------------------|---------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|-----------------------|-----------------------|---------------------------|----------------------------|---------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|-----------------------|-----------------------|---------------------------|----------------------------|---------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|-----------------------|-----------------------|---------------------------|----------------------------|---------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|-----------------------|-----------------------|---------------------------|----------------------------|---------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|-----------------------|-----------------------|---------------------------|----------------------------|---------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|----------------------|----------------------|--------------------------|---------------------------|--------------------------|
|363540732        |SophieMu |2058613.0|19650      |T1  |2021-09-29 09:10:47 UTC|68916589  |shape_extractor_rotateRectangle|3.6.0                   |[1050.4191831303847]   |[586.3417273222584]    |[166.83521772360572]       |[105.5890062929832]         |[53.471134209473654]       |                       |                       |                           |                            |                           |                       |                       |                           |                            |                           |                       |                       |                           |                            |                           |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                       |                       |                           |                            |                           |                      |                      |                          |                           |                          |                       |                       |                           |                            |                           |                      |                      |                          |                           |                          |                       |                       |                           |                            |                           |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                       |                       |                           |                            |                           |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                       |                       |                           |                            |                           |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                       |                       |                           |                            |                           |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |
|363540732        |SophieMu |2058613.0|19650      |T5  |2021-09-29 09:10:47 UTC|68916589  |shape_extractor_rotateRectangle|3.6.0                   |                       |                       |                           |                            |                           |                       |                       |                           |                            |                           |                       |                       |                           |                            |                           |                       |                       |                           |                            |                           |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                       |                       |                           |                            |                           |                      |                      |                          |                           |                          |                       |                       |                           |                            |                           |                      |                      |                          |                           |                          |                       |                       |                           |                            |                           |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                       |                       |                           |                            |                           |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                       |                       |                           |                            |                           |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                       |                       |                           |                            |                           |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |                      |                      |                          |                           |                          |

### `shape_reducer_hdbscan_beta2_proposed.csv`

|subject_id|workflow_id|task     |reducer|data.aggregation_version|data.frame0.T5_tool2_rotateRectangle_x|data.frame0.T5_tool2_rotateRectangle_y|data.frame0.T5_tool2_rotateRectangle_width|data.frame0.T5_tool2_rotateRectangle_height|data.frame0.T5_tool2_rotateRectangle_angle|data.frame0.T5_tool2_cluster_labels|data.frame0.T5_tool2_cluster_probabilities|data.frame0.T5_tool2_clusters_persistance|data.frame0.T5_tool2_clusters_count|data.frame0.T5_tool2_clusters_x|data.frame0.T5_tool2_clusters_y|data.frame0.T5_tool2_clusters_width|data.frame0.T5_tool2_clusters_height|data.frame0.T5_tool2_clusters_angle|
|----------|-----------|---------|-------|------------------------|--------------------------------------|--------------------------------------|------------------------------------------|-------------------------------------------|------------------------------------------|-----------------------------------|------------------------------------------|-----------------------------------------|-----------------------------------|-------------------------------|-------------------------------|-----------------------------------|------------------------------------|-----------------------------------|
|68916566  |19650      |T5       |shape_reducer_hdbscan|3.6.0                   |[436.59771728515625, 432.73236083984375, 611.7157296143918, 426.0833361359319]|[452.50030517578125, 671.61572265625, 653.0216844292783, 687.7042540833444]|[314.8631591796875, 337.61419677734375, 150.29815502902886, 321.95838876329236]|[357.73187255859375, 145.81781005859375, 312.38866849495884, 106.83355726299885]|[0.0, 0.0, 141.3934993121655, 8.146048794026228]|[0, 0, -1, -1]                     |[1.0, 1.0, 0.0, 0.0]                      |[1.9792783143494543]                     |[2]                                |[434.6650390625]               |[562.0580139160156]            |[326.2386779785156]                |[251.77484130859375]                |[0.0]                              |
|68916567  |19650      |T5       |shape_reducer_hdbscan|3.6.0                   |[573.7241821289062, 436.1628723144531, 418.0433654785156, 512.1576242568685, 457.2506408691406, 508.6251220703125]|[522.6204223632812, 706.9711303710938, 650.672607421875, 664.8049148334748, 675.1312866210938, 598.09619140625]|[353.963134765625, 298.0769348144531, 311.6398010253906, 396.32604787298186, 395.7224426269531, 305.14031982421875]|[449.511474609375, 75.0, 145.14727783203125, 171.09201359476873, 100.050537109375, 245.80743408203125]|[-16.161378545762243, 0.0, 0.0, 5.132554581112165, 0.0, 41.820168356772314]|[-1, -1, 0, -1, 0, -1]             |[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]            |[2.366356820306829]                      |[2]                                |[437.6470031738281]            |[662.9019470214844]            |[353.6811218261719]                |[122.59890747070312]                |[0.0]                              |

### `point_reducer_hdbscan_beta2_proposed.csv`

|subject_id|workflow_id|task     |reducer|data.aggregation_version|data.frame0.T5_tool1_points_x|data.frame0.T5_tool1_points_y|data.frame0.T5_tool1_cluster_labels|data.frame0.T5_tool1_cluster_probabilities|data.frame0.T5_tool1_clusters_persistance|data.frame0.T5_tool1_clusters_count|data.frame0.T5_tool1_clusters_x|data.frame0.T5_tool1_clusters_y|data.frame0.T5_tool1_clusters_var_x|data.frame0.T5_tool1_clusters_var_y|data.frame0.T5_tool1_clusters_var_x_y|data.frame0.T5_tool0_points_x|data.frame0.T5_tool0_points_y|data.frame0.T5_tool0_cluster_labels|data.frame0.T5_tool0_cluster_probabilities|data.frame0.T5_tool0_clusters_persistance|data.frame0.T5_tool0_clusters_count|data.frame0.T5_tool0_clusters_x|data.frame0.T5_tool0_clusters_y|data.frame0.T5_tool0_clusters_var_x|data.frame0.T5_tool0_clusters_var_y|data.frame0.T5_tool0_clusters_var_x_y|
|----------|-----------|---------|-------|------------------------|-----------------------------|-----------------------------|-----------------------------------|------------------------------------------|-----------------------------------------|-----------------------------------|-------------------------------|-------------------------------|-----------------------------------|-----------------------------------|-------------------------------------|-----------------------------|-----------------------------|-----------------------------------|------------------------------------------|-----------------------------------------|-----------------------------------|-------------------------------|-------------------------------|-----------------------------------|-----------------------------------|-------------------------------------|
|68916566  |19650      |T5       |point_reducer_hdbscan|3.6.0                   |[705.6357421875, 502.3570861816406, 741.53466796875, 462.0648498535156]|[755.5377197265625, 747.808837890625, 813.7115745544434, 675.5194091796875]|[0, 0, -1, -1]                     |[1.0, 1.0, 0.0, 0.0]                      |[1.822331090997288]                      |[2]                                |[603.9964141845703]            |[751.6732788085938]            |[20661.105993774254]               |[29.86780721694231]                |[785.558356018737]                   |[584.4207763671875, 510.2391357421875, 607.1107559204102, 483.341064453125]|[631.3662109375, 705.771240234375, 698.5321044921875, 647.1511840820312]|[0, 0, -1, -1]                     |[1.0, 1.0, 0.0, 0.0]                      |[1.7840162190484607]                     |[2]                                |[547.3299560546875]            |[668.5687255859375]            |[2751.457902908325]                |[2768.0541923344135]               |[-2759.743571996689]                 |
|68916567  |19650      |T5       |point_reducer_hdbscan|3.6.0                   |[864.7122802734375, 545.7782592773438, 775.2195434570312, 544.750244140625, 494.58294677734375, 826.4796142578125]|[800.5791625976562, 737.7403564453125, 716.1311645507812, 738.0211791992188, 737.8495483398438, 860.8558959960938]|[-1, 0, 0, -1, -1, -1]             |[0.0, 1.0, 1.0, 0.0, 0.0, 0.0]            |[2.637513702411564]                      |[2]                                |[660.4989013671875]            |[726.9357604980469]            |[26321.65144301206]                |[233.47858716733754]               |[-2479.020369183272]                 |[575.895751953125, 538.0859375, 465.0027770996094, 535.9432983398438, 515.4890441894531, 517.1012573242188]|[728.9179077148438, 722.3557739257812, 699.0550537109375, 709.8389892578125, 700.5172729492188, 687.095458984375]|[-1, -1, -1, 0, 0, -1]             |[0.0, 0.0, 0.0, 1.0, 1.0, 0.0]            |[2.2467478825913325]                     |[2]                                |[525.7161712646484]            |[705.1781311035156]            |[209.18825642438605]               |[43.447197468951344]               |[95.33437724690884]                  |

## [More Info](https://z.umn.edu/solar_jets_aggregation)