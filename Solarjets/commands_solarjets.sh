#!/bin/bash
# cd ~/aggregation_for_caesar && cd solar_jets_box_workflow_2 && panoptes_aggregation config solar-jet-hunter-workflows.csv 19650 -v 4
# cd ~/aggregation_for_caesar && cd solar_jets_box_workflow_2 && panoptes_aggregation extract solar-jet-hunter-classifications.csv Extractor_config_workflow_19650_V4.15.yaml -o beta2
cd ~/aggregation_for_caesar && cd solar_jets_box_workflow_2 && panoptes_aggregation reduce shape_extractor_rotateRectangle_beta2.csv Reducer_config_workflow_19650_V4.15_shape_extractor_rotateRectangle.yaml -o beta2 #-c 8 -s
# cd ~/aggregation_for_caesar && cd solar_jets_box_workflow_2 && panoptes_aggregation reduce question_extractor_beta2.csv Reducer_config_workflow_19650_V4.15_question_extractor.yaml -o beta2 #-c 8 -s
cd ~/aggregation_for_caesar && cd solar_jets_box_workflow_2 && panoptes_aggregation reduce point_extractor_by_frame_beta2.csv Reducer_config_workflow_19650_V4.15_point_extractor_by_frame.yaml -o beta2 #-c 8 -s