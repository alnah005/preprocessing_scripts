#!/bin/bash

# cd ~/aggregation_for_caesar && cd "htr_gold_standard" && panoptes_aggregation config acls-htr-gold-standard-data-workflows.csv 18227 -v 5
# cd ~/aggregation_for_caesar && cd "htr_gold_standard" && panoptes_aggregation extract acls-htr-gold-standard-data-classifications.csv Extractor_config_workflow_18227_V5.12.yaml -o gold
# cd ~/aggregation_for_caesar && cd "htr_gold_standard" && panoptes_aggregation reduce line_text_extractor_gold.csv Reducer_config_workflow_18227_V5.12_line_text_extractor.yaml -o htr #-c 8 -s
# cd ~/aggregation_for_caesar/"htr_gold_standard" && echo "creating box csv" && python -u lines_analysis.py
# cd ~/aggregation_for_caesar/ && echo "after running subjects to links google colab, downloading images to specified dir" && python -u download_images.py
# cd ~/aggregation_for_caesar/"htr_gold_standard" && echo "adding location to all_boxes.csv and saving to all_boxes_with_locations.csv" && python -u add_location_to_all_boxes.py
# cd ~/aggregation_for_caesar/"htr_gold_standard" && echo "using location boxes to create angular_labels_<degree>.csv and moves images from the locations into a folder" && python -u create_diagonal_box_dataset.py
# cd ~/aggregation_for_caesar/"htr_gold_standard" && echo "converting angular_labels_<degree>.csv into 3 jsons (train,val,test)" && python -u convert_angular_labels_to_coco.py
# cd ~/aggregation_for_caesar/"htr_gold_standard" && echo "moving actual images based on the 3 jsons" && python -u split_train_test_per_json.py
# cd ~/aggregation_for_caesar/"htr_gold_standard" && echo "adding width and height to the 3 jsons based on the moved paths" && python -u add_image_height_width_to_json.py