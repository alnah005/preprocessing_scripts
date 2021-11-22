#!/bin/bash
# cd ~/aggregation_for_caesar && cd htr && panoptes_aggregation reduce question_extractor_htr.csv Reducer_config_workflow_5339_V10.17_question_extractor.yaml -o htr -c 8 -s
# cd ~/aggregation_for_caesar/"htr" && echo "creating box csv" && python -u lines_analysis.py
# cd ~/aggregation_for_caesar/htr && echo "adding location to all_boxes.csv and saving to all_boxes_with_locations.csv" && python -u add_location_to_all_boxes.py
# cd ~/aggregation_for_caesar/htr && echo "using location boxes to create angular_labels_<degree>.csv and moves images from the locations into a folder" && python -u create_diagonal_box_dataset.py
# cd ~/aggregation_for_caesar/htr && echo "converting angular_labels_<degree>.csv into 3 jsons (train,val,test)" && python -u convert_angular_labels_to_coco.py
# cd ~/aggregation_for_caesar/htr && echo "moving actual images based on the 3 jsons" && python -u split_train_test_per_json.py
# cd ~/aggregation_for_caesar/htr && echo "adding width and height to the 3 jsons based on the moved paths" && python -u add_image_height_width_to_json.py
# cd ~/aggregation_for_caesar/htr && echo "cropping images and creating transcription dataset given jsons" && python -u crop_transcription_from_boxes.py
# cd ~/aggregation_for_caesar/htr && echo "creating transcription dataset given jsons" && python -u create_lstm_dataset_given_cropped_images_and_labels.py


# cd ~/aggregation_for_caesar/htr && echo "creating transcription dataset given jsons and box locations without rotating any images" && python -u all_boxes_with_locations_to_non_rotated_train_test_val.py
