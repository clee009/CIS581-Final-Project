import json
import os
import pdb
import cv2
import numpy as np


img_folder = "/home/ziyan/02_research/4D-Humans/Baseline_video"

json_path = os.path.join(img_folder,"createml_with_id.json")
with open(json_path, 'r') as file:
    annotations = json.load(file)

def integration(annotations, facing_direction_whole_dataset):
    # Rearrange data
    rearranged_data = {
        entry["image"]: [
            {
                "id": annotation["id"],
                "label": annotation["label"],
                "coordinates": [
                    annotation["coordinates"]["x"],
                    annotation["coordinates"]["y"],
                    annotation["coordinates"]["width"],
                    annotation["coordinates"]["height"],
                ],
            }
            for annotation in entry["annotations"]
        ]
        for entry in annotations
    }



    # Update rearranged_data with matching facing direction and position data
    for key in rearranged_data.keys():
        if key in facing_direction_whole_dataset:
            # Iterate over entries in rearranged_data[key]
            for item in rearranged_data[key]:
                # Get the bounding box coordinates
                coordinates = item["coordinates"]

                # Extract the center point of the bounding box (x_center, y_center)
                x_center = coordinates[0]
                y_center = coordinates[1]

                # Match with the position in the second dictionary
                for i, position in enumerate(facing_direction_whole_dataset[key]["position_2d"]):
                    pos_x, pos_y = position  # Extract position coordinates
                    
                    # Check if the positions are approximately equal
                    if abs(x_center - pos_x) < 10 and abs(y_center - pos_y) < 10:
                        # Add facing direction and position data to the item
                        item["facing_direction_2d"] = facing_direction_whole_dataset[key]["facing_direction_2d"][i]
                        item["facing_direction_3d"] = facing_direction_whole_dataset[key]["facing_direction_3d"][i]
                        item["position_2d"] = position
                        break  # Stop searching after finding a match


    # Create a new dictionary to store matched entries
    matched_data = {}

    # Iterate over keys in rearranged_data
    for key in rearranged_data.keys():
        if key in facing_direction_whole_dataset:
            matched_data[key] = []  # Initialize an empty list for this key

            # Iterate over entries in rearranged_data[key]
            for item in rearranged_data[key]:
                # Get the bounding box coordinates
                coordinates = item["coordinates"]
                
                if item['label'] == 'basketball':
                    matched_item = {
                            "id": item["id"],
                            "label": item["label"],
                            "coordinates" : {
                                "x": coordinates[0],
                                "y": coordinates[1],
                                "width": coordinates[2],
                                "height": coordinates[3],
                            },
                        }
                    matched_data[key].append(matched_item)
                    continue

                # Extract the center point of the bounding box (x_center, y_center)
                x_center = coordinates[0]
                y_center = coordinates[1]

                # Match with the position in the second dictionary
                for i, position in enumerate(facing_direction_whole_dataset[key]["position_2d"]):
                    pos_x, pos_y = position  # Extract position coordinates

                    # Check if the positions are approximately equal
                    if abs(x_center - pos_x) < 10 and abs(y_center - pos_y) < 10:
                        # Create a new matched item
                        matched_item = {
                            "id": item["id"],
                            "label": item["label"],
                            "coordinates" : {
                                "x": coordinates[0],
                                "y": coordinates[1],
                                "width": coordinates[2],
                                "height": coordinates[3],
                            },
                            "facing_direction_2d": facing_direction_whole_dataset[key]["facing_direction_2d"][i],
                            "facing_direction_3d": facing_direction_whole_dataset[key]["facing_direction_3d"][i],
                            "position_2d": position,
                        }
                        matched_data[key].append(matched_item)  # Add to the new dictionary
                        break  # Stop searching after finding a match



    integration_list = []                    
    for key, item in matched_data.items():
        dictFor1img = {
            "image": key,
        }
        dictFor1img["annotations"] = []
        for subdict in item:
            dictFor1img["annotations"].append(subdict)
        integration_list.append(dictFor1img)
    
    return integration_list

