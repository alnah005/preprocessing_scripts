# -*- coding: utf-8 -*-

import os
import pandas as pd
import cv2 as cv 
import json
import numpy as np
from collections import defaultdict
from PIL import Image as ImPil
from nms import nms
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Union
from scipy import ndimage as nd
from itertools import islice
from multiprocessing import Process, Manager

# number of cpu cores
n_processes = 24

cwd = os.getcwd()

required_files = {
    # 'annotations_0': cwd+"/final_annotations_10_test.json",
    # 'annotations_1': cwd+"/final_annotations_10_train.json",
    # 'annotations_2': cwd+"/final_annotations_10_val.json",
    'annotations_3': cwd+"/../../detections.json",
    
    # 'image_path_0': cwd+"/../../text_recognition/data/ASM/Original_Images_rotated_10_test/",
    # 'image_path_1': cwd+"/../../text_recognition/data/ASM/Original_Images_rotated_10_train/",
    # 'image_path_2': cwd+"/../../text_recognition/data/ASM/Original_Images_rotated_10_val/",
    'image_path_3': cwd+"/../../text_recognition/data/umnDatasetSample/",
    
    # 'output_path_0': cwd+"/../../text_recognition/data/ASM/Original_Images_cropped_for_transcription_test/",
    # 'output_path_1': cwd+"/../../text_recognition/data/ASM/Original_Images_cropped_for_transcription_train/",
    # 'output_path_2': cwd+"/../../text_recognition/data/ASM/Original_Images_cropped_for_transcription_val/",
    'output_path_3': cwd+"/../../text_recognition/data/umnDatasetSample_cropped/",
}
for i in required_files.values():
    assert os.path.exists(i)


annotations = {}
for k in required_files.keys():
    if 'annotations_' in k[:12]:
        assert '.json' in required_files[k][-5:]
        assert 'image_path_'+k[12:] in required_files.keys()
        assert 'output_path_'+k[12:] in required_files.keys()
        with open(required_files[k]) as json_file:
            annotations[k] = json.load(json_file)


def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """
    #import numpy as np
    #import scipy.ndimage as nd
    assert len(data.shape) == 2
    if invalid is None: invalid = data == ''

    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

def rotate_through_mask_coloring(image_output, centering_ratio, image_top_left,image_bottom_left,image_top_right,image_bottom_right,verbose=False):
    mask_width, mask_height, _ = image_output['blank'].shape
    begin_ratio = (1 - centering_ratio)*0.5
    end_ratio = centering_ratio*0.5
    mask_center = image_output['blank'][int(begin_ratio*mask_width):int(0.5*mask_width)+int(end_ratio*mask_width),int(begin_ratio*mask_height):int(0.5*mask_height)+int(end_ratio*mask_height)]
    result = [["" for j in range(mask_center.shape[1])] for i in range(mask_center.shape[0])]
    for i in range(mask_center.shape[0]):
        for j in range(mask_center.shape[1]):
            r,g,b = mask_center[i,j,:]
            for color, mask in zip(['red','green','blue','white'],[image_top_left,image_top_right,image_bottom_left,image_bottom_right]):
                r_m,g_m,b_m = mask
                if r==r_m and g==g_m and b==b_m:
                    result[i][j] = color

    result = fill(np.array(result))
    
    output_result = ImPil.fromarray(image_output['img'])
    top_left, top_right, bottom_left,bottom_right = result[0][0],result[0][-1],result[-1][0],result[-1][-1]
    if verbose:
        print(result)
        print("top_left, top_right, bottom_left,bottom_right")
        print(top_left, top_right, bottom_left,bottom_right)
    
    if top_left == "red" and top_right == "green" and bottom_left == "blue" and bottom_right == "white":
        if verbose:
            print("do nothing")
    elif top_left == "green" and top_right == "white" and bottom_left == "red" and bottom_right == "blue":
        if verbose:
            print("rotate once to the right")
        output_result = output_result.rotate(-90,expand=True)
    elif top_left == "blue" and top_right == "red" and bottom_left == "white" and bottom_right == "green":
        if verbose:
            print("rotate once to the left")
        output_result = output_result.rotate(90,expand=True)
    elif top_left == "white" and top_right == "blue" and bottom_left == "green" and bottom_right == "red":
        if verbose:
            print("image is inverted rotate 180 degrees")
        output_result = output_result.rotate(180,expand=True)
    else:
        if verbose:
            print("could not locate edges")
        if top_left == "" and top_right == "" and bottom_left == "" and bottom_right == "":
            return None, False
        return output_result, False
    return output_result, True

def crop_angular_box_given_rect(img,rect,verbose=False):    
    box = cv.boxPoints(rect)
    box = np.int0(box)
    image_top_left = (255,0,0)
    image_bottom_left = (0,0,255)
    image_top_right = (0,255,0)
    image_bottom_right = (255,255,255)
    
    max_height, max_width, _ = img.shape
    for i in range(len(box[:,0])):
        box[i,0] = min(max_width-1,box[i,0])
    for i in range(len(box[:,1])):
        box[i,1] = min(max_height-1,box[i,1])
    
    
    min_x = round(min(box[:,0]))
    max_x = round(max(box[:,0]))
    min_y = round(min(box[:,1]))
    max_y = round(max(box[:,1]))

    blank_image = np.zeros(img.shape, np.uint8)
    blank_image[min_y:int(min_y+round(0.5*(max_y-min_y))),min_x:int(min_x+round(0.5*(max_x-min_x)))] = image_top_left
    blank_image[int(min_y+round(0.5*(max_y-min_y))):max_y,min_x:int(min_x+round(0.5*(max_x-min_x)))] = image_bottom_left
    blank_image[min_y:int(min_y+round(0.5*(max_y-min_y))),int(min_x+round(0.5*(max_x-min_x))):max_x] = image_top_right
    blank_image[int(min_y+round(0.5*(max_y-min_y))):max_y,int(min_x+round(0.5*(max_x-min_x))):max_x] = image_bottom_right
    
    image_output = {} 
    image_input = {'img': img,'blank': blank_image}
    for k,v in image_input.items():
    #     cv.drawContours(v, [box], 0, (0, 0, 255), 2) # UNCOMMENT TO VISUALIZE ROTATED BOX
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # the perspective transformation matrix
        M = cv.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv.warpPerspective(v, M, (width, height),flags=cv.INTER_NEAREST)

        image_output[k] = warped
    
    centering_ratios=[
        1,0.8,0.6,0.5,
        0.45,0.40,0.35,
        0.30,0.25,0.20,0.15,0.10
    ]
    
    result, status = rotate_through_mask_coloring(image_output,centering_ratios[0],image_top_left,image_bottom_left,image_top_right,image_bottom_right,verbose)
    output_status = status
    output_result = result
    index = 1
    while not(status) and index < len(centering_ratios):
        result, status = rotate_through_mask_coloring(image_output,centering_ratios[index],image_top_left,image_bottom_left,image_top_right,image_bottom_right,verbose)
        if result is not None:
            output_result = result
            output_status = status
        index += 1
    return output_result, output_status

@dataclass
class RotatedBBox:
    image_id: int
    category_id: int
    bbox: List[float]
    segmentation: List[List[float]]
    text: Optional[str] = None
    box_id: Optional[int] = None
    area: Optional[float] = None
    iscrowd: Optional[Union[int,str,float]] = None
    score: Optional[float] = None
    
    def __post_init__(self):
        assert isinstance(self.image_id,int)
        assert isinstance(self.category_id,int)
        assert isinstance(self.bbox,list)
        for k in self.bbox:
            assert isinstance(k,float)
        assert len(self.bbox)==5
        assert isinstance(self.segmentation,list)
        for k in self.segmentation:
            assert isinstance(k,list)
            for v in k:
                assert isinstance(v, float)
        assert isinstance(self.text,type(None)) or isinstance(self.text,str)        
        assert isinstance(self.box_id,type(None)) or isinstance(self.box_id,int)
        assert isinstance(self.area,type(None)) or isinstance(self.area,int) or isinstance(self.area,float)
        assert isinstance(self.iscrowd, type(None)) or isinstance(self.iscrowd, int) or isinstance(self.iscrowd, float) or isinstance(self.iscrowd, str)
        
        
@dataclass
class RotatedBBoxImage:
    image_id: int
    file_name: str
    BBoxes: List[RotatedBBox]
    height: Optional[Union[int,float]] = None
    width: Optional[Union[int,float]] = None
    _id_from_filename: str = None
    _format: str = None
    def __post_init__(self):
        assert isinstance(self.image_id,int)
        assert isinstance(self.file_name,str)
        assert isinstance(self.height,type(None)) or isinstance(self.height, int) or isinstance(self.height, float)
        assert isinstance(self.width,type(None)) or isinstance(self.width, int) or isinstance(self.width, float)
    
    def addBboxFromJson(self,json: Dict[str,float]):
        assert 'image_id' in json.keys()
        assert json['image_id'] == self.image_id
        assert 'category_id' in json.keys()
        assert 'bbox' in json.keys()
        assert 'segmentation' in json.keys()
        self.BBoxes.append(
            RotatedBBox(
                box_id=json.get('id', None),
                image_id=self.image_id,
                category_id=json['category_id'],
                bbox=json['bbox'],
                segmentation=json['segmentation'],
                area=json.get('area', None),
                iscrowd=json.get('iscrowd', None),
                text=json.get('text',None),
                score=json.get('score',None),
            )
        )
        
    def getfileId(self):
        if self._id_from_filename is None:
            self._id_from_filename = '.'.join(self.file_name.split('.')[:-1])
            self._format = self.file_name.split('.')[-1]
        return self._id_from_filename
    
    def getformat(self):
        if self._format is None:
            self._format = self.file_name.split('.')[-1]
            self._id_from_filename = '.'.join(self.file_name.split('.')[:-1])
        return self._format
    
    def getFullPath(self,parentPath:str):
        full_path = os.path.join(parentPath, self.file_name)
        if not(os.path.exists(full_path)):
            print(full_path,"does not exist")
            return None
        return full_path



Images_per_annotation = {annot: {} for annot in annotations.keys()}

for annot in annotations.keys():
    for image in annotations[annot].get("images",[]):
        assert 'id' in image.keys()
        assert 'file_name' in image.keys()
        assert not(image['id'] in Images_per_annotation[annot])
        height = image.get('height', None)
        width = image.get('width', None)
        Images_per_annotation[annot][image['id']] = RotatedBBoxImage(
            image_id=image['id'],
            file_name=image['file_name'],
            height=height,
            width=width,
            BBoxes=[],
        )
    print("Number of images processed for ",required_files[annot],len(Images_per_annotation[annot]))
    num_boxes = 0
    for box in annotations[annot].get("annotations",[]):
        image_id = box.get('image_id',None)
        assert image_id is not None
        Images_per_annotation[annot][image_id].addBboxFromJson(box)
        num_boxes += 1
    print("Number of boxes processed for ",required_files[annot],num_boxes)



def process_images(image_list,annot,final_annotations_json,successful_per_annotation,successful_per_annotation_default_behavior,successful_per_annotation_no_default_behavior,errors):
    for im_id in image_list.keys():
        img_name, img_format = image_list[im_id].getfileId(), image_list[im_id].getformat()
        full_path = image_list[im_id].getFullPath(required_files[image_parent_path])
        if full_path is not None:
            image = cv.imread(full_path)
        else:
            print("couldn't find ", annot, im_id, "continuing")
            continue
        for box_num in range(len(image_list[im_id].BBoxes)):
            box = image_list[im_id].BBoxes[box_num]
            text = box.text
            cx: float = box.bbox[0]+(0.5*box.bbox[2])
            cy: float = box.bbox[1]+(0.5*box.bbox[3])
            w: float = box.bbox[2]+40
            h: float = box.bbox[3]+40
            a: float = box.bbox[4]*180/np.pi
            rect = [[cx,cy],[w,h],a]
            try:
                new_path = os.path.join(required_files['output_path_'+annot[12:]],img_name+'_'+str(box_num)+'.'+img_format)
                rotated_image_pil, status = crop_angular_box_given_rect(image, rect)
                assert not(rotated_image_pil is None)
                if not(status):
                    if rotated_image_pil.size[1]>rotated_image_pil.size[0]:
                        print(full_path,rect, "rotating by default", new_path)
                        rotated_image_pil = rotated_image_pil.rotate(-90,expand=True)
                    else:
                        print(full_path,rect, "not rotating by default", new_path)
                rotated_image_pil.save(new_path)
                rotated_image = np.array(rotated_image_pil)
                height, width = rotated_image.shape[0], rotated_image.shape[1]
                new_annot = {
                        'im_path': new_path,
                        'im_id': box.image_id,
                        'bbox': box.bbox,
                        'text': text,
                        'crop_h': height,
                        'crop_w': width,
                        'area': height*width,
                        'box_score': box.score, 
                }
                if box.box_id is not None:
                    new_annot['box_id'] = box.box_id
                final_annotations_json[annot].append(new_annot)
                successful_per_annotation[annot]+= 1
                if status:
                    successful_per_annotation_no_default_behavior[annot] += 1
                else:
                    successful_per_annotation_default_behavior[annot] += 1
            except:
                print("error", full_path,"box",asdict(box))
                errors[annot] += 1
                continue


def chunks(data: dict, SIZE=10000):
    # data must be a one dimensional dict
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}

manager = Manager()
successful_per_annotation = manager.dict({annot: 0 for annot in annotations.keys()})
successful_per_annotation_no_default_behavior = manager.dict({annot: 0 for annot in annotations.keys()})
successful_per_annotation_default_behavior = manager.dict({annot: 0 for annot in annotations.keys()})
final_annotations_json = manager.dict({annot: manager.list() for annot in annotations.keys()})
errors = manager.dict({annot: 0 for annot in annotations.keys()})
for annot in annotations.keys():
    annotation_num = annot.split('_')[-1]
    image_parent_path = 'image_path_'+annotation_num
    process_dicts = []
    for item in chunks(Images_per_annotation[annot],SIZE=1 + round(len(Images_per_annotation[annot])/n_processes)):
        process_dicts.append(item)
    try:
        processes_list = list()
        for i,dic in enumerate(process_dicts):
            pr = Process(target=process_images,
                         args=(dic,annot,final_annotations_json,successful_per_annotation,successful_per_annotation_default_behavior,successful_per_annotation_no_default_behavior,errors))
            pr.start()
            processes_list.append(pr)
        for p in processes_list:
            p.join()
    except Exception:
        print(traceback.format_exc())
    new_file = '.'.join(required_files[annot].split('.')[:-1])
    new_file += '_cropped.'+ required_files[annot].split('.')[-1]
    with open(new_file, "w") as outfile: 
        json.dump(list(final_annotations_json[annot]), outfile,indent=1)                  
print(successful_per_annotation)
print(successful_per_annotation_no_default_behavior)
print(successful_per_annotation_default_behavior)
print(errors)

