from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils import visualizer as vis
import numpy as np
import cv2
from PIL import Image
import random
import shutil
import os


from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

coco_meta = MetadataCatalog.get("coco_2017_val")
coco_data = DatasetCatalog.get("coco_2017_val")

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)


if os.path.exists("co_occurence"):
    shutil.rmtree("co_occurence")
    os.mkdir("co_occurence")
else:
    os.mkdir("co_occurence")

"""
if os.path.exists("co_source"):
    shutil.rmtree("co_source")
    os.mkdir("co_source")
else:
    os.mkdir("co_source")    
"""
    

for img_num in range(500):

    source_img_index = int(np.random.uniform(len(coco_data)))

    while not coco_data[source_img_index]["annotations"]:
        source_img_index = int(np.random.uniform(len(coco_data)))


    source_img_meta = coco_data[source_img_index]

    source_img = cv2.imread(source_img_meta["file_name"])

    sel_obj = int(np.random.uniform(len(source_img_meta["annotations"])-1))

    source_img_mask = source_img_meta["annotations"][sel_obj]["segmentation"]

    if len(source_img_mask) > 1:
        source_img_mask = np.array(source_img_mask[0]).astype(np.int32).reshape(-1,1,2)
    else:
        source_img_mask = np.array(source_img_mask).astype(np.int32).reshape(-1,1,2)

    source_img_height = source_img_meta["height"]
    source_img_width = source_img_meta["width"]


    mask = np.zeros_like(source_img)
    cv2.fillPoly(mask, [source_img_mask], (255,255,255))
    source_obj_pos = np.where(mask == 255)
    source_obj_vals = source_img[np.where(mask == 255)]

    min_x, max_x = source_obj_pos[1].min(), source_obj_pos[1].max()
    min_y, max_y = source_obj_pos[0].min(), source_obj_pos[0].max()

    obj_width = max_x - min_x
    obj_height = max_y - min_y

    if (obj_width*obj_height)/(source_img_height*source_img_width) < 0.05:
        continue
        
    
    for co_idx in range(10):

        try:

            y_pols = source_obj_pos[0]
            new_y_begin = int(np.random.uniform(0+obj_height, source_img_height-obj_height-1))
            new_max_y = (max_y - source_obj_pos[0][0]) + new_y_begin 
            new_min_y = new_y_begin - (source_obj_pos[0][0] - min_y) 

            counter = 0
            while new_max_y >= source_img_height or new_min_y < 0:
                new_y_begin = int(np.random.uniform(0+obj_height, source_img_height-obj_height-1))
                new_max_y = (max_y - source_obj_pos[0][0]) + new_y_begin 
                new_min_y = new_y_begin - (source_obj_pos[0][0] - min_y) 
                counter += 1
                if counter == 10:
                    break

            new_y_pols = np.zeros_like(y_pols)
            new_y_pols[0] = new_y_begin

            for i in range(len(y_pols)-1):
                new_y_pols[i+1] = new_y_pols[i] + y_pols[i+1] - y_pols[i]

            x_pols = source_obj_pos[1]
            new_x_begin = int(np.random.uniform(0+obj_width, source_img_width-obj_width-1))
            new_max_x = (max_x - source_obj_pos[1][0]) + new_x_begin 
            new_min_x = new_x_begin - (source_obj_pos[1][0] - min_x) 

            counter = 0
            while new_max_x >= source_img_width or new_min_x < 0:
                new_x_begin = int(np.random.uniform(0+obj_width, source_img_width-obj_width-1))
                new_max_x = (max_x - source_obj_pos[1][0]) + new_x_begin 
                new_min_x = new_x_begin - (source_obj_pos[1][0] - min_x) 
                counter += 1
                if counter == 10:
                    break

            new_x_pols = np.zeros_like(x_pols)
            new_x_pols[0] = new_x_begin

            for i in range(len(x_pols)-1):
                new_x_pols[i+1] = new_x_pols[i] + x_pols[i+1] - x_pols[i]

            res_img = source_img.copy()

            for x, y, c, val in zip(new_x_pols, new_y_pols, source_obj_pos[2], source_obj_vals):
                res_img[y, x, c] = val
            img_name = source_img_meta["file_name"].split("/")[-1]
            img_name = img_name[:-4] + "_" + str(co_idx) + img_name[-4:]
            #cv2.imwrite(os.path.join("co_occurence", img_name), res_img)
            
            instances = predictor(res_img)
            v = Visualizer(res_img[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
            out_test_img = v.draw_instance_predictions(instances["instances"].to("cpu"))
            cv2.imwrite('co_occurence/{}'.format(img_name), out_test_img.get_image()[:, :, ::-1]) 
            
        except Exception as e:
            print(e)
            
        instances = predictor(source_img)
        v = Visualizer(source_img[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
        out_test_img = v.draw_instance_predictions(instances["instances"].to("cpu"))
        cv2.imwrite('co_occurence/{}'.format(source_img_meta["file_name"].split("/")[-1]), out_test_img.get_image()[:, :, ::-1]) 
        
 


