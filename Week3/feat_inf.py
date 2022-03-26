from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils import visualizer as vis
import numpy as np
import cv2
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

if os.path.exists("feat_inf"):
    shutil.rmtree("feat_inf")
    os.mkdir("feat_inf")
else:
    os.mkdir("feat_inf")

for i in range(1000):

    try:
        source_img_index = int(np.random.uniform(len(coco_data)))
    
        while not coco_data[source_img_index]["annotations"]:
            source_img_index = int(np.random.uniform(len(coco_data)))
    
    
        source_img_meta = coco_data[source_img_index]
    
        source_img = cv2.imread(source_img_meta["file_name"])
    
        sel_obj = int(np.random.uniform(len(source_img_meta["annotations"])-1))
    
        source_img_height = source_img_meta["height"]
        source_img_width = source_img_meta["width"]
    
        x, y, w, h = list(map(lambda x:int(x), source_img_meta["annotations"][sel_obj]["bbox"]))
    
        if (w*h)/(source_img_height*source_img_width) < 0.025:
            continue
            
        instances = predictor(source_img)
        v = Visualizer(source_img[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
        out_test_img = v.draw_instance_predictions(instances["instances"].to("cpu"))
        cv2.imwrite(os.path.join("feat_inf", source_img_meta["file_name"].split("/")[-1]), out_test_img.get_image()[:, :, ::-1]) 
    
        box_inf_img = source_img.copy()
    
        box_mask = np.ones(box_inf_img.shape, dtype=bool)
        box_mask[y:y+h, x:x+w] = False
    
        box_inf_img[box_mask] = 0
    
        img_name = source_img_meta["file_name"].split("/")[-1]
        img_name = img_name[:-4] + "_box" + img_name[-4:]
        
        instances = predictor(box_inf_img)
        v = Visualizer(box_inf_img[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
        out_test_img = v.draw_instance_predictions(instances["instances"].to("cpu"))
        cv2.imwrite(os.path.join("feat_inf", img_name), out_test_img.get_image()[:, :, ::-1]) 
    
    
        seg_inf_img = source_img.copy()
    
        source_img_mask = source_img_meta["annotations"][sel_obj]["segmentation"]
    
        if len(source_img_mask) > 1:
            source_img_mask = np.array(source_img_mask[0]).astype(np.int32).reshape(-1,1,2)
        else:
            source_img_mask = np.array(source_img_mask).astype(np.int32).reshape(-1,1,2)
    
        seg_mask = np.zeros_like(source_img)
        cv2.fillPoly(seg_mask, [source_img_mask], (255,255,255))
        #source_obj_pos = np.where(mask == 255)
        seg_inf_img[seg_mask == 0] = 0
    
        img_name = source_img_meta["file_name"].split("/")[-1]
        img_name = img_name[:-4] + "_seg" + img_name[-4:]
    
        instances = predictor(seg_inf_img)
        v = Visualizer(seg_inf_img[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
        out_test_img = v.draw_instance_predictions(instances["instances"].to("cpu"))
        cv2.imwrite(os.path.join("feat_inf", img_name), out_test_img.get_image()[:, :, ::-1]) 
    
        #cv2.imwrite(os.path.join("feat_inf", img_name), seg_inf_img)
    
        noisy_img = source_img.copy()
        noisy_box = box_inf_img
    
        seg_box_diff = ((seg_mask-255)*255) - box_mask.astype(int)*255
        noisy_img[seg_box_diff == 255] = 0
    
        noisy_img[box_mask] = cv2.randn(noisy_img[box_mask], (127), (127))
    
        img_name = source_img_meta["file_name"].split("/")[-1]
        img_name = img_name[:-4] + "_noisy_random" + img_name[-4:]
    
        instances = predictor(noisy_img)
        v = Visualizer(noisy_img[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
        out_test_img = v.draw_instance_predictions(instances["instances"].to("cpu"))
        cv2.imwrite(os.path.join("feat_inf", img_name), out_test_img.get_image()[:, :, ::-1]) 
        
    except Exception as e:
        print(e)

