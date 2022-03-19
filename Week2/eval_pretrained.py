# import some common libraries
import numpy as np
import cv2
import random
import time
import os
import pickle
import pycocotools.mask as mask
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.structures import BoxMode

from kitti_dataset_detectron2 import get_kitti_data
    
    
for d in ["train", "val", "test"]:
    DatasetCatalog.register("kitti_{}".format(d), lambda d = d : get_kitti_data(d))
    MetadataCatalog.get("kitti_{}".format(d)).set(thing_classes=["Car", "Pedestrian"])    
        
kitti_metadata = MetadataCatalog.get("kitti_train")

pre_models = ["COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"]
val_dicts = get_kitti_data("val")
vis_imgs = random.sample(val_dicts, 20)

for pre_model in pre_models:

    cfg = get_cfg()
    model = pre_model
    
    cfg.merge_from_file(model_zoo.get_config_file(model))
    
    cfg.DATASETS.TRAIN = ("kitti_train",)
    cfg.DATASETS.VAL = ('kitti_val',)
    #cfg.DATASETS.TEST = ('kitti_test',)
    cfg.DATASETS.TEST = ()
    

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    t = time.localtime()
    current_time = time.strftime("%H_%M", t)
    model_name = pre_model.split("/")[1].split("_")
    model_name = model_name[0] + model_name[1]
    output_dir = "pretrained_" + model_name + "_" + str(current_time)
    
    os.mkdir(output_dir)

    cfg.OUTPUT_DIR = output_dir
    

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    predictor = DefaultPredictor(cfg)
    
    preds = []
    for i in val_dicts:
    
        im = cv2.imread(i["file_name"])
        instances = predictor(im)
        outputs = instances["instances"].to("cpu")
        outputs.pred_classes = torch.tensor(np.where(outputs.pred_classes == 2, 0, np.where(outputs.pred_classes == 0, 1, -1)))
        outputs = outputs[outputs.pred_classes != -1]
        instances["instances"] = outputs
        preds.append(instances)
        
    evaluator = COCOEvaluator("kitti_val", cfg, True, output_dir=output_dir)
    evaluator.reset()
    evaluator.process(val_dicts, preds)
    results = evaluator.evaluate()

    print(results)
    
    with open(os.path.join(output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)    
    
    os.mkdir(os.path.join(output_dir, "visualization"))
    for i, d in enumerate(vis_imgs):
        im = cv2.imread(d["file_name"])
        instances = predictor(im)
        outputs = instances["instances"].to("cpu")
        outputs.pred_classes = torch.tensor(np.where(outputs.pred_classes == 2, 0, np.where(outputs.pred_classes == 0, 1, -1)))
        outputs = outputs[outputs.pred_classes != -1]
        instances["instances"] = outputs  
        
        v = Visualizer(im[:, :, ::-1], metadata=kitti_metadata, scale=1)
        out_test_img = v.draw_instance_predictions(instances["instances"].to("cpu"))
    
        cv2.imwrite(output_dir + '/visualization/kitti_val_sample_{}.png'.format(i), out_test_img.get_image()[:, :, ::-1])    