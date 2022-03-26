# import some common libraries
import numpy as np
import os
import cv2


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

cfg = get_cfg()


cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

if not os.path.exists("ooc_preds"):
    os.mkdir("ooc_preds")
ooc_path = "/export/home/mcv/datasets/out_of_context/"

for i in os.listdir(ooc_path):
    
    im = cv2.imread(os.path.join(ooc_path, i))
    instances = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
    out_test_img = v.draw_instance_predictions(instances["instances"].to("cpu"))
    cv2.imwrite('ooc_preds/{}'.format(i), out_test_img.get_image()[:, :, ::-1])  