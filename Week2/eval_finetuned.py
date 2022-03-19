import cv2
import os
import pickle
import pycocotools.mask as mask
import random
import time

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.structures import BoxMode

from kitti_dataset_detectron2 import get_kitti_data
from LossEvalHook import *


for d in ["train", "val", "test"]:
    DatasetCatalog.register("kitti_{}".format(d), lambda d = d : get_kitti_data(d))
    MetadataCatalog.get("kitti_{}".format(d)).set(thing_classes=["Car", "Pedestrian"])    
        
kitti_metadata = MetadataCatalog.get("kitti_train")

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.VAL[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks


cfg = get_cfg()
model = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
# model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
#model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

cfg.merge_from_file(model_zoo.get_config_file(model))

cfg.DATASETS.TRAIN = ("kitti_train",)
cfg.DATASETS.VAL = ('kitti_val',)
#cfg.DATASETS.TEST = ('kitti_test',)
cfg.DATASETS.TEST = ()
#cfg.TEST.EVAL_PERIOD = 500


cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
cfg.SOLVER.MAX_ITER = 2000
cfg.SOLVER.BASE_LR = 0.005

t = time.localtime()
current_time = time.strftime("%H_%M", t)
model_name = model.split("/")[1].split("_")
model_name = model_name[0] + model_name[1]
output_dir = "finetuned_" + model_name + "_" + str(current_time)

os.mkdir(output_dir)
os.mkdir(os.path.join(output_dir, "inference"))
cfg.OUTPUT_DIR = output_dir

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
val_dicts = get_kitti_data("val")

predictor = DefaultPredictor(cfg)

os.mkdir(os.path.join(output_dir, "visualization"))
for i, d in enumerate(random.sample(val_dicts, 50)):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  
    v = Visualizer(im[:, :, ::-1], metadata=kitti_metadata, scale=0.75)
    out_test_img = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imwrite(output_dir + '/visualization/kitti_val_sample_{}.png'.format(i), out_test_img.get_image()[:, :, ::-1])


evaluator = COCOEvaluator("kitti_val", cfg, True, output_dir=output_dir)
val_loader = build_detection_test_loader(cfg, "kitti_val")
inference_results = inference_on_dataset(trainer.model, val_loader, evaluator)
print(inference_results)

with open(os.path.join(output_dir, "inference", "metrics.pkl"), "wb") as f:
    pickle.dump(inference_results, f)
