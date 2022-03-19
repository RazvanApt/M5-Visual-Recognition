import cv2
import os
import pickle
import pycocotools.mask as mask

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


def polygonFromMask(maskedArr): # https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    RLEs = mask.frPyObjects(segmentation, maskedArr.shape[0], maskedArr.shape[1])
    RLE = mask.merge(RLEs)
    # RLE = mask.encode(np.asfortranarray(maskedArr))
    #area = mask.area(RLE)
    [x, y, w, h] = cv2.boundingRect(maskedArr)
    
    return segmentation[0], [x, y, w, h]


def get_kitti_data(mode):

    classes = {
        "1": 0,
        "2": 1
    }

    dataset_dicts = []
    with open("{}_files.txt".format(mode), "r") as f:
        imgs = f.readlines()

    with open("{}_annots.pkl".format(mode), "rb") as f:
        annots = pickle.load(f)

    for img in imgs:

        label = {}
        img_info = img.split("/")
        frame_id = str(int(img_info[-1].split(".")[0]))
        
        if frame_id not in annots[img_info[-2]].keys():
            continue

        img_annot = annots[img_info[-2]][frame_id]

        label["file_name"] = img.strip()
        label["image_id"] = img_info[-2] + "/" + frame_id 
        label["height"] = int(img_annot[0]["img_height"])
        label["width"] = int(img_annot[0]["img_width"])

        masks = []
        size = [label["height"], label["width"]]
        for ants in img_annot:
        
            if ants["class_id"] != "10":
                try:
                    segmentation, box = polygonFromMask(mask.decode({"size": size,
                                            "counts": ants["rle"].encode(encoding="UTF-8")
                                            }))
                except Exception as e:
                    print(e)
                    continue
                
                masks.append({
                                "bbox": box,
                                "bbox_mode" : BoxMode.XYWH_ABS,
                                "category_id": classes[ants["class_id"]],
                                #"segmentation": {}
                                "segmentation": [segmentation]
                      })

        label["annotations"] = masks
        dataset_dicts.append(label)
            
    return dataset_dicts