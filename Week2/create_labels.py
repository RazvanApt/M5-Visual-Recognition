import os
import pickle

val_split = ["0002", "0006", "0007", "0008", "0010",
             "0013", "0014", "0016", "0018"]

path = "/export/home/mcv/datasets/KITTI-MOTS/instances_txt"
val_annots = {}
train_annots = {}

for txt in os.listdir(path):
    with open(os.path.join(path, txt)) as f:
        img_txt = f.readlines()

        annot_dict = {}

        for annot in img_txt:

            img_dict = {}
            mask = annot.split()

            if mask[0] not in annot_dict.keys():
                annot_dict[mask[0]] = []

            #img_dict["time_frame"] = mask[0]
            img_dict["object_id"] = mask[1]
            img_dict["class_id"] = mask[2]
            img_dict["img_height"] = mask[3]
            img_dict["img_width"] = mask[4]
            img_dict["rle"] = mask[-1]

            annot_dict[mask[0]].append(img_dict)
            
        #print(annot_list)
        
    if txt.split(".")[0] in val_split:
        val_annots[txt.split(".")[0]] = annot_dict
    else:
        train_annots[txt.split(".")[0]] = annot_dict

with open("train_annots.pkl", "wb") as f:
    pickle.dump(train_annots, f)
    
with open("val_annots.pkl", "wb") as f:
    pickle.dump(val_annots, f)