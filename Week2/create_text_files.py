import os 

val_split = ["0002", "0006", "0007", "0008", "0010",
             "0013", "0014", "0016", "0018"]

val_data = []
train_data = []
test_data = []


dataset_path = "/export/home/mcv/datasets/KITTI-MOTS/"

train_path = os.path.join(dataset_path, "training/image_02")
test_path = os.path.join(dataset_path, "testing/image_02")


for folder in os.listdir(train_path):

    if folder in val_split:
        for img in os.listdir(os.path.join(train_path, folder)):
            val_data.append(os.path.join(train_path, folder, img))
    else:
        for img in os.listdir(os.path.join(train_path, folder)):
            train_data.append(os.path.join(train_path, folder, img))

print(test_path)

for folder in os.listdir(test_path):

    for img in os.listdir(os.path.join(test_path, folder)):
        test_data.append(os.path.join(test_path, folder, img))


with open('train_files.txt', 'w') as f:
    for item in train_data:
        f.write("%s\n" % item)

with open('val_files.txt', 'w') as f:
    for item in val_data:
        f.write("%s\n" % item)

with open('test_files.txt', 'w') as f:
    for item in test_data:
        f.write("%s\n" % item)