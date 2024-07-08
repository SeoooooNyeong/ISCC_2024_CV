import os
import shutil
import random

input_image_folder = "images" 
input_label_folder = "labels"  
output_train_image_folder = "dataset/train/images"  
output_val_image_folder = "dataset/val/images" 
output_train_label_folder = "dataset/train/labels"  
output_val_label_folder = "dataset/val/labels" 

os.makedirs(output_train_image_folder, exist_ok=True)
os.makedirs(output_val_image_folder, exist_ok=True)
os.makedirs(output_train_label_folder, exist_ok=True)
os.makedirs(output_val_label_folder, exist_ok=True)

class_label_map = {
    0: "3red",
    1: "3redleft",
    2: "3yellow",
    3: "4red",
    4: "4redleft",
    5: "4yellow",
    6: "4greenleft",
    7: "4green",
}

class_files = {class_name: [] for class_name in class_label_map.values()}

for label_filename in os.listdir(input_label_folder):
    if label_filename.endswith(".txt"):
        with open(os.path.join(input_label_folder, label_filename), 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.split()[0])
                class_name = class_label_map[class_id]
                class_files[class_name].append(label_filename)

train_count = 1000
val_count = 100

for class_name, files in class_files.items():
    random.shuffle(files)
    
    if len(files) < train_count + val_count:
        train_files = files[:len(files) - 50]
        val_files = files[len(files) - 50:]
    else:
        train_files = files[:train_count]
        val_files = files[train_count:train_count + val_count]

    for file in train_files:
        image_filename = file.replace(".txt", ".jpg")
        shutil.copyfile(os.path.join(input_image_folder, image_filename), os.path.join(output_train_image_folder, image_filename))
        shutil.copyfile(os.path.join(input_label_folder, file), os.path.join(output_train_label_folder, file))

    for file in val_files:
        image_filename = file.replace(".txt", ".jpg")
        shutil.copyfile(os.path.join(input_image_folder, image_filename), os.path.join(output_val_image_folder, image_filename))
        shutil.copyfile(os.path.join(input_label_folder, file), os.path.join(output_val_label_folder, file))

