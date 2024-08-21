import os
import json
import shutil

# FIXME: set data path
image_folder = "./비수도권/원천데이터_0611/1920_1080/raw"
label_folder = "./비수도권/라벨링데이터_1026/1920_1080/m_train_1920_1080_daylight_1"
# image_folder = "./raw_images"
# label_folder = "./raw_labels"
 
output_image_folder = "images"
output_label_folder = "labels"

os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

class_label_map = {
    "3red": 0,
    "3redleft": 1,
    "3yellow": 2,
    "4red": 3,
    "4redleft": 4,
    "4yellow": 5,
    "4greenleft": 6,
    "4green": 7,
}

class_count = {class_name: 0 for class_name in class_label_map}

cnt2 = 0
cnt3 = 0
cnt4 = 0

def determine_class(annotation):
    global cnt2, cnt3, cnt4
    try:
        attributes = annotation["attribute"][0]
        light_count = annotation["light_count"]
        if light_count == "3":
            cnt3 += 1
            if attributes["red"] == "on" and attributes["green"] == "off" and attributes["yellow"] == "off" and attributes["left_arrow"] == "off" and attributes["x_light"] == "off" and attributes["others_arrow"] == "off":
                return "3red"
            elif attributes["red"] == "on" and attributes["green"] == "off" and attributes["yellow"] == "off" and attributes["left_arrow"] == "on" and attributes["x_light"] == "off" and attributes["others_arrow"] == "off":
                return "3redleft"
            elif attributes["yellow"] == "on" and attributes["red"] == "off" and attributes["green"] == "off" and attributes["left_arrow"] == "off" and attributes["x_light"] == "off" and attributes["others_arrow"] == "off":
                return "3yellow"
            elif attributes["yellow"] == "on" and attributes["red"] == "on" and attributes["green"] == "off" and attributes["left_arrow"] == "off" and attributes["x_light"] == "off" and attributes["others_arrow"] == "off":
                return "3redyellow"

        elif light_count == "4":
            cnt4 += 1
            if attributes["red"] == "on" and attributes["green"] == "off" and attributes["yellow"] == "off" and attributes["left_arrow"] == "off" and attributes["x_light"] == "off" and attributes["others_arrow"] == "off":
                return "4red"
            elif attributes["red"] == "on" and attributes["green"] == "off" and attributes["yellow"] == "off" and attributes["left_arrow"] == "on" and attributes["x_light"] == "off" and attributes["others_arrow"] == "off":
                return "4redleft"
            elif attributes["yellow"] == "on" and attributes["red"] == "off" and attributes["green"] == "off" and attributes["left_arrow"] == "off" and attributes["x_light"] == "off" and attributes["others_arrow"] == "off":
                return "4yellow"
            elif attributes["green"] == "on" and attributes["red"] == "off" and attributes["yellow"] == "off" and attributes["left_arrow"] == "off" and attributes["x_light"] == "off" and attributes["others_arrow"] == "off":
                return "4green"
            elif attributes["green"] == "on" and attributes["red"] == "off" and attributes["yellow"] == "off" and attributes["left_arrow"] == "on" and attributes["x_light"] == "off" and attributes["others_arrow"] == "off":
                return "4greenleft"
            elif attributes["green"] == "off" and attributes["red"] == "on" and attributes["yellow"] == "on" and attributes["left_arrow"] == "off" and attributes["x_light"] == "off" and attributes["others_arrow"] == "off":
                return "4redyellow"
            
        elif light_count == "2":
            cnt2 += 1
        return None
    except KeyError as e:
        print(f"KeyError: {e} - annotation: {annotation}")
    return None


def convert_to_yolo_format(box, img_width, img_height):
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

cnt = 0
for filename in os.listdir(label_folder):
    if filename.endswith(".json"):
        with open(os.path.join(label_folder, filename), 'r') as f:
            data = json.load(f)
        
        img_filename = data["image"]["filename"]
        img_width, img_height = data["image"]["imsize"]

        label_lines = []
        for annotation in data["annotation"]:
            if annotation["class"] == "traffic_light" and "type" in annotation and annotation["type"] == "car" and "direction" in annotation and annotation["direction"] == "horizontal":
                cnt += 1
                class_name = determine_class(annotation)
                if class_name and class_name in class_label_map:
                    class_label = class_label_map[class_name]
                    box = annotation["box"]
                    yolo_box = convert_to_yolo_format(box, img_width, img_height)
                    label_lines.append(f"{class_label} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")
                    class_count[class_name] += 1
        
        if label_lines:
            label_filename = os.path.splitext(img_filename)[0] + ".txt"
            with open(os.path.join(output_label_folder, label_filename), 'w') as f:
                f.write("\n".join(label_lines))
            
            src_image_path = os.path.join(image_folder, img_filename)
            dst_image_path = os.path.join(output_image_folder, img_filename)
            if os.path.exists(src_image_path):
                shutil.copyfile(src_image_path, dst_image_path)

for class_name, count in class_count.items():
    print(f"{class_name}: {count}")

print(cnt2, cnt3, cnt4)