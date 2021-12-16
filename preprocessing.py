import cv2
import os
import json
import random
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from utils import init_device_seed, get_video_list

DATA_DIR = "D:/해커톤-자갈치_비식별화"
dict_label = {
    "male": 2, "female": 3, "child": 4, "teenager": 5,
    "adult": 6, "senior": 7, "long_sleeve": 8,  "short_sleeve": 9,
    "sleeveless": 10, "onepice": 11, "top_red": 12, "top_orange": 13,
    "top_yellow": 14, "top_green": 15, "top_blue": 16, "top_purple": 17,
    "top_pink": 18, "top_brown": 19, "top_white": 20, "top_grey": 21,
    "top_black": 22, "long_pants": 23, "short_pants": 24, "skirt": 25,
    "bottom_none": 26, "bottom_red": 27, "bottom_orange": 28, "bottom_yellow": 29,
    "bottom_green": 30, "bottom_blue": 31, "bottom_purple": 32, "bottom_pink": 33,
    "bottom_brown": 34, "bottom_white": 35, "bottom_grey": 36, "bottom_black": 37,
    "acc_carrier": 38, "acc_umbrella": 39, "acc_bag": 40, "acc_hat": 41,
    "acc_glasses": 42, "acc_none": 43, "pet": 44
}

def split_video(list_video):
    list_video2 = list_video.copy()
    random.shuffle(list_video2)
    list_train = list_video2[:int(len(list_video) * 0.8)]
    list_val = list_video2[int(len(list_video) * 0.8):int(len(list_video) * 0.9)]
    list_test = list_video2[int(len(list_video) * 0.9):]

    return list_train, list_val, list_test

def video2frame(file_path):
    vidcap = cv2.VideoCapture(file_path)
    success, image = vidcap.read()
    count = 0
    list_images = []
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        list_images.append(image)
        success,image = vidcap.read()
        count += 1
    return list_images

def json2data(file_path):
    with open(file_path, "r", encoding='UTF-8') as json_f:
        json_data = json.load(json_f)
    return json_data

def preprocess_annot(list_images, json_data, file_name, dataset_type, lines_hot):
    annots = json_data["annotations"]
    categories = json_data["categories"]
    dict_person = dict()

    for cat in categories:
        if cat["supercategory"] == "person":
            dict_person[cat["id"]] = cat

    annots2 = annots.copy()
    random.shuffle(annots2)
    cnt = 0
    for ann in annots2:
        if not "person" in ann["id"]:
            continue

        try:
            frame = ann["frame"]
            bbox = ann["bbox"]
            # print(ann["accessories"], ann["top_color"], ann["bottom_color"])

            hot_code = np.zeros(45, dtype=np.uint8)
            hot_code[dict_label[dict_person[ann["id"]]["gender"]]] = 1
            hot_code[dict_label[dict_person[ann["id"]]["age"]]] = 1
            hot_code[dict_label[ann["top_type"]]] = 1
            hot_code[dict_label["top_" + ann["top_color"]]] = 1
            
            if ann["bottom_type"] == "none":
                hot_code[dict_label["bottom_none"]] = 1
            else:
                hot_code[dict_label[ann["bottom_type"]]] = 1
                hot_code[dict_label["bottom_" + ann["bottom_color"]]] = 1

            hot_code[dict_label["acc_" + ann["accessories"]]] = 1
            hot_code[dict_label["pet"]] = ann["pet"]
            hot_code = hot_code[2:]
            hot_line = " ".join(map(str, list(hot_code)))

            image_name_ext = f"{file_name}_{ann['id']}_{ann['frame']}.jpg"

            # if not os.path.isfile(f"./data/{dataset_type}/{image_name_ext}.jpg"):
            image_crop = list_images[frame][int(bbox[1]):int(bbox[3]+1), int(bbox[0]):int(bbox[2]+1)]
            im = Image.fromarray(image_crop)
            im.save(f"./data/{dataset_type}/{image_name_ext}")

            lines_hot.append(f"{image_name_ext} {hot_line}\n")

        except:
            continue

        cnt += 1
        if cnt == 100:
            break
        # print(hot_code, np.sum(hot_code))

        # plt.imshow(image_crop)
        # plt.show()

    # print(len(annots))
    
    # for ann in annots:
    #     frame = ann["frame"]
    #     bbox = ann["bbox"]

    #     plt.imshow(list_images[frame])
    #     ax = plt.gca()
    #     rect = patches.Rectangle((bbox[0],bbox[1]),
    #              bbox[2]-bbox[0],
    #              bbox[3]-bbox[1],
    #              linewidth=2,
    #              edgecolor='cyan',
    #              fill = False)

    #     ax.add_patch(rect)
    #     plt.show()

def make_dataset(list_type, dataset_type):
    lines_hot = []
    for idx, (folder_name, file_name) in enumerate(list_type):
        # if dataset_type == "train" and idx < 385:
        #     continue
        print(f"{idx}/{len(list_type)} {file_name}")
        list_images = video2frame(f"{DATA_DIR}/{folder_name}/{file_name}.mp4")
        json_data = json2data(f"{DATA_DIR}/{folder_name}/{file_name}.json")
        preprocess_annot(list_images, json_data, file_name, dataset_type, lines_hot)

    f = open(f"./data/{dataset_type}.txt", 'w')
    f.writelines(lines_hot)
    f.close()



if __name__ == "__main__":
    init_device_seed(1234, '0')
    list_date_fol = os.listdir(DATA_DIR)

    list_video = get_video_list()  

    if os.path.isfile("./data/list_types.pkl"):
        with open("./data/list_types.pkl", 'rb') as f: 
            list_types = pickle.load(f)
    else:
        list_types = split_video(list_video)
        with open("./data/list_types.pkl", 'wb') as f:
            pickle.dump(list_types, f)

    make_dataset(list_types[0], "train")
    make_dataset(list_types[1], "val")
    make_dataset(list_types[2], "test")

    






