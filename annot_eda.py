import os
import json
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import json2data
from utils import get_video_list

DATA_DIR = "D:/해커톤-자갈치_비식별화"

def dict_count(element, dic_one):
    if element in dic_one:
        dic_one[element] += 1
    else:
        dic_one[element] = 1


if __name__ == "__main__":
    dict_gender = dict()
    dict_age = dict()
    dict_top_type = dict()
    dict_top_color = dict()
    dict_bottom_type = dict()
    dict_bottom_color = dict()
    dict_accessories = dict()
    dict_pet = dict()
    list_width = []
    list_height = []
    len_annots = []
    len_persons = []
    cnt_videos = 0

    list_date_fol = os.listdir(DATA_DIR)

    list_video = get_video_list()

    for folder_name, file_name in list_video:
        cnt_annots = 0
        cnt_persons = 0
        cnt_videos += 1
        print(folder_name, file_name)
        json_data = json2data(f"{DATA_DIR}/{folder_name}/{file_name}.json")

        annots = json_data["annotations"]
        categories = json_data["categories"]

        for ann in annots:
            if "person" in ann["id"]:
                dict_count(ann["top_type"], dict_top_type)
                dict_count(ann["top_color"], dict_top_color)
                dict_count(ann["bottom_type"], dict_bottom_type)
                dict_count(ann["bottom_color"], dict_bottom_color)
                dict_count(ann["accessories"], dict_accessories)
                dict_count(ann["pet"], dict_pet)
                list_width.append(ann["bbox"][2]-ann["bbox"][0])
                list_height.append(ann["bbox"][3]-ann["bbox"][1])
                cnt_annots += 1

        for cat in categories:
            if cat["supercategory"] == "person":
                dict_count(cat["gender"], dict_gender)
                dict_count(cat["age"], dict_age)
                cnt_persons += 1

        len_annots.append(cnt_annots)
        len_persons.append(cnt_persons)
    
    print(dict_gender)
    print(dict_age)
    print(dict_top_type)
    print(dict_top_color)
    print(dict_bottom_type)
    print(dict_bottom_color)
    print(dict_accessories)
    print(dict_pet)
    print(cnt_videos)
    print(np.sum(len_annots), np.sum(len_persons))
    print(np.mean(len_annots), np.std(len_annots))
    print(np.mean(len_persons), np.std(len_persons))
                