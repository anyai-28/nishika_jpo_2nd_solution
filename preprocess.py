#!/usr/bin/env python
# coding: utf-8

# # Import

# In[ ]:


import os
import json

from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


# # Read Config

# In[ ]:


SETTINGS = json.load(open('settings.json', 'r'))


# # For Table

# In[ ]:


os.makedirs(SETTINGS['PROCESSED_DATA_DIR'], exist_ok=True)


# In[ ]:

print('===== Make Table =====')
df_train = pd.read_csv(SETTINGS['RAW_DATA_DIR'] + "train.csv")


# In[ ]:


dic_id2label = {}
for i, (r_id, c_id) in tqdm(enumerate(zip(df_train['gid'].values, df_train['cite_gid'].values))):
    label = None
    try:
        label = dic_id2label[r_id]
    except:
        pass
    try:
        label = dic_id2label[c_id]
    except:
        pass
    
    if label:
        dic_id2label[r_id] = label
        dic_id2label[c_id] = label
    else:
        dic_id2label[r_id] = i
        dic_id2label[c_id] = i


# In[ ]:


df_train['label'] = df_train['gid']
df_train['label'] = df_train['label'].map(dic_id2label)


# In[ ]:


df_train.to_csv(SETTINGS['PROCESSED_DATA_DIR'] + 'train_w_group_label.csv', index=False)


# # For Image

# In[ ]:

print('===== Make Image =====')
def crop_image(path):

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    img2 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    x1 = [] #x座標の最小値
    y1 = [] #y座標の最小値
    x2 = [] #x座標の最大値
    y2 = [] #y座標の最大値
    for i in range(1, len(contours)):
        ret = cv2.boundingRect(contours[i])
        x1.append(ret[0])
        y1.append(ret[1])
        x2.append(ret[0] + ret[2])
        y2.append(ret[1] + ret[3])

    if x1:
        x1_min = min(x1)
        y1_min = min(y1)
        x2_max = max(x2)
        y2_max = max(y2)
        crop_img = img[y1_min:y2_max, x1_min:x2_max]
        # crop後の画像が小さすぎる場合は、cropしない
        if np.min([crop_img.shape[0],crop_img.shape[1]]) < 50:
            if (crop_img.shape[0]*crop_img.shape[1]) < (img.shape[0]*img.shape[1]*0.1):
                crop_img = img
    else:
        crop_img = img

    return crop_img


def make_crop(input):
    input_path, output_path, output_dir = input
    crop_img = crop_image(input_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(output_path, crop_img)
    return


# In[ ]:

print('===== Train Data =====')
df = pd.read_csv(SETTINGS['RAW_DATA_DIR'] + "train.csv")

paths = df['path'].values
base_input_path = SETTINGS['RAW_DATA_DIR'] + 'apply_images/'
input_paths = [base_input_path+path for path in paths]
output_paths = []
inputs = []
for input_path in tqdm(input_paths):
    output_path = SETTINGS['PROCESSED_DATA_DIR'] + f'crop_apply_images/' + '/'.join(input_path.split('/')[-2:])
    output_dir = SETTINGS['PROCESSED_DATA_DIR'] + f'crop_apply_images/' + input_path.split('/')[-2]
    output_paths.append(output_path)
    inputs.append([input_path, output_path, output_dir])
_ = Parallel(n_jobs=-1)(delayed(make_crop)(input) for input in tqdm(inputs)) # 並列処理


# In[ ]:

print('===== Test Data =====')
test_df = pd.read_csv(SETTINGS['RAW_DATA_DIR'] + "test.csv")

paths = test_df['path'].values
base_input_path = SETTINGS['RAW_DATA_DIR'] +'apply_images/'
input_paths = [base_input_path+path for path in paths]
output_paths = []
inputs = []
for input_path in tqdm(input_paths):
    output_path = SETTINGS['PROCESSED_DATA_DIR'] + f'crop_apply_images/' + '/'.join(input_path.split('/')[-2:])
    output_dir = SETTINGS['PROCESSED_DATA_DIR'] + f'crop_apply_images/' + input_path.split('/')[-2]
    output_paths.append(output_path)
    inputs.append([input_path, output_path, output_dir])
_ = Parallel(n_jobs=-1)(delayed(make_crop)(input) for input in tqdm(inputs)) # 並列処理


# In[ ]:

print('===== Cite Data =====')
cite_df = pd.read_csv(SETTINGS['RAW_DATA_DIR'] + "cite_v2.csv")

paths = cite_df['path'].values
base_input_path = SETTINGS['RAW_DATA_DIR'] +'cite_images/'
input_paths = [base_input_path+path for path in paths]
output_paths = []
inputs = []
for input_path in tqdm(input_paths):
    output_path = SETTINGS['PROCESSED_DATA_DIR'] + f'crop_cite_images/' + '/'.join(input_path.split('/')[-2:])
    output_dir = SETTINGS['PROCESSED_DATA_DIR'] + f'crop_cite_images/' + input_path.split('/')[-2]
    output_paths.append(output_path)
    inputs.append([input_path, output_path, output_dir])
_ = Parallel(n_jobs=-1)(delayed(make_crop)(input) for input in tqdm(inputs)) # 並列処理

print('===== Finish =====')
# In[ ]:




