#!/usr/bin/env python
# coding: utf-8

# # Import

# In[ ]:


import os
import gc
import json
import yaml
import copy
import typing as tp
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import cv2

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningDataModule, LightningModule

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

from cuml.neighbors import NearestNeighbors

import warnings
warnings.simplefilter('ignore')


# # Config

# In[ ]:


IMG_SIZE = 384

SETTINGS = json.load(open('settings.json', 'r'))
if SETTINGS["USE_PRETRAIN"].lower()=='true':
    MODEL_DIR = SETTINGS["PRETRAINED_MODEL_CHECKPOINT_DIR"]
else:
    MODEL_DIR = SETTINGS["MODEL_CHECKPOINT_DIR"]
DEVICE = SETTINGS['SUBMISSION_DEVICE']
METRIC = 'euclidean'

print(f'use model dir = {MODEL_DIR}')


# # Define Function

# ## preprocess

# In[ ]:


def preprocess(path):

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


# ## Transform

# In[ ]:


transform = A.Compose([
    A.LongestMaxSize(p=1.0, max_size=IMG_SIZE),
    A.PadIfNeeded(p=1.0, min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=[255, 255, 255]),
    A.Affine(p=1.0, scale=0.9, cval=[255, 255, 255]),
    A.Normalize(p=1.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# ## Model

# In[ ]:


class ImageEmbeddingModel(nn.Module):
    
    def __init__(
        self, base_name: str, fc_dim: int,
        pretrained=False, in_channels: int=3, fc_drop=0.0, use_bn=False, drop_path_rate=0.0
    ):
        """Initialize"""
        self.base_name = base_name
        super(ImageEmbeddingModel, self).__init__()
        
        # # prepare backbone
        if hasattr(timm.models, base_name):
            base_model = timm.create_model(
                base_name, num_classes=0, pretrained=pretrained, in_chans=in_channels, drop_path_rate=drop_path_rate)
            in_features = base_model.num_features
            print("load imagenet pretrained:", pretrained)
        else:
            raise NotImplementedError

        self.backbone = base_model
        print(f"{base_name}: {in_features}")
        
        self.fc = nn.Linear(in_features, fc_dim, bias=False)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.use_bn = use_bn

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.fc(x)
        if self.use_bn:
            x = self.bn(x)
        x = F.normalize(x)
        return x


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()
        self.dims = dims

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)
        self.head = nn.Identity()

        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        

@register_model
def convnext_base(pretrained=False, in_22k=False, img_size=224, **kwargs):
    num_classes = 21841 if in_22k else 1000
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=num_classes,**kwargs)
    if pretrained:
        if img_size==224:
            url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        else:
            url = model_urls['convnext_base_22k_384']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, img_size=224, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


# In[ ]:


class ConvNextEmbeddingModel(nn.Module):
    
    def __init__(
        self, base_name: str, fc_dim: int, in_22k=False, img_size=384,
        pretrained=False, drop_path_rate=0.0, use_bn=False
    ):
        """Initialize"""
        self.base_name = base_name
        super(ConvNextEmbeddingModel, self).__init__()
        
        # # prepare backbone
        base_model = eval(base_name)(pretrained=pretrained,
                                     drop_path_rate=drop_path_rate,
                                     in_22k=in_22k,
                                     img_size=img_size)
        in_features = base_model.dims[-1]
        self.backbone = base_model
        print(f"{base_name}: {in_features}")
        
        self.fc = nn.Linear(in_features, fc_dim, bias=False)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.use_bn = use_bn

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone.forward_features(x)
        x = self.fc(x)
        if self.use_bn:
            x = self.bn(x)
        x = F.normalize(x)
        return x


# In[ ]:


class Model(LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        out = self.model(x)
        return out


# ## Get Embedding

# In[ ]:


def get_image_embeddings(query_image, dic_models, transform):
    query_image = transform(image=query_image)['image']
    query_image = query_image.to(DEVICE)
    query_image = query_image.unsqueeze(0)
    
    cat_embeds = []
    for _, models in dic_models.items():
        embed_list = []
        for model in models:
            # model.to(DEVICE)
            # model.eval()
            with torch.no_grad():
                embed = model(query_image)
                embed_list.append(embed.detach().cpu().numpy())
        cat_embeds.append(np.mean(embed_list, axis=0))
        del embed_list
        gc.collect()
        torch.cuda.empty_cache()
        
    del query_image
    gc.collect()
    torch.cuda.empty_cache()
    
    return np.concatenate(cat_embeds, axis=1)


# ## KNN

# In[ ]:


def knn_w_postprocess(query_image_embedding, cite_embeddings, train_embeddings, kn_cite, kn_train, df_train, df_cite):
    
    _, pred_indexs = kn_cite.kneighbors(np.reshape(query_image_embedding, (1, -1)))
    
    _, sim_train_index = kn_train.kneighbors(np.reshape(query_image_embedding, (1, -1)))
    sim_train_cite_gid = df_train.at[sim_train_index[0][0], 'cite_gid']

    label = df_train.at[sim_train_index[0][0], 'label']
    sim_train_cite_gids = df_train['cite_gid'][df_train['label']==label].unique()
    
    num = len(sim_train_cite_gids)
    
    pred_indexs = pred_indexs[0][:(15-num+1)]
    pred_indexs = np.append(pred_indexs, df_cite['gid'][df_cite['gid'].isin(sim_train_cite_gids)].index) # trainのciteを追加

    tmp_dsts = []
    tmp_idxs = []
    for sim_train_cite_gid in sim_train_cite_gids:
        sim_cite_index = df_cite[df_cite['gid']==sim_train_cite_gid].index[0]

        aug_embedding = cite_embeddings[sim_cite_index] + (query_image_embedding-train_embeddings[sim_train_index[0][0]])
        aug_embedding = np.reshape(aug_embedding, (1, -1))
        aug_embedding /= np.linalg.norm(aug_embedding, axis=1, keepdims=True)
        tmp_dst, tmp_idx = kn_cite.kneighbors(aug_embedding)
        tmp_dsts.extend(tmp_dst[0])
        tmp_idxs.extend(tmp_idx[0])
        pred_indexs = np.append(pred_indexs, tmp_idx[0][0]) # 一番似ていたidを追加

    sort_idx = np.argsort(tmp_dsts)
    pred_indexs = np.append(pred_indexs, [tmp_idxs[i] for i in sort_idx]) # 類似度が高かった順に追加
    pred_indexs = sorted(set(pred_indexs), key=pred_indexs.tolist().index) # 順序を保持したまま重複削除
    pred_indexs = pred_indexs[:20]
    
    return pred_indexs


# ## get_result

# In[ ]:


def get_result(query_image_path, dic_models, transform, cite_embeddings, train_embeddings, kn_cite, kn_train, df_train, df_cite):
    query_image = preprocess(query_image_path)
    query_image_embedding = get_image_embeddings(query_image, dic_models, transform)
    result_indexs = knn_w_postprocess(query_image_embedding, cite_embeddings, train_embeddings, kn_cite, kn_train, df_train, df_cite)
    return result_indexs


# # Data Load

# ## Model

# In[ ]:


print('===== Load Models =====')

dic_models = {}
dic_model_paths = {
                    'convnext':[
                        MODEL_DIR + 'convnext/last_fold_0.ckpt',
                        MODEL_DIR + 'convnext/last_fold_1.ckpt',
                        MODEL_DIR + 'convnext/last_fold_2.ckpt',
                        MODEL_DIR + 'convnext/last_fold_3.ckpt',
                        MODEL_DIR + 'convnext/last_fold_4.ckpt'
                    ],
                    'swin':[
                        MODEL_DIR + 'swin/last_fold_0.ckpt',
                        MODEL_DIR + 'swin/last_fold_1.ckpt',
                        MODEL_DIR + 'swin/last_fold_2.ckpt',
                        MODEL_DIR + 'swin/last_fold_3.ckpt',
                        MODEL_DIR + 'swin/last_fold_4.ckpt'
                    ],
                  }

for k, model_paths in dic_model_paths.items():
    models = []
    if k == 'swin':
        for path in model_paths:
            model = ImageEmbeddingModel('swin_large_patch4_window12_384', 512, use_bn=True, pretrained=False)
            model = Model.load_from_checkpoint(path, model=model, map_location='cpu')
            model.to(DEVICE)
            model.eval()
            models.append(model)
        dic_models['swin'] = models
    else:
        for path in model_paths:
            model = ConvNextEmbeddingModel('convnext_base', 512, img_size=IMG_SIZE, pretrained=False)
            model = Model.load_from_checkpoint(path, model=model, map_location='cpu', strict=False)
            model.to(DEVICE)
            model.eval()
            models.append(model)
        dic_models['convnext'] = models


# ## Embeddings

# In[ ]:


print('===== Load Embeddings =====')


# In[ ]:


cite_embeddings = []
cite_embeddings.append(np.load(MODEL_DIR + 'convnext/cite_mean_embed.npy'))
cite_embeddings.append(np.load(MODEL_DIR + 'swin/cite_mean_embed.npy'))
cite_embeddings = np.concatenate(cite_embeddings, axis=1)


# In[ ]:


train_embeddings = []
train_embeddings.append(np.load(MODEL_DIR + 'convnext/train_mean_embed.npy'))
train_embeddings.append(np.load(MODEL_DIR + 'swin/train_mean_embed.npy'))
train_embeddings = np.concatenate(train_embeddings, axis=1)


# ## KNN

# In[ ]:


print('===== Set KNN =====')

kn_cite = NearestNeighbors(n_neighbors=20, metric=METRIC)
kn_train = NearestNeighbors(n_neighbors=20, metric=METRIC)

kn_cite.fit(cite_embeddings)
kn_train.fit(train_embeddings)


# ## DataFrame

# In[ ]:


df_train = pd.read_csv(SETTINGS['RAW_DATA_DIR'] + 'train.csv')
df_cite = pd.read_csv(SETTINGS['RAW_DATA_DIR'] + 'cite_v2.csv')
df_test = pd.read_csv(SETTINGS['RAW_DATA_DIR'] + 'test.csv')


# In[ ]:


dic_id2label = {}
for i, (r_id, c_id) in enumerate(zip(df_train['gid'].values, df_train['cite_gid'].values)):
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


cite_idx2id = dict(zip(df_cite.index, df_cite['gid']))


# # Make Sub

# In[ ]:


print('===== Make Submit =====')
pred = []
for p in tqdm(df_test['path'].values):
    query_image_path = SETTINGS['RAW_DATA_DIR'] + f"apply_images/{p}"
    result_indexs = get_result(query_image_path, dic_models, transform, cite_embeddings, train_embeddings, kn_cite, kn_train, df_train, df_cite)
    result = ' '.join([str(cite_idx2id[i]) for i in result_indexs])
    pred.append(result)


# In[ ]:


smpl_sub = pd.read_csv(SETTINGS['RAW_DATA_DIR'] + "sample_submission.csv")
sub_df = smpl_sub.copy()
sub_df['cite_gid'] = pred
sub_df.head()


# In[ ]:


os.makedirs(SETTINGS['SUBMISSION_DIR'], exist_ok=True)
sub_df.to_csv(SETTINGS['SUBMISSION_DIR'] + 'submit.csv', index=False)


# In[ ]:


print('===== Finish =====')

