#!/usr/bin/env python
# coding: utf-8

# # Import

# In[ ]:


import os
import gc
import json
import yaml
import json
import copy
import pickle
import random
import shutil
import itertools
import typing as tp
from typing import Any, Dict, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from addict import Dict as yaml2dic

from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningDataModule, LightningModule
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar, EarlyStopping, ProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger

from pytorch_metric_learning import losses

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import (
    DualTransform,
    ImageOnlyTransform,
    NoOp,
    to_tuple,
)

from augly.image.functional import overlay_emoji, overlay_image, overlay_text
from augly.image.transforms import BaseTransform
from augly.utils import pathmgr
from augly.utils.base_paths import MODULE_BASE_DIR
from augly.utils.constants import FONT_LIST_PATH, FONTS_DIR, SMILEY_EMOJI_DIR

import timm
import faiss

import warnings
warnings.simplefilter('ignore')


# # Define Functions

# In[ ]:


def read_yaml(fpath):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return yaml2dic(yml)


# ## Dataset

# In[ ]:


class JpoDataset_label(Dataset):
    def __init__(self, cfg, df, base_transform=None, hard_transform=None, mode=False):
        self.path = df["path"].values
        if "label" in df.keys():
            self.label = df["label"].values
        else:
            self.label = None
        if "cite_path" in df.keys():
            self.cite_path = df["cite_path"].values
        else:
            self.cite_path = None
        self._base_transform = base_transform
        self._hard_transform = hard_transform
        self.mode = mode
        self._cfg = cfg

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        if (self.mode=='test')or(self.mode=='pred'):
            # test
            path = self.path[idx]
            if self.mode=='test':
                img = self.__read_image(SETTINGS['PROCESSED_DATA_DIR'] + f'crop_apply_images/{path}')
            else:
                img = self.__read_image(SETTINGS['PROCESSED_DATA_DIR'] + f'crop_cite_images/{path}')
            img = self._base_transform(image=img)['image']
            return {'images':img}
        elif self.mode=='val':
            # train or val
            path, cite_path = self.path[idx], self.cite_path[idx]
            label = self.label[idx]
            img = self.__read_image(SETTINGS['PROCESSED_DATA_DIR'] + f'crop_apply_images/{path}')
            cite_img = self.__read_image(SETTINGS['PROCESSED_DATA_DIR'] + f'crop_cite_images/{cite_path}')
            imgs = [self._base_transform(image=img)['image']] + [self._base_transform(image=cite_img)['image']]
            return {'images':imgs, 'labels':label}
        else:
            # train or val
            path, cite_path = self.path[idx], self.cite_path[idx]
            label = self.label[idx]
            img = self.__read_image(SETTINGS['PROCESSED_DATA_DIR'] + f'crop_apply_images/{path}')
            cite_img = self.__read_image(SETTINGS['PROCESSED_DATA_DIR'] + f'crop_cite_images/{cite_path}')
            imgs = [self._base_transform(image=img)['image']] + [self._base_transform(image=cite_img)['image']]
            for i in range(self._cfg.General.ncrops -2):
                if i%2 ==0:
                    imgs += [self._hard_transform(image=img)['image']]
                else:
                    imgs += [self._hard_transform(image=cite_img)['image']]

            return {'images':imgs, 'labels':label}

    
    def __read_image(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        img = np.array(img)
        # img = img.astype('float32')
        return img


# ## Augmentation

# In[ ]:


import pickle
import random
from typing import Any, Dict, List, Optional

from augly.image.functional import overlay_emoji, overlay_image, overlay_text
from augly.image.transforms import BaseTransform
from augly.utils import pathmgr
from augly.utils.base_paths import MODULE_BASE_DIR
from augly.utils.constants import FONT_LIST_PATH, FONTS_DIR, SMILEY_EMOJI_DIR

from albumentations.core.transforms_interface import (
    DualTransform,
    ImageOnlyTransform,
    NoOp,
    to_tuple,
)

class MakePadTextLogo(ImageOnlyTransform):
    def __init__(
        self,
        opacity: float = 1.0,
        size: int = 256,
        p: float = 1.0,
    ):
        super().__init__(p)
        self.opacity = opacity
        self.size = size

        with open(Path(FONTS_DIR) / FONT_LIST_PATH) as f:
            font_list = [s.strip() for s in f.readlines()]
            blacklist = [
                'TypeMyMusic',
                'PainttheSky-Regular',
            ]
            self.font_list = [
                f for f in font_list
                if all(_ not in f for _ in blacklist)
            ]

        self.font_lens = []
        for ff in self.font_list:
            font_file = Path(MODULE_BASE_DIR) / ff.replace('.ttf', '.pkl')
            with open(font_file, 'rb') as f:
                self.font_lens.append(len(pickle.load(f)))

    def apply(
        self, image: np.array, metadata: Optional[List[Dict[str, Any]]] = None, **params
    ) -> np.array:

        base_image = image.copy()
        h, w, _ = base_image.shape
        
        back_img = np.zeros((self.size, self.size, 3), dtype=image.dtype)
        back_img[:] = 255
        
        i = random.randrange(0, len(self.font_list))
        # i = random.choice([255, 62, 80, 220, 350, 157, 319, 52, 0])
        kwargs = dict(
            font_file=Path(MODULE_BASE_DIR) / self.font_list[i],
            font_size=0,
            # font_size=random.uniform(0.05, 0.2),
            # color=[random.randrange(0, 256) for _ in range(3)],
            color=[0,0,0],
            x_pos=0,
            # x_pos=random.uniform(0.0, 0.75),
            metadata=metadata,
            opacity=self.opacity,
        )

        mode = random.choice(['vertical', 'horizontal'])

        if mode == 'vertical':
            if np.random.rand() > 0.3:
                resize_rate = np.random.rand() * 0.3 + 0.5
                resize_h, resize_w = int(h*resize_rate), int(w*resize_rate)
                resize_img = Image.fromarray(base_image)
                resize_img = resize_img.resize((resize_w, resize_h)) # w,h

                start_w = int((self.size-resize_w)/2)
                use_img = Image.fromarray(back_img)

                text_num = random.randrange(1, 3)

                offset = int(random.uniform(0, 0.2)*self.size)
                img_bottom = (resize_h+offset)/self.size
                kwargs['font_size'] = (1-img_bottom)/text_num
                kwargs['font_size'] = np.min([kwargs['font_size'], 0.2])
                # kwargs['x_pos'] = random.uniform(0.1, 0.5)
                try:
                    for j in range(text_num):
                        if j == 0:
                            y_pos = random.uniform(img_bottom, 1-(kwargs['font_size']*text_num))
                        else:
                            y_pos += kwargs['font_size']
                        text = [random.randrange(0, self.font_lens[i]) for _ in range(random.randrange(5, 15))]
                        kwargs['x_pos'] = np.max([0.5-(len(text)*0.3*kwargs['font_size']), 0])
                        use_img = overlay_text(
                            use_img,
                            text=text,
                            y_pos=y_pos,
                            **kwargs,
                        )
                    use_img = np.array(use_img)
                    use_img[offset:resize_h+offset, start_w:(start_w+resize_w)] = resize_img
                    return use_img
                except OSError:
                    return image
            else:
                resize_rate = np.random.rand() * 0.3 + 0.3
                resize_h, resize_w = int(h*resize_rate), int(w*resize_rate)
                resize_img = Image.fromarray(base_image)
                resize_img = resize_img.resize((resize_w, resize_h)) # w,h

                start_w = int((self.size-resize_w)/2)
                use_img = Image.fromarray(back_img)

                text_num = random.randrange(1, 4)

                offset = int(random.uniform(0, 0.2)*self.size)
                img_up = (resize_h+offset)/self.size
                kwargs['font_size'] = (1-img_up)/text_num*0.8
                kwargs['font_size'] = np.min([kwargs['font_size'], 0.2])
                try:
                    for j in range(text_num):
                        if j == 0:
                            y_pos = random.uniform(0, 1-img_up-(kwargs['font_size']*text_num))
                        else:
                            y_pos += kwargs['font_size']
                        text = [random.randrange(0, self.font_lens[i]) for _ in range(random.randrange(5, 15))]
                        kwargs['x_pos'] = np.max([0.5-(len(text)*0.3*kwargs['font_size']), 0])
                        use_img = overlay_text(
                            use_img,
                            text=text,
                            y_pos=y_pos,
                            **kwargs,
                        )
                    use_img = np.array(use_img)
                    if offset>0:
                        use_img[-(resize_h+offset):-offset, start_w:(start_w+resize_w)] = resize_img
                    else:
                        use_img[-resize_h:, start_w:(start_w+resize_w)] = resize_img
                    return use_img
                except OSError:
                    return image
        else:
            if np.random.rand() > 0.5:
                resize_rate = np.random.rand() * 0.3 + 0.4
                resize_h, resize_w = int(h*resize_rate), int(w*resize_rate)
                resize_img = Image.fromarray(base_image)
                resize_img = resize_img.resize((resize_w, resize_h)) # w,h

                start_h = int((self.size-resize_h)/2)
                use_img = Image.fromarray(back_img)

                text_num = random.randrange(1, 3)

                img_right = resize_w/self.size
                kwargs['font_size'] = random.uniform(0.1, 0.2)
                # kwargs['x_pos'] = random.uniform(0.1, 0.5)
                try:
                    for j in range(text_num):
                        if j == 0:
                            y_pos = random.uniform(0.3, 0.7-(kwargs['font_size']*text_num))
                        else:
                            y_pos += kwargs['font_size']
                        text = [random.randrange(0, self.font_lens[i]) for _ in range(random.randrange(5, 15))]
                        kwargs['x_pos'] = img_right+0.05
                        use_img = overlay_text(
                            use_img,
                            text=text,
                            y_pos=y_pos,
                            **kwargs,
                        )
                    use_img = np.array(use_img)
                    use_img[start_h:(start_h+resize_h), :resize_w] = resize_img
                    return use_img
                except OSError:
                    return image   
            else:
                resize_rate = np.random.rand() * 0.3 + 0.4
                resize_h, resize_w = int(h*resize_rate), int(w*resize_rate)
                resize_img = Image.fromarray(base_image)
                resize_img = resize_img.resize((resize_w, resize_h)) # w,h

                start_h = int((self.size-resize_h)/2)
                use_img = Image.fromarray(back_img)

                text_num = random.randrange(1, 3)

                img_right = resize_w/self.size
                kwargs['font_size'] = random.uniform(0.1, 0.2)
                # kwargs['x_pos'] = random.uniform(0.1, 0.5)
                try:
                    for j in range(text_num):
                        if j == 0:
                            y_pos = random.uniform(0.3, 0.7-(kwargs['font_size']*text_num))
                        else:
                            y_pos += kwargs['font_size']
                        text = [random.randrange(0, self.font_lens[i]) for _ in range(random.randrange(10, 15))]
                        kwargs['x_pos'] = 0.02
                        use_img = overlay_text(
                            use_img,
                            text=text,
                            y_pos=y_pos,
                            **kwargs,
                        )
                    use_img = np.array(use_img)
                    use_img[start_h:(start_h+resize_h), -resize_w:] = resize_img
                    return use_img
                except OSError:
                    return image   

    def get_params(self):
        return {"metadata": []}

    def get_transform_init_args_names(self):
        return ("overlaytext",)


# ## Lightning DataModule

# In[ ]:


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        cfg,
        test_df=None
    ):
        super().__init__()
        self._train_df = train_df
        self._val_df = val_df
        self._cfg = cfg
        self._test_df = test_df

    def __create_dataset(self, mode=None):
        _dataset = eval(self._cfg.Dataset.type)
        if mode=='train':
            base_transform = self.__get_transform(self._cfg.Transform.train.base)
            hard_transform = self.__get_transform(self._cfg.Transform.train.hard)
            dataset = _dataset(self._cfg, self._train_df, base_transform, hard_transform, mode)
        elif mode=='val':
            transform = self.__get_transform(self._cfg.Transform.val)
            dataset = _dataset(self._cfg, self._val_df, transform, transform, mode)
        elif mode=='test':
            transform = self.__get_transform(self._cfg.Transform.test)
            dataset = _dataset(self._cfg, self._test_df, transform, None, mode)
        elif mode=='pred':
            transform = self.__get_transform(self._cfg.Transform.test)
            dataset = _dataset(self._cfg, self._test_df, transform, None, mode)
        return dataset

    def train_dataloader(self):
        dataset = self.__create_dataset(mode='train')
        return DataLoader(dataset, **self._cfg.Dataset.loader.train)

    def val_dataloader(self):
        dataset = self.__create_dataset(mode='val')
        return DataLoader(dataset, **self._cfg.Dataset.loader.val)
    
    def test_dataloader(self):
        dataset = self.__create_dataset(mode='test')
        return DataLoader(dataset, **self._cfg.Dataset.loader.test)
    
    def predict_dataloader(self):
        dataset = self.__create_dataset(mode='pred')
        return DataLoader(dataset, **self._cfg.Dataset.loader.test)

    #ref: https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution
    def __get_transform(self, conf_augmentation):
        def __get_object(trans):
            if trans.name in {"Compose", "OneOf"}:
                if str(trans.member)[:4] == 'list':
                    augs_tmp = eval(trans.member)
                else:
                    augs_tmp = [__get_object(aug) for aug in trans.member]
                return getattr(A, trans.name)(augs_tmp, **trans.params)

            if hasattr(A, trans.name):
                return getattr(A, trans.name)(**trans.params)
            else:
                return eval(trans.name)(**trans.params)

        if conf_augmentation is None:
            return None
        else:
            augs = [__get_object(aug) for aug in conf_augmentation]
            augs.append(ToTensorV2())
            return A.Compose(augs)


# ## Model

# In[ ]:


# 参考：https://github.com/lyakaap/ISC21-Descriptor-Track-1st/blob/6e3a51be54e1aae8f41a1703bc75da7c143b5c53/exp/v83.py#L103
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


# ## Lightning Module

# In[ ]:


class Model(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self._criterion = eval(self._cfg.Loss.name)(**self._cfg.Loss.params)                             if self._cfg.Loss.params                             else eval(self._cfg.Loss.name)()
        if self._cfg.CrossBatchMemory.enable:
            self._criterion = losses.CrossBatchMemory(self._criterion, **self._cfg.CrossBatchMemory.params)        
        self.model = eval(self._cfg.Model.type)(**self._cfg.Model.params)
        self.log_metric = []
        
        
    def forward(self, x):
        out = self.model(x)
        return out
    
        
    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch, 'train')
        return {
            "loss": loss, "labels": labels, "preds": preds
        }
        
    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch, 'val')
        return {
            "loss": loss, "labels": labels, "preds": preds
        }
    
    def predict_step(self, batch, batch_idx):
        feats = batch['features']
        preds = self.forward(feats).squeeze(1)
        return {
            "preds": preds.detach().cpu()
        }
    
    def __share_step(self, batch, mode):
        images, labels = batch['images'], batch['labels']
        images = torch.cat([image for image in images], dim=0)
        if mode=='train':
            labels = torch.tile(labels, dims=(self._cfg.General.ncrops,))
        else:
            labels = torch.tile(labels, dims=(2,))
    
        if self._cfg.General.debug:
            for i in range(10):
                plot_image = images[labels==labels[i]]
                plt.figure(figsize=(10,10))
                for i, img in enumerate(plot_image):
                    plt.subplot(3,3,i+1)
                    plt.imshow(img.detach().cpu().numpy().transpose(1,2,0))
                plt.show()
        
        logits = self.forward(images)
        loss = self._criterion(logits, labels)

        dic = dict()
        dic[f"{mode}_loss/fold_{self._cfg.General.fold}"] = loss.detach().item()
        self.log_dict(dic, prog_bar=True)
        
        return loss, logits.detach().cpu(), labels.detach().cpu()
            

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'val')
        
    def __share_epoch_end(self, outputs, mode):
        preds = []
        labels = []
        for out in outputs:
            pred, label = out['preds'], out['labels']
            preds.append(pred)
            labels.append(label)
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        
        dic = dict()
        dic["epoch"] = self.current_epoch
        dic[f"{mode}_loss/fold_{self._cfg.General.fold}"] = torch.stack([o["loss"] for o in outputs]).detach().mean().item()
        if mode=='val':
            self.log_metric.append(dic[_cfg.Callbacks.ModelCheckpoint.monitor])
            if _cfg.Callbacks.ModelCheckpoint.mode == 'min':
                dic[f"best_{_cfg.Callbacks.ModelCheckpoint.monitor}"] = np.min(self.log_metric)
            else:
                dic[f"best_{_cfg.Callbacks.ModelCheckpoint.monitor}"] = np.max(self.log_metric)
        self.log_dict(dic, prog_bar=True)
        
        
    def configure_optimizers(self):
        optimizer = eval(self._cfg.Optimizer.name)(self.parameters(), **self._cfg.Optimizer.params)
        scheduler = {'scheduler': eval(self._cfg.Scheduler.name)(optimizer, **self._cfg.Scheduler.params),
                    'interval': self._cfg.Scheduler.interval
                    }
        
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        optimizer.step(closure=optimizer_closure)            


# ## inference

# In[ ]:


def to_device(
    tensors: tp.Union[tp.Tuple[torch.Tensor], tp.Dict[str, torch.Tensor]],
    device: torch.device, *args, **kwargs
):
    if isinstance(tensors, tuple):
        return (t.to(device, *args, **kwargs) for t in tensors)
    elif isinstance(tensors, dict):
        return {
            k: t.to(device, *args, **kwargs) for k, t in tensors.items()}
    else:
        return tensors.to(device, *args, **kwargs)


# In[ ]:


def run_inference_loop(model, loader, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = to_device(batch, device)
            x = batch["images"]
            y = model(x)
            pred_list.append(y.detach().cpu().numpy())
            # pred_list.append(y.sigmoid().detach().cpu().numpy())
        
    pred_arr = np.concatenate(pred_list)
    del pred_list
    return pred_arr


# In[ ]:


class FaissKNeighbors:
    def __init__(self, k=20):
        self.index = None
        self.d = None
        self.k = k

    def fit(self, X):
        X = X.copy(order='C')
        self.d = X.shape[1]
        self.index = faiss.IndexFlatIP(self.d)
        # self.index = faiss.IndexFlatL2(self.d)
        self.index.add(X.astype(np.float32))

    def predict(self, X):
        X = X.copy(order='C')
        X = np.reshape(X, (-1, self.d))
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        if X.shape[0] == 1:
            return distances[0], indices[0]
        else:
            return distances, indices


# # Read Config

# In[ ]:


CFG = read_yaml("./config/config_train_swin.yaml")

SETTINGS = json.load(open('settings.json', 'r'))
SETTINGS['MODEL_CHECKPOINT_DIR'] += "swin/"
SETTINGS['LOGS_DIR'] += "swin/"


# # Read Data, Split folds

# In[ ]:


df_train = pd.read_csv(SETTINGS['PROCESSED_DATA_DIR'] + "train_w_group_label.csv")
df_test = pd.read_csv(SETTINGS['RAW_DATA_DIR'] + "test.csv")
df_cite = pd.read_csv(SETTINGS['RAW_DATA_DIR'] + "cite_v2.csv")
smpl_sub = pd.read_csv(SETTINGS['RAW_DATA_DIR'] + "sample_submission.csv")


# ## Make fold

# In[ ]:


train = df_train.copy()
test = df_test.copy()


# In[ ]:


if CFG.General.num_folds>1:
    CV = StratifiedKFold(n_splits=CFG.General.num_folds, shuffle=True, random_state=CFG.General.seed)
    train["fold"] = -1
    for fold_id, (tr_idx, val_idx) in enumerate(CV.split(train["gid"], train["category"], train["gid"])):
        # print(val_idx)
        train.loc[val_idx, "fold"] = fold_id
else:
    train["fold"] = 0


# # Train

# In[ ]:


print('===== Train Start =====')
for fold in range(CFG.General.num_folds):
    
    print(f"[fold {fold}]")
    _cfg = copy.deepcopy(CFG)
    seed_everything(_cfg.General.seed)

    _cfg.General.fold = fold
    _cfg.Callbacks.ModelCheckpoint.filename = f'best_{_cfg.Callbacks.ModelCheckpoint.monitor}_fold_{_cfg.General.fold}'
    _cfg.Callbacks.ModelCheckpoint.dirpath = SETTINGS['MODEL_CHECKPOINT_DIR']
    _cfg.Callbacks.ModelCheckpoint.monitor += f'/fold_{_cfg.General.fold}'
    if 'EarlyStopping' in _cfg.Callbacks.keys():
        _cfg.Callbacks.EarlyStopping.monitor += f'/fold_{_cfg.General.fold}'

    train_df = train[train["fold"] != fold]
    val_df = train[train["fold"] == fold]

    datamodule = DataModule(train_df, val_df, _cfg)
    model = Model(_cfg)

    loggers = []
    loggers.append(CSVLogger(save_dir=SETTINGS['LOGS_DIR'], name=f'fold_{fold}'))

    callbacks = []
    callbacks.append(RichProgressBar())
    callbacks.append(LearningRateMonitor())
    checkpoint_callback = ModelCheckpoint(**_cfg.Callbacks.ModelCheckpoint)
    callbacks.append(checkpoint_callback)
    if 'EarlyStopping' in _cfg.Callbacks.keys():
        callbacks.append(EarlyStopping(**_cfg.Callbacks.EarlyStopping))

    trainer = Trainer(
        max_epochs=_cfg.General.epoch,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=SETTINGS['LOGS_DIR'],
        fast_dev_run=_cfg.General.debug,
        # plugins=DDPPlugin(find_unused_parameters=False),
        **_cfg.Trainer
    )

    trainer.fit(model, datamodule=datamodule)

    shutil.copy(SETTINGS['MODEL_CHECKPOINT_DIR']+'last.ckpt', SETTINGS['MODEL_CHECKPOINT_DIR']+f'last_fold_{fold}.ckpt')
    
    # break


# # Inference OOF & Test

# In[ ]:


test_idx2id = dict(zip(df_test.index, df_test['gid']))
cite_idx2id = dict(zip(df_cite.index, df_cite['gid']))


# In[ ]:


tmp = train.copy()
tmp['path'] = train['cite_path']

train_embeds = []
val_embeds = []

print('===== Make OOF Embeddings =====')

for fold in range(CFG.General.num_folds):
    
    print(f"[fold {fold}]")
    
    val_idx = train.query("fold == @fold").index.values
    
    best_model_path = SETTINGS['MODEL_CHECKPOINT_DIR'] + f'last_fold_{fold}.ckpt'
    model = Model.load_from_checkpoint(best_model_path, cfg=CFG)
    
    # # inference
    datamodule = DataModule(None, None, CFG, train.loc[val_idx])
    val_embed = run_inference_loop(model, datamodule.test_dataloader(), CFG.General.device)
    
    datamodule = DataModule(None, None, CFG, tmp)
    train_embed = run_inference_loop(model, datamodule.predict_dataloader(), CFG.General.device)
    
    val_embeds.append(val_embed)
    train_embeds.append(train_embed)
    # break


# In[ ]:


print('===== OOF Score =====')

for fold in range(CFG.General.num_folds):
    print(f"[fold {fold}]")
    val_idx = train.query("fold == @fold").index.values
    
    kn = FaissKNeighbors()
    kn.fit(train_embeds[fold])
    dist, idx = kn.predict(val_embeds[fold])
    
    cnt = 0
    for n, idx_ in tqdm(enumerate(idx)):
        cites = [train['cite_gid'].iloc[i] for i in idx_]
        if train['cite_gid'].loc[val_idx].iloc[n] in cites:
            cnt += 1
    score = np.round(cnt/len(val_idx), 3)
    print(f'score = {score}, num = {cnt}')
    print()


# In[ ]:


cite_embeds = []
train_embeds = []    

print('===== Make train/cite Embeddings =====')
for fold in range(CFG.General.num_folds):
    print(f"[fold {fold}]")
    
    best_model_path = SETTINGS['MODEL_CHECKPOINT_DIR'] + f'last_fold_{fold}.ckpt'
    model = Model.load_from_checkpoint(best_model_path, cfg=CFG)
    
    # # inference    
    datamodule = DataModule(None, None, CFG, df_cite)
    cite_embed = run_inference_loop(model, datamodule.predict_dataloader(), CFG.General.device)
    
    datamodule = DataModule(None, None, CFG, df_train)
    train_embed = run_inference_loop(model, datamodule.test_dataloader(), CFG.General.device)
    
    if (np.isnan(np.sum(cite_embed)) == False) & (np.isnan(np.sum(train_embed)) == False):
        print(f'fold {fold} Embedding OK')
    else:
        print(f'fold {fold} Embedding contain Nan. Please Retry.')
        
    np.save(SETTINGS['MODEL_CHECKPOINT_DIR'] + f'/cite_embed_{fold}.npy', cite_embed)
    np.save(SETTINGS['MODEL_CHECKPOINT_DIR'] + f'/train_embed_{fold}.npy', train_embed)
    
    cite_embeds.append(cite_embed)
    train_embeds.append(train_embed)


# In[ ]:


cite_mean_embed = np.mean(cite_embeds, axis=0)
train_mean_embed = np.mean(train_embeds, axis=0)


# In[ ]:


print('===== Save train/cite Embeddings =====')
np.save(SETTINGS['MODEL_CHECKPOINT_DIR'] + '/cite_mean_embed.npy', cite_mean_embed)
np.save(SETTINGS['MODEL_CHECKPOINT_DIR'] + '/train_mean_embed.npy', train_mean_embed)


# In[ ]:


print('===== Finish =====')

