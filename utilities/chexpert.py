import json
import os
from glob import glob
from itertools import chain

import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
# 确保这个导入路径相对于您的项目结构是正确的
def load_json(file_path):
    with open(file_path, 'rt') as f:
        return json.load(f)
class chexpert(Dataset):
    """
    Reference: https://github.com/Optimization-AI/ICCV2021_DeepAUC
    """
    gray_images = True
    task = 'multilabel'  # choose from 'binary', 'multiclass' or 'multilabel'
    num_labels = 5  # choose None for binary and multiclass

    def __init__(self,
                 root='',
                 mode='train',
                 transform=None,
                 class_index=-1,
                 use_frontal=True,
                 use_upsampling=False,
                 flip_label=False,
                 verbose=False,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'],
                 **kwargs,
                 ):

        # change test to valid b.c. no test split is reserved.
        # 确定要加载的 .csv 文件的名称
        filename_mode = mode
        if mode == 'test':
            mode = 'valid'  # 内部逻辑使用 'valid'
            filename_mode = 'test' # 但加载 'test.csv'
        elif mode == 'valid':
            # filename_mode = 'valid' # (原始代码) 为 'valid' 模式加载 'val.csv'
            filename_mode = 'test'  # [!! 已修改 !!] 强制 'valid' 模式也加载 'test.csv'

        # load data from csv
        self.classes = train_cols
        self.df = pd.read_csv(os.path.join(root, f'{filename_mode}.csv'))
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '', regex=False)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '', regex=False)
        if filename_mode == 'test':
            self.df['Path'] = self.df['Path'].str.replace('valid/', 'test/', regex=False)
        # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print('Upsampling %s...' % col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        # impute missing values
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)
                # self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            elif col in ['No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 'Lung Lesion', 'Pneumonia',
                         'Pneumothorax', 'Pleural Other', 'Fracture', 'Support Devices']:  # other labels
                self.df[col].replace(-1, 0, inplace=True)
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)

        self._num_images = len(self.df)

        # 0 --> -1
        if flip_label and class_index != -1:  # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)

        # assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert root != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1:  # 5 classes
            if verbose:
                print('Multi-label mode: True, Number of classes: [%d]' % len(train_cols))
                print('-' * 30)
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()

        self.class_index = class_index
        self.transform = transform

        self._images_list = [os.path.join(root, path) for path in self.df['Path'].tolist()]
        if class_index != -1:
            self.targets = self.df[train_cols].values[:, class_index].tolist()
        else:
            self.targets = self.df[train_cols].values.tolist()

        if verbose:
            if class_index != -1:
                if flip_label:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[-1] + self.value_counts_dict[1])
                    if verbose:
                        print('-' * 30)
                        print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1]))
                        print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                        print('-' * 30)
                else:
                    self.imratio = self.value_counts_dict[1] / (self.value_counts_dict[0] + self.value_counts_dict[1])
                    if verbose:
                        print('-' * 30)
                        print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[1], self.value_counts_dict[0]))
                        print('%s(C%s): imbalance ratio is %.4f' % (self.select_cols[0], class_index, self.imratio))
                        print('-' * 30)
            else:
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    try:
                        imratio = self.value_counts_dict[class_key][1] / (
                                    self.value_counts_dict[class_key][0] + self.value_counts_dict[class_key][1])
                    except:
                        if len(self.value_counts_dict[class_key]) == 1:
                            only_key = list(self.value_counts_dict[class_key].keys())[0]
                            if only_key == 0:
                                self.value_counts_dict[class_key][1] = 0
                                imratio = 0  # no postive samples
                            else:
                                self.value_counts_dict[class_key][1] = 0
                                imratio = 1  # no negative samples

                    imratio_list.append(imratio)
                    if verbose:
                        # print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images' % (
                        self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0]))
                        print('%s(C%s): imbalance ratio is %.4f' % (select_col, class_key, imratio))
                        print()
                        # print ('-'*30)
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list

        pos_ratio = np.array(self.targets).mean(axis=0)
        self.weight = np.stack([pos_ratio, 1 - pos_ratio], axis=1)
        self.norm_weight = None

    @property
    def class_counts(self):
        return self.value_counts_dict

    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)

    @property
    def data_size(self):
        return self._num_images

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        image = Image.open(self._images_list[idx])

        if self.transform:
            image = self.transform(image)
        #下面if...else语句中的执行语句相同,因此无需采用if...else语句
        #reshape = -1将数组拉伸成一维数组
        if self.class_index != -1:  # multi-class mode
            label = np.array(self.targets[idx]).reshape(-1).astype(np.float32)
        else:
            label = np.array(self.targets[idx]).reshape(-1).astype(np.float32)

        return image, label