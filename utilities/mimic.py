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
class mimic(Dataset): # <-- 确保继承了 Dataset
    gray_images = True
    task = 'multilabel' 
    num_labels = 14 
    def __init__(self, root, mode='train', transform=None):
        self.root = root
        self.transform = transform
        
        # 标签顺序必须与您的预处理脚本(preprocess_chexclusion_parallel.py)一致
        self.classes = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
        ]
        self.weight = None
        self.norm_weight = None

        # --- 这是修改的核心 ---
        # 这个逻辑符合您只有 train/test 两个 split 的情况
        if mode == 'train':
            print(f"Loading MIMIC train split: train_x.json")
            self.x = load_json(os.path.join(root, 'train_x.json'))
            self.y = load_json(os.path.join(root, 'train_y.json'))
        else: # (mode == 'valid' 或 mode == 'test')
            print(f"Loading MIMIC valid/test split: test_x.json")
            self.x = load_json(os.path.join(root, 'test_x.json'))
            self.y = load_json(os.path.join(root, 'test_y.json'))
        # --- 修改结束 ---

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        # 路径拼接方式
        image_path = os.path.join(self.root, self.x[item])
        
        # 明确加载为灰度图 ('L')
        image = Image.open(image_path).convert('RGB')
        
        label = np.asarray(self.y[item]).astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, label
