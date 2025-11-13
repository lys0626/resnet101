import json
import os
from glob import glob
from itertools import chain
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
def load_json(file_path):
    with open(file_path, 'rt') as f:
        return json.load(f)
class nihchest(Dataset):
    gray_images = True
    task = 'multilabel'
    num_labels = 14

    def __init__(self, root='', mode='train', transform=None):

        self.root = root
        self.transform = transform

        # 1. 加载主 CSV 文件
        df_all = pd.read_csv(os.path.join(self.root, 'Data_Entry_2017.csv'))

        # 2. 确定并读取划分文件
        if mode == 'train':
            split_filename = 'train_list.txt'
        elif mode == 'valid':
            split_filename = 'val_list.txt'
        elif mode == 'test':
            split_filename = 'test_list.txt'
        else:
            raise ValueError(f"不支持的 mode: {mode}")

        split_filepath = os.path.join(self.root, split_filename)
        try:
            with open(split_filepath, 'rt') as f:
                image_indices_in_split = {x.strip('\n') for x in f.readlines()}
        except FileNotFoundError:
            print(f"错误：未找到文件 {split_filepath}。")
            raise

        # 3. 筛选 DataFrame
        df = df_all[df_all['Image Index'].isin(image_indices_in_split)].copy()
        print(f"为模式 '{mode}' 加载了 {len(df)} 条记录 (从 {split_filename})")

        # 4. 构建图像路径 (核心修改：模仿之前能跑通的代码，主要查找 img_384)
        #    假设之前的代码能跑通是因为文件在 img_384 且扩展名为 png
        #    如果扩展名是 jpg，请将下面的 '.png' 改为 '.jpg'
        image_folder = 'img_384' # 主要查找这个文件夹
        image_extension = '.png' # 假设扩展名是 png (如果不是请修改!)

        img_paths = {}
        missing_files = []
        print(f"正在 {os.path.join(self.root, image_folder)} 中查找 {image_extension} 文件...")
        for img_index in df['Image Index']:
             # 直接构建期望的路径
             expected_path = os.path.join(self.root, image_folder, img_index)
             # 检查文件是否存在 (比 glob 更直接)
             if os.path.exists(expected_path):
                 img_paths[img_index] = expected_path
             else:
                 # 如果在 img_384 找不到，作为备选，尝试 images_0XX (假设扩展名也相同)
                 found_in_subfolder = False
                 for i in range(12):
                      subfolder_path = os.path.join(self.root, f'images_{i+1:03}', img_index)
                      if os.path.exists(subfolder_path):
                           img_paths[img_index] = subfolder_path
                           found_in_subfolder = True
                           break # 找到就停止搜索子目录
                 if not found_in_subfolder:
                      missing_files.append(img_index)
                      # print(f"警告：无法在 {image_folder} 或 images_0XX 中找到文件 {img_index}") # 可以取消注释以查看具体哪个文件找不到

        if missing_files:
             print(f"警告：总共有 {len(missing_files)} 个图像文件未在预期位置 ({image_folder} 或 images_0XX) 找到。")

        df['path'] = df['Image Index'].map(img_paths.get)

        # 移除路径未找到的行
        original_count = len(df)
        df = df.dropna(subset=['path'])
        if len(df) < original_count:
             print(f"警告：因缺少图像文件而最终删除了 {original_count - len(df)} 行 ({mode} 模式)。")

        # 5. 处理标签 (保持不变)
        df['Finding Labels'] = df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
        all_labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
        all_labels = [x for x in all_labels if len(x) > 0]
        self.classes = all_labels

        for c_label in self.classes:
            if len(c_label) > 1:
                df[c_label] = df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

        # 6. 设置最终数据 (保持不变)
        self.x = df['path'].values.tolist()
        self.y = df[self.classes].values.astype(np.float32)

        # 7. 计算权重 (保持不变)
        if len(self.y) > 0:
            pos_counts = np.sum(self.y, axis=0)
            neg_counts = len(self.y) - pos_counts
            weight_pos = np.divide(1.0, pos_counts, out=np.zeros_like(pos_counts, dtype=float), where=pos_counts!=0)
            weight_neg = np.divide(1.0, neg_counts, out=np.zeros_like(neg_counts, dtype=float), where=neg_counts!=0)
            self.weight = np.stack([weight_neg, weight_pos], axis=1)

            norm_denom = np.sqrt(np.sum(np.sum(self.y, axis=0)**2))
            self.norm_weight = np.divide(np.sum(self.y, axis=0), norm_denom, out=np.zeros_like(np.sum(self.y, axis=0), dtype=float), where=norm_denom!=0)
        else:
             print(f"警告：未找到模式 '{mode}' 的有效数据。无法计算权重。")
             self.weight = np.ones((len(self.classes), 2)) if self.classes else np.array([])
             self.norm_weight = np.zeros(len(self.classes)) if self.classes else np.array([])

    def get_number_classes(self):
        # 使用您在类顶部定义的 num_labels
        return self.num_labels
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # (保持不变)
        img_path = self.x[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.y[idx]
            filename = os.path.basename(img_path)
            if self.transform:
                image = self.transform(image)
            data = {'image': image, 'target': label, 'name': filename}
            return data
        except FileNotFoundError:
             print(f"错误：图像文件未找到 {img_path}。跳过。")
             next_idx = (idx + 1) % len(self.x) if len(self.x) > 0 else 0
             if len(self.x) > 0: return self.__getitem__(next_idx)
             else: raise IndexError("数据集为空或无法加载任何图像")
        except Exception as e:
            print(f"加载图像 {img_path} 时出错：{e}。跳过。")
            next_idx = (idx + 1) % len(self.x) if len(self.x) > 0 else 0
            if len(self.x) > 0: return self.__getitem__(next_idx)
            else: raise IndexError("数据集为空或无法加载任何图像")