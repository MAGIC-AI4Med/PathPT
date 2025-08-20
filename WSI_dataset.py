
import pandas as pd
from pathlib import Path

import torch.utils.data as data
import numpy as np
import h5py
import os
import math


class WSI_Data(data.Dataset):
    def __init__(self, 
                 dataset_path=None, 
                 feature_path = None,
                 train_path = None,
                 fold = None,
                 patch_num = None,
                 state=None,
                 train_info = None,
                 val_kwargs=None):
        # Set all input args as attributes
        self.__dict__.update(locals())

        #---->data and label
        self.fold = fold
        self.patch_num = patch_num
        self.feature_dir = feature_path
        self.train_dir = train_path
        self.csv_dir = dataset_path + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, dtype={'train': str, 'val': str, 'test': str})

        #---->order
        # self.shuffle = self.dataset_cfg.data_shuffle

        #---->split dataset
        if state == 'train':
            self.data = self.slide_data.loc[:, 'train'].dropna()
            self.label = self.slide_data.loc[:, 'train_label'].dropna()
        if state == 'val':
            train_data = self.slide_data.loc[:, 'train'].dropna().to_list()
            train_all_data = train_info['train_IDs']
            
            self.data = []
            self.label = []
            for k,v in train_all_data.items():
                for each_slide in v:
                    if each_slide not in train_data:
                        self.data.append(each_slide)
                        self.label.append(int(k)-1)
        if state == 'test':
            self.data = self.slide_data.loc[:, 'test'].dropna()
            self.label = self.slide_data.loc[:, 'test_label'].dropna()
            
        self.num_classes = max(self.label) + 1 # subtype, no normal
        
        #---->segmentation params
        self.wsi_dir_path = val_kwargs['wsi_dir_path']
        self.patch_size = val_kwargs['patch_size']
        self.thd = val_kwargs['thd']
        self.seg_flag = False

    def get_wsi_path(self, slide_id):
        for file in os.listdir(self.wsi_dir_path):
            file_id = os.path.splitext(file)[0]
            if slide_id == file_id and file.endswith(('.svs', '.tif', '.tiff', '.ndpi')):
                return os.path.join(self.wsi_dir_path, file)
        raise FileNotFoundError(f"WSI file for slide ID {slide_id} not found in {self.wsi_dir_path}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = self.data[idx]
        # label = int(self.label[idx])
        wsi_label = int(self.label[idx])
        h5file_path = Path(self.feature_dir) / f'{slide_id}.h5'
        
        with h5py.File(h5file_path, 'r') as f:
            features = f['features'][:]
            coords = f['coords'][:]
        
        # if self.patch_num is not None:
        #     M, D = features.shape
        #     N = self.patch_num
        #     position_indices = np.zeros(N, dtype=np.int64)
            
        #     # Case 1: 特征足够多，随机截取连续块
        #     if M >= N:
        #         start_idx = np.random.randint(0, M - N + 1)
        #         selected_features = features[start_idx:start_idx + N]
        #         position_indices = np.arange(start_idx, start_idx + N)  # 记录原始位置
            
        #     # Case 2: 特征不足，重采样（允许重复）
        #     else:
        #         repeat_times = math.ceil(N / M)
        
        #         # 重复特征形成扩展序列（保持连续性）
        #         extended_features = np.tile(features, (repeat_times, 1))  # 形状: (repeat_times*M, D)
                
        #         # 随机选择起始位置
        #         max_start = extended_features.shape[0] - N
        #         start_idx = np.random.randint(0, max_start + 1)
                
        #         # 截取连续块
        #         selected_features = extended_features[start_idx:start_idx + N]
                
        #         # 计算原始位置（考虑重复）
        #         extended_indices = np.tile(np.arange(M), repeat_times)  # 扩展的索引
        #         position_indices = extended_indices[start_idx:start_idx + N]  # 截取对应的索引


        #     return selected_features, idx, label, position_indices
        
        # else:
        #     position_indices = np.arange(len(features))
        #     return features, idx, label, position_indices
        
        if self.state not in ['train','val']:
            if self.patch_num is not None:
                M, D = features.shape
                N = self.patch_num

                # Case 1: 特征足够多，随机截取连续块
                if M >= N:
                    start_idx = np.random.randint(0, M - N + 1)
                    selected_features = features[start_idx:start_idx + N]
                
                # Case 2: 特征不足，重采样（允许重复）
                else:
                    repeat_times = math.ceil(N / M)
            
                    # 重复特征形成扩展序列（保持连续性）
                    extended_features = np.tile(features, (repeat_times, 1))  # 形状: (repeat_times*M, D)
                    
                    # 随机选择起始位置
                    max_start = extended_features.shape[0] - N
                    start_idx = np.random.randint(0, max_start + 1)
                    
                    # 截取连续块
                    selected_features = extended_features[start_idx:start_idx + N]

                return selected_features, -1, wsi_label, -1
            
            else:
                return features, -1, wsi_label, -1
        else:
            # labeled_h5file_path = str(h5file_path).replace('h5_files', 'h5_with_label')
            assert self.train_dir != None
            # labeled_h5file_path = Path(self.feature_dir) / f'{slide_id}.h5'
            # with h5py.File(labeled_h5file_path, 'r') as f:
            #     features = f['features'][:]
            #     coords = f['coords'][:]
        
            labeled_h5file_path = Path(self.train_dir) / f'{slide_id}.h5'
            with h5py.File(labeled_h5file_path, 'r') as f:
                if 'labels' in f: 
                    patch_labels = f['labels'][:]
                else:
                    patch_labels = None
                    
                # new_features = f['features'][:]
                # print(f'new_features shape: {new_features.shape}')
            
            if self.patch_num is not None:
                M, D = features.shape
                N = self.patch_num

                # Case 1: 特征足够多，随机截取连续块
                if M >= N:
                    start_idx = np.random.randint(0, M - N + 1)
                    selected_features = features[start_idx:start_idx + N]
                    if patch_labels is not None: 
                        selected_labels = patch_labels[start_idx:start_idx + N]
                    else:
                        selected_labels = None
                
                # Case 2: 特征不足，重采样（允许重复）
                else:
                    repeat_times = math.ceil(N / M)
            
                    # 重复特征形成扩展序列（保持连续性）
                    extended_features = np.tile(features, (repeat_times, 1))  # 形状: (repeat_times*M, D)
                    if patch_labels is not None: 
                        extended_labels = np.tile(patch_labels, repeat_times)
                    else:
                        extended_labels = None
                    
                    # 随机选择起始位置
                    max_start = extended_features.shape[0] - N
                    start_idx = np.random.randint(0, max_start + 1)
                    
                    # 截取连续块
                    selected_features = extended_features[start_idx:start_idx + N]
                    
                    if patch_labels is not None: 
                        selected_labels = extended_labels[start_idx:start_idx + N]
                    else:
                        selected_labels = None

                if selected_labels is not None:
                    return selected_features, selected_labels, wsi_label, idx
                else:
                    return selected_features, -1, wsi_label, idx
            
            else:
                if patch_labels is not None:
                    if not self.seg_flag:
                        return features, patch_labels, wsi_label, idx
                    else:
                        # print('dataset')
                        # print(features.shape, coords.shape, patch_labels.shape, wsi_label, slide_id)
                        return features, coords, patch_labels, wsi_label, slide_id
                else:
                    if not self.seg_flag:
                        return features, -1, wsi_label, idx
                    else:
                        return features, coords, -1, wsi_label, slide_id


                            # features, idx, label, position_indices
                