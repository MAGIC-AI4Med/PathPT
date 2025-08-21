import pandas as pd
from pathlib import Path
import torch.utils.data as data
import numpy as np
import h5py
import math



class WSI_Data(data.Dataset):
    def __init__(self, 
                 dataset_path=None, 
                 feature_path = None,
                 fold = None,
                 patch_num = None,
                 state=None,
                 train_info = None):
        # Set all input args as attributes
        self.__dict__.update(locals())

        # data and label
        self.fold = fold
        self.patch_num = patch_num
        self.feature_dir = feature_path
        self.csv_dir = dataset_path + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, dtype={'train': str, 'val': str, 'test': str})

        # order
        # self.shuffle = self.dataset_cfg.data_shuffle

        # split dataset
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
            
        self.num_classes = max(self.label) + 1 # subtypes, no normal


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = self.data[idx]
        wsi_label = int(self.label[idx])
        h5file_path = Path(self.feature_dir) / f'{slide_id}.h5'
        
        with h5py.File(h5file_path, 'r') as f:
            features = f['features'][:]
            # coords = f['coords'][:]
        
        if self.state not in ['train','val']: # test
            if self.patch_num is not None:
                M, D = features.shape
                N = self.patch_num

                # Case 1: enough features, randomly select a continuous block
                if M >= N:
                    start_idx = np.random.randint(0, M - N + 1)
                    selected_features = features[start_idx:start_idx + N]
                
                # Case 2: not enough features, resample with replacement
                else:
                    repeat_times = math.ceil(N / M)
            
                    # Repeat features to form an extended sequence (maintaining continuity)
                    extended_features = np.tile(features, (repeat_times, 1))  # shape: (repeat_times*M, D)
                    
                    # Randomly select a starting position
                    max_start = extended_features.shape[0] - N
                    start_idx = np.random.randint(0, max_start + 1)
                    
                    # Select a continuous block
                    selected_features = extended_features[start_idx:start_idx + N]

                return selected_features, -1, wsi_label, -1
            
            else:
                return features, -1, wsi_label, -1
            
        else: # train or val
            if self.patch_num is not None:
                M, D = features.shape
                N = self.patch_num

                # Case 1: enough features, randomly select a continuous block
                if M >= N:
                    start_idx = np.random.randint(0, M - N + 1)
                    selected_features = features[start_idx:start_idx + N]
                
                # Case 2: not enough features, resample with replacement
                else:
                    repeat_times = math.ceil(N / M)
            
                    # Repeat features to form an extended sequence (maintaining continuity)
                    extended_features = np.tile(features, (repeat_times, 1))  # shape: (repeat_times*M, D)
                    
                    # Randomly select a starting position
                    max_start = extended_features.shape[0] - N
                    start_idx = np.random.randint(0, max_start + 1)
                    
                    # Select a continuous block
                    selected_features = extended_features[start_idx:start_idx + N]
                return selected_features, -1, wsi_label, idx
            
            else:
                return features, -1, wsi_label, idx
                