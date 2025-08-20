import random
import utils
import os
import evaluation
import json
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from feat_transform import FeatureTransform

class SelectDataset(Dataset):
    def __init__(self, data, labels, use_aug):
        """
        创建一个自定义数据集
        
        参数:
            data: 数据，可以是numpy数组或者张量
            labels: 标签，可以是numpy数组或者张量
        """
        # 确保data和labels都是张量格式
        self.data =  data
        self.labels = labels
        self.use_aug = use_aug
        
        if self.use_aug:
            self.feature_transform = FeatureTransform(
                    noise_level=0.05,
                    mixup_prob=0.3,
                    rotation_prob=0.3,
                    p=0.5,
                    smoothing = 0.1,
                    num_classes = len(set(self.labels)),
                )
        
    def __len__(self):
        """返回数据集的大小"""
        return len(self.data)

    def get_labels(self):
        return self.labels
    
    def __getitem__(self, idx):
        """根据索引获取一个样本"""
        feature = self.data[idx]
        label = self.labels[idx]
        
         ###### augmentation
        if self.use_aug:
            rand_idx = random.randint(0,len(self.data)-1)
            feat2 = self.data[rand_idx]
            label2 = self.labels[rand_idx]
            aug_feature, aug_label = self.feature_transform(feature, feat2, label, label2)
            
            return aug_feature, aug_label
        ##############
        else:
            return feature, label


class gen_dataloader(): #val_size not implemented
    def __init__(self, feature_root, division_json, shot=10, dataset_name='BRCA', batch_size=256):
        self.feat_paths = glob.glob(feature_root + '/*.h5')
        self.division_json = division_json
        self.shot = shot
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        
        # self.testset = SubWSIDataset(h5_path_lst=all_h5_paths, division_json=division_json, dataset=self.dataset, division='train_IDs')
        # print('loading test dataset...')
        
        self.wsi_feats, self.wsi_labels = load_WSI_feats(self.feat_paths, division_json, dataset_name)
        
        # self.testset = SubWSIDataset(h5_path_lst=self.feat_paths, 
        #                              division_json=division_json, 
        #                              dataset_name=self.dataset_name, 
        #                              division='test_IDs')
        self.testset = WSIDataset(self.feat_paths, 
                                  self.wsi_feats, 
                                  self.wsi_labels, 
                                  division_json=self.division_json, 
                                  dataset_name=self.dataset_name,  
                                  division='test_IDs')
        # self.testset = None
        
    def get_classnames(self):
        return self.testset.get_classnames()

    def gen_testloader(self, metric):
        if metric is evaluation.sub_bacc_wf1:
            test_loader = DataLoader(self.testset, batch_size=1, shuffle=False) # wsidataset batch size is 1
        elif metric is evaluation.simple_dice_auc:
            test_loader = DataLoader(self.trainset_patch, batch_size=1, shuffle=False) # wsidataset batch size is 1
        elif isinstance(metric, dict):
            test_loader = dict()
            test_loader['patch'] = DataLoader(self.trainset_patch, batch_size=1, shuffle=False)
            test_loader['wsi'] = DataLoader(self.testset, batch_size=1, shuffle=False)# wsidataset batch size is 1
        return test_loader
    
    def split_train_val(self, seed = 0):
        # print('spliting train and validation dataset...')
        with open(self.division_json, 'r', encoding='utf-8') as f:
            meta = json.load(f)[self.dataset_name.upper()]
        
        h5_path_lst = self.feat_paths

        trainval_labeled_feats_dict = {}
        for h5_path in h5_path_lst:
            slide_ID = h5_path.split('/')[-1].split('.h5')[0]
            for label, ID_list in meta['train_IDs'].items():
                if label not in trainval_labeled_feats_dict:
                    trainval_labeled_feats_dict[label] = []
                if slide_ID in ID_list:
                    trainval_labeled_feats_dict[label].append(h5_path)
        
        train_feats_dict = dict()
        val_feats_dict = dict()
        for label, feat_list in trainval_labeled_feats_dict.items():
            random.seed(seed)
            random.shuffle(feat_list)
            if label not in train_feats_dict:
                train_feats_dict[label] = []
            if label not in val_feats_dict:
                val_feats_dict[label] = []
            
            train_feats_dict[label].extend(feat_list[:self.shot])
            val_feats_dict[label].extend(feat_list[self.shot:])
        
        train_list,val_list = [],[]
        for k,v in train_feats_dict.items():
            train_list.extend(v)
        for k,v in val_feats_dict.items():
            val_list.extend(v)
        
        return train_list, val_list
        
    def gen_wsiloader(self, feat_paths):
        # print('loading wsi dataset...')
        # wsi_data = SubWSIDataset(h5_path_lst=feat_paths, 
        #                        division_json=self.division_json, 
        #                        dataset_name=self.dataset_name, 
        #                        division='train_IDs')
        # wsi_loader = DataLoader(wsi_data, batch_size=1, shuffle=False)
        wsi_data = WSIDataset(feat_paths, 
                            self.wsi_feats, 
                            self.wsi_labels, 
                            division_json=self.division_json, 
                            dataset_name=self.dataset_name,  
                            division='train_IDs')
        wsi_loader = DataLoader(wsi_data, batch_size=1, shuffle=False)
        
        return wsi_loader
    




class SubWSIDataset(Dataset): # Only for test data #label是从1开始的
    def __init__(self, h5_path_lst=None, division_json=None, dataset_name=None, division='test_IDs'):
        self.data = []
        self.label = []
        if division == 'test_IDs':
            print("loading subtyping test dataset...")
        elif division == 'train_IDs':
            print("loading subtyping train/validation dataset...")
            
        with open(division_json, 'r') as f:
            meta = json.load(f)[dataset_name.upper()]
        for h5_path in tqdm(h5_path_lst):
            for label, labeled_train_list in meta[division].items():
                if os.path.splitext(os.path.basename(h5_path))[0] in labeled_train_list:
                    # print(f'time:{datetime.now().strftime("%m%d%H%M%S")}')
                    # print(f'loading h5: {os.path.basename(h5_path)} of label {label}')
                    # if division == 'train_IDs':
                    #     coords, data = utils.read_h5(h5_path)
                    # else:
                    coords, data = utils.read_h5(h5_path)
                    self.data.append(data)
                    
                    self.label.append(int(label))
                    # print(f'time:{datetime.now().strftime("%m%d%H%M%S")}')
        name2label = meta['name2label']
        self.classnames = sorted(name2label.keys(), key=lambda x: name2label[x])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label
    
    def get_classnames(self): #注意这里的classnames里没有normal
        return self.classnames


def load_WSI_feats(h5_path_lst, division_json, dataset_name):
    print('loading wsi features...')
    with open(division_json, 'r') as f:
        meta = json.load(f)[dataset_name.upper()]
    
    wsi_feats = dict()
    wsi_labels = dict()
    for h5_path in tqdm(h5_path_lst):
        for k,v in meta.items():
            if k not in ['train_IDs', 'test_IDs']:
                continue
            for label, labeled_train_list in v.items():
                if os.path.splitext(os.path.basename(h5_path))[0] not in labeled_train_list:
                    continue
                coords, data = utils.read_h5(h5_path)
                wsi_feats[h5_path] = data
                wsi_labels[h5_path] = int(label)
    
    return wsi_feats, wsi_labels





class WSIDataset(Dataset): # Only for test data #label是从1开始的
    def __init__(self, h5_path_lst, wsi_feats, wsi_labels, division_json=None, dataset_name=None,  division='test_IDs'):
        self.data = []
        self.label = []
        # if division == 'test_IDs':
        #     print("loading subtyping test dataset...")
        # elif division == 'train_IDs':
        #     print("loading subtyping train/validation dataset...")
        
        with open(division_json, 'r') as f:
            meta = json.load(f)[dataset_name.upper()]
            
        for h5_path in h5_path_lst:
            if os.path.splitext(os.path.basename(h5_path))[0] not in sum(meta[division].values(), []):
                continue
            self.data.append(wsi_feats[h5_path])
            self.label.append(wsi_labels[h5_path])

        name2label = meta['name2label']
        self.classnames = sorted(name2label.keys(), key=lambda x: name2label[x])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label
    
    def get_classnames(self): #注意这里的classnames里没有normal
        return self.classnames