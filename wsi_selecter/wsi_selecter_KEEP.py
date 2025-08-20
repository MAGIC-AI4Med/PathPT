from torch.utils.data import Dataset, DataLoader
import json
import torch
import os
from utils import load_all_prompts
import params
import random
from data import SubWSIDataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm 
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from collections import Counter

def get_classnames_from_division_file(division_file, dataset):
    with open(division_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)[dataset]
    name2label = meta['name2label']
    classnames = sorted(name2label.keys(), key=lambda x: name2label[x])
    return classnames

def generate_patch_label_3d(logits, logits_thd, wsi_labels):
    """
    三维版本 (支持 batch 处理)
    输入:
        logits: [batch_size, num_patches, num_classes]
        logits_thd: 置信度阈值 (标量)
        wsi_labels: [batch_size] 每个样本的原始标签 (0-based)
    输出:
        patch_labels: [batch_size, num_patches]
    """
    # 参数校验
    assert logits.dim() == 3, "输入logits必须是3维张量"
    batch_size, num_patches, num_classes = logits.shape
    device = logits.device

    # 转换WSI标签 (0-based转1-based)
    wsi_labels = wsi_labels + 1  # [batch_size]

    # 获取每个patch的最大logit及对应类别
    max_values, max_indices = torch.max(logits, dim=2)  # [batch_size, num_patches]

    # 初始化所有patch标签为 -wsi_label (广播机制)
    patch_labels = -wsi_labels.view(batch_size, 1).expand(-1, num_patches)  # [batch_size, num_patches]

    # 获取正常类 (类别0) 的logits
    normal_logits = logits[:, :, 0]  # [batch_size, num_patches]

    # 获取肿瘤类 (对应wsi_label) 的logits
    tumor_logits = logits.gather(
        dim=2,
        index=wsi_labels.view(batch_size, 1, 1).expand(-1, num_patches, 1)
    ).squeeze(2)  # [batch_size, num_patches]

    # 构建正常patch的掩码
    normal_mask = (normal_logits > logits_thd) & (max_indices == 0)  # [batch_size, num_patches]
    
    # 构建肿瘤patch的掩码
    tumor_mask = (tumor_logits > logits_thd) & (max_indices == wsi_labels.unsqueeze(1))  # [batch_size, num_patches]

    # 更新标签
    patch_labels = torch.where(
        normal_mask,
        torch.zeros_like(patch_labels),
        torch.where(
            tumor_mask,
            wsi_labels.unsqueeze(1).expand(-1, num_patches),
            patch_labels
        )
    )

    return patch_labels

def generate_patch_label(logits, logits_thd, wsi_label):
    
    try:
        wsi_label = wsi_label.item()
    except ValueError:
        print("WSI label must have exactly one element.")
    
    wsi_label += 1  ## wsi label from 0, for patch label: 0 represents normal
    
    row_max_values, row_max_indices = torch.max(logits, dim=1)
    
    normal_col = logits[:,0]
    normal_col_mask = (normal_col> logits_thd) & (row_max_indices == 0)
    normal_col_indices = normal_col_mask.nonzero(as_tuple=True)[0]
    
    tumor_col = logits[:,wsi_label]
    tumor_col_mask = (tumor_col > logits_thd) & (row_max_indices == wsi_label)
    tumor_col_indices = tumor_col_mask.nonzero(as_tuple=True)[0]
    
    label = [-1*wsi_label]*logits.shape[0]
    
    for id in normal_col_indices.flatten():
        id = id.item()
        label[id] = 0
    
    for id in tumor_col_indices.flatten():
        id = id.item()
        label[id] = wsi_label
    
    return label

def predict_wsi_label(logits):
    patch_pred = torch.argmax(logits, dim=1)
    class_counts = torch.bincount(patch_pred, minlength=logits.shape[1])
    
    all_normal_flag = class_counts[1:].sum() == 0
    
    max_val = class_counts[1:].max()
    mask = (class_counts[1:] == max_val)
    tumor_equal_flag = mask.sum() > 1
    
    if all_normal_flag or tumor_equal_flag:
        patch_pred = torch.argmax(logits[:,1:], dim=1)
        class_counts = torch.bincount(patch_pred, minlength=logits.shape[1]-1)
        patch_prob = class_counts/class_counts.sum()
    else:
        patch_prob = class_counts[1:]/class_counts[1:].sum()
    
    final_pred = torch.argmax(patch_prob).item()
    return final_pred, patch_prob[final_pred]

def sample_from_sublists(list_of_lists, sample_size=10):
    sampled_data = []
    for sublist in list_of_lists:
        if len(sublist) >= sample_size:
            sampled = random.sample(sublist, sample_size)
        else:
            sampled = sublist  # 如果不足 10 个，取全部
        sampled_data.append(sampled)
    return sampled_data    


def patch_selector(label_model, label_type, dataset_name, vision_feats, wsi_label, logits_thd = 0., text_embedding = None, device='cuda:0'):

    # print('labeling patches...')    
    # prompts_lst = load_prompts_combination(param["zero_shot_template_file"], classnames)
    
    if label_type == 'zeroshot':
        if text_embedding is not None:
            text_features = text_embedding.cpu()
        else:
            ## manual prompt embedding
            tokenizer = AutoTokenizer.from_pretrained(params.KEEP_PATH, trust_remote_code=True)
            label_model.eval()
            
            classnames = get_classnames_from_division_file(params.DATASET_DIVISION, dataset_name)
            classnames = ['Normal'] + classnames
            prompts_lst = load_all_prompts(dataset_name, classnames)
            
            prompts_sublist = sample_from_sublists(prompts_lst, sample_size=20)
            
            with torch.no_grad():
                tokens = [tokenizer(prompts, max_length=256, padding='max_length', truncation=True, return_tensors='pt').to(device) for prompts in prompts_sublist]
                manual_text_features = [torch.mean(label_model.encode_text(token_input),dim = 0).cpu().numpy() for token_input in tokens] #(class, num_template*num_classname)
                manual_text_features = torch.from_numpy(np.array(manual_text_features)).cpu()
                text_features = manual_text_features / manual_text_features.norm(dim=-1, keepdim=True)
        
        logits = vision_feats.squeeze(0) @ text_features.t()
    else:
        label_model.eval()
        with torch.no_grad():
            _, logits = label_model(vision_feats.to(device))
    
    logits = F.softmax(logits, dim=-1).cpu()
    
    if len(logits.shape) == 2:
        patch_labels = generate_patch_label(logits, logits_thd, wsi_label)
    else:
        patch_labels = generate_patch_label_3d(logits, logits_thd, wsi_label)
    
    return patch_labels

def wsi_prompt_selector(train_wsi_loader, prompt_lst, PPL_model = None, device = 'cuda:0'):
    
    model = AutoModel.from_pretrained(params.KEEP_PATH, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(params.KEEP_PATH, trust_remote_code=True)
    model.eval()
    
    with torch.no_grad():
        tokens = [tokenizer(prompts, max_length=256, padding='max_length', truncation=True, return_tensors='pt').to(device) for prompts in prompt_lst]
        text_features = [model.encode_text(token_input) for token_input in tokens] #(class, num_template*num_classname)
    
    prompt_group_num = np.prod([len(cls_feats) for cls_feats in text_features]) ## too large
    
    cls_prompt_feats = [[] for cls_feats in text_features]
    
    prompt_num = 200
    select_num = 100
    
    prompt_embeddings = torch.zeros(prompt_num,len(text_features),text_features[0].shape[1], device=device)
    for idx in range(prompt_num): # random select 1000 prompt classifiers
        random.seed(idx)
        prompt_feats = [random.choice(cls_feats) for cls_feats in text_features]
        prompt_feats = torch.stack(prompt_feats)
        
        if PPL_model is not None:
            PPL_model.eval()
            with torch.no_grad():
                prompts = PPL_model.prompt_learner()
                tokenized_prompts = PPL_model.tokenized_prompts
                token_text_features = PPL_model.model.text(inputs_embeds=prompts, attention_mask=tokenized_prompts['attention_mask'].to(device)) 
                token_text_features = torch.nn.functional.normalize(token_text_features.pooler_output)
            
            prompt_feats += token_text_features
            prompt_feats = prompt_feats / prompt_feats.norm(dim=-1, keepdim=True)
        
        for cls_i in range(len(text_features)):
            cls_prompt_feats[cls_i].append(prompt_feats[cls_i])
        
        prompt_embeddings[idx,:,:] = prompt_feats
        
    gt_labels = []
    all_pred_labels = [[] for _ in range(prompt_num)]
    for data, _, wsi_label, _ in tqdm(train_wsi_loader):
        data = data.squeeze(0)
        # gt_labels.append(wsi_label.item()-1)
        gt_labels.append(wsi_label.item())  # for csv data
        
        all_logits = torch.einsum('nk,mck->mnc', data, prompt_embeddings.cpu())
        
        logits_list = [all_logits[i] for i in range(prompt_num)]
        
        for pt_idx, logits in enumerate(logits_list):
            patch_pred = torch.argmax(logits, dim=1)
            class_counts = torch.bincount(patch_pred, minlength=logits.shape[1])
            
            all_normal_flag = class_counts[1:].sum() == 0
            
            max_val = class_counts[1:].max()
            mask = (class_counts[1:] == max_val)
            tumor_equal_flag = mask.sum() > 1
            
            
            if all_normal_flag or tumor_equal_flag:
                patch_pred = torch.argmax(logits[:,1:], dim=1)
                class_counts = torch.bincount(patch_pred, minlength=logits.shape[1]-1)
                patch_prob = class_counts/class_counts.sum()
            else:
                patch_prob = class_counts[1:]/class_counts[1:].sum()
            
            final_pred = torch.argmax(patch_prob).item()
            all_pred_labels[pt_idx].append(final_pred)
    
    bacc_list = []
    for pred_labels in all_pred_labels:
        balanced_acc = balanced_accuracy_score(gt_labels, pred_labels)
        bacc_list.append(balanced_acc)
    
    sorted_indices = np.argsort(bacc_list)[::-1]

    # 获取前select_num个最大值的索引（如果列表长度不足select_num，则获取所有）
    top_k_indices = sorted_indices[:min(select_num, len(bacc_list))]

    # 获取对应的值
    top_k_values = [bacc_list[i] for i in top_k_indices]
    
    # selected_embeddings = []
    # for j in top_k_indices:
    #     cls_embeddings = [cls_prompt_feats[cls_i][j] for cls_i in range(len(cls_prompt_feats))]
    #     cls_embeddings = torch.stack(cls_embeddings)
    #     cls_embeddings = cls_embeddings / cls_embeddings.norm(dim=-1, keepdim=True)
    #     selected_embeddings.append(cls_embeddings)
    
    selected_embeddings = []
    for cls_i in range(len(cls_prompt_feats)):
        cls_embeddings = []
        for j in top_k_indices:
            cls_embeddings.append(cls_prompt_feats[cls_i][j])
        stacked_cls_embeddings = torch.stack(cls_embeddings).mean(dim=0)
        selected_embeddings.append(stacked_cls_embeddings)
    
    selected_prompt_embedding = torch.stack(selected_embeddings)
    selected_prompt_embedding = selected_prompt_embedding / selected_prompt_embedding.norm(dim=-1, keepdim=True)
    
    return selected_prompt_embedding


def patch_prompt_selector(train_dataset, dataset_name, param, device='cuda:0'):
    """
    选择每个类别的最佳提示
    
    参数:
    dataset (str): 数据集名称，如'BRAIN'
    param (dict): 参数字典，包含train_dir_path和zero_shot_template_file等
    device (str): 使用的设备，默认为'cuda:0'
    k (int): 每个类别选择的提示数量，默认为10
    
    返回:
    dict: 包含每个类别选中的提示的字典
    """
    # k = param['topn']
    
    # h5_path_lst = [os.path.join(param['train_dir_path'], path) for path in os.listdir(param['train_dir_path']) if path.endswith('.h5')]
    # selectset = AnnoPatchDataset(h5_path_lst=h5_path_lst, division_json=params.DATASET_DIVISION, shots=param['shots'][0], dataset=dataset_name)
    selectloader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    
    classnames = get_classnames_from_division_file(params.DATASET_DIVISION, dataset_name)
    classnames = ['Normal'] + classnames
    
    # prompts_lst = load_prompts_combination(param["zero_shot_template_file"], classnames)
    prompts_lst = load_all_prompts(dataset_name, classnames)
    
    model = AutoModel.from_pretrained(params.KEEP_PATH, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(params.KEEP_PATH, trust_remote_code=True)
    model.eval()
    
    with torch.no_grad():
        tokens = [tokenizer(prompts, max_length=256, padding='max_length', truncation=True, return_tensors='pt').to(device) for prompts in prompts_lst]
        text_features = [model.encode_text(token_input) for token_input in tokens] #(class, num_template*num_classname)
    
    print('Prompt selecting...')
    all_logits = [] # (class, num_img, num_template*num_classname)
    all_labels = [] # (class, num_img)
    for text_label, class_feature in enumerate(text_features):
        batch_logits = []
        batch_labels = []
        for img_features, img_label in tqdm(selectloader):
            img_features = img_features.to(device)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            class_feature = class_feature / class_feature.norm(dim=-1, keepdim=True)
            logits = img_features @ class_feature.t()
            batch_logits.append(logits)
            batch_labels.append(img_label)

        all_logits.append(torch.cat(batch_logits).cpu())
        all_labels.append(torch.cat(batch_labels).cpu())

    select_prompts = [[] for i in range(len(classnames))]
    select_features = []
    for text_label, logits in enumerate(all_logits):
        img_label = all_labels[text_label]
        mask = (img_label == text_label)
        
        positive_logits = logits[mask]
        negative_logits = -logits[~mask]
        
        positive_scores = positive_logits.sum(dim=0)
        negative_scores = negative_logits.sum(dim=0)
        
        positive_num = mask.sum()
        negative_num = (~mask).sum()
        
        scores = (positive_scores / positive_num) + (negative_scores / negative_num)
        # print(f"Class {text_label} scores: {scores}")
        
        final_k = min(k,len(scores))
        topk_ids = torch.topk(scores, final_k).indices
        select_features.append(text_features[text_label][topk_ids])
        for i in topk_ids:
            select_prompts[text_label].append(prompts_lst[text_label][i])
    
    # 返回选出的prompts作为字典
    meta = {i: select_prompts[i] for i in range(len(classnames))}
    return meta
