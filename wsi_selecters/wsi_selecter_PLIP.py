from torch.utils.data import Dataset, DataLoader
import json
import torch
from utils import load_all_prompts
import params
import random
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm 
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

def get_classnames_from_division_file(division_file, dataset):
    with open(division_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)[dataset]
    name2label = meta['name2label']
    classnames = sorted(name2label.keys(), key=lambda x: name2label[x])
    return classnames

def generate_patch_label_3d(logits, logits_thd, wsi_labels):
    """
    3D version (supports batch processing)
    Input:
        logits: [batch_size, num_patches, num_classes]
        logits_thd: confidence threshold (scalar)
        wsi_labels: [batch_size] original label for each sample (0-based)
    Output:
        patch_labels: [batch_size, num_patches]
    """
    # Parameter validation
    assert logits.dim() == 3, "Input logits must be a 3D tensor"
    batch_size, num_patches, num_classes = logits.shape
    device = logits.device

    # Convert WSI labels (0-based to 1-based)
    wsi_labels = wsi_labels + 1  # [batch_size]

    # Get maximum logit and corresponding class for each patch
    max_values, max_indices = torch.max(logits, dim=2)  # [batch_size, num_patches]

    # Initialize all patch labels as -wsi_label (broadcast mechanism)
    patch_labels = -wsi_labels.view(batch_size, 1).expand(-1, num_patches)  # [batch_size, num_patches]

    # Get normal class (class 0) logits
    normal_logits = logits[:, :, 0]  # [batch_size, num_patches]

    # Get tumor class (corresponding to wsi_label) logits
    tumor_logits = logits.gather(
        dim=2,
        index=wsi_labels.view(batch_size, 1, 1).expand(-1, num_patches, 1)
    ).squeeze(2)  # [batch_size, num_patches]

    # Build normal patch mask
    normal_mask = (normal_logits > logits_thd) & (max_indices == 0)  # [batch_size, num_patches]
    
    # Build tumor patch mask
    tumor_mask = (tumor_logits > logits_thd) & (max_indices == wsi_labels.unsqueeze(1))  # [batch_size, num_patches]

    # Update labels
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
            sampled = sublist  # If less than 10, take all
        sampled_data.append(sampled)
    return sampled_data    


def patch_selector(label_model, label_type, dataset_name, vision_feats, wsi_label, logits_thd = 0., text_embedding = None, device='cuda:0'):

    # print('labeling patches...')    
    # prompts_lst = load_prompts_combination(param["zero_shot_template_file"], classnames)
    
    if label_type == 'zeroshot':
        if text_embedding is not None:
            text_features = text_embedding.cpu()
        # Not used
        # else:
        #     ## manual prompt embedding
        #     tokenizer = AutoTokenizer.from_pretrained(params.KEEP_PATH, trust_remote_code=True)
        #     label_model.eval()
            
        #     classnames = get_classnames_from_division_file(params.DATASET_DIVISION, dataset_name)
        #     classnames = ['Normal'] + classnames
        #     prompts_lst = load_all_prompts(dataset_name, classnames)
            
        #     prompts_sublist = sample_from_sublists(prompts_lst, sample_size=20)
            
        #     with torch.no_grad():
                
        #         tokenizer = conch_tokenizer.get_tokenizer()
        #         prompt_input = conch_tokenizer.tokenize(tokenizer, prompts_sublist).to(device)

        #         text_feature = label_model.encode_text(prompt_input, embed_cls=False)
        #         text_features =  text_feature / text_feature.norm(dim=-1, keepdim=True)
        
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

## conch subtyping
def topj_pooling_conch(logits, topj):
    """
    logits: N x 1 logit for each patch
    coords: N x 2 coordinates for each patch
    topj: tuple of the top number of patches to use for pooling
    ss: spatial smoothing by k-nn
    ss_k: k in k-nn for spatial smoothing
    """
    # Sums logits across topj patches for each class, to get class prediction for each topj
    maxj = min(max(topj), logits.size(0)) # Ensures j is smaller than number of patches. Unlikely for number of patches to be < 10, but just in case
    values, _ = logits.topk(maxj, 0, True, True) # maxj x C
    preds = {j : values[:min(j, maxj)].sum(dim=0, keepdim=True) for j in topj} # dict of 1 x C logit scores
    preds = {key: val.argmax(dim=1) for key,val in preds.items()} # dict of predicted class indices
    return preds

def wsi_prompt_selector(base_model, tokenizer, train_wsi_loader, prompt_lst, device = 'cuda:0'):
    
    base_model.eval()    
    with torch.no_grad():
        tokens = [tokenizer(list(prompts),add_special_tokens=True,max_length=77,pad_to_max_length=True,return_tensors='pt').to(device) for prompts in prompt_lst]
        text_features = [base_model.get_text_features(**token_input) for token_input in tokens]
    
    cls_prompt_feats = [[] for _ in text_features]
    
    prompt_num = 200
    select_num = 100
    
    prompt_embeddings = torch.zeros(prompt_num,len(text_features),text_features[0].shape[1], device=device)
    for idx in range(prompt_num): # randomly select 1000 prompt classifiers
        random.seed(idx)
        prompt_feats = [random.choice(cls_feats) for cls_feats in text_features]
        prompt_feats = torch.stack(prompt_feats)
        
        for cls_i in range(len(text_features)):
            cls_prompt_feats[cls_i].append(prompt_feats[cls_i])
        
        prompt_embeddings[idx,:,:] = prompt_feats
        
    gt_labels = []
    all_pred_labels = [[] for _ in range(prompt_num)]
    for data, _, wsi_label, _ in tqdm(train_wsi_loader):
        data = data.squeeze(0)
        # gt_labels.append(wsi_label.item()-1)
        gt_labels.append(wsi_label.item())
        
        all_logits = torch.einsum('nk,mck->mnc', data, prompt_embeddings.cpu())
        
        logits_list = [all_logits[i] for i in range(prompt_num)]
        
        for pt_idx, logits in enumerate(logits_list):
            
            preds = topj_pooling_conch(logits[:,1:], topj = [1,5,10,50,100])
            all_pred_labels[pt_idx].append(preds[100].item())
    
    bacc_list = []
    for pred_labels in all_pred_labels:
        balanced_acc = balanced_accuracy_score(gt_labels, pred_labels)
        bacc_list.append(balanced_acc)
    
    sorted_indices = np.argsort(bacc_list)[::-1]

    # Get the indices of the top select_num maximum values (if list length is less than select_num, get all)
    top_k_indices = sorted_indices[:min(select_num, len(bacc_list))]

    # Get corresponding values
    top_k_values = [bacc_list[i] for i in top_k_indices]
    
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

# Not used
# def patch_prompt_selector(train_dataset, dataset_name, param, device='cuda:0'):
#     """
#     Select the best prompt for each class
    
#     Parameters:
#     dataset (str): Dataset name, such as 'BRAIN'
#     param (dict): Parameter dictionary, including train_dir_path and zero_shot_template_file, etc.
#     device (str): Device to use, default is 'cuda:0'
#     k (int): Number of prompts to select for each class, default is 10
    
#     Returns:
#     dict: Dictionary containing selected prompts for each class
#     """
#     # k = param['topn']
    
#     # h5_path_lst = [os.path.join(param['train_dir_path'], path) for path in os.listdir(param['train_dir_path']) if path.endswith('.h5')]
#     # selectset = AnnoPatchDataset(h5_path_lst=h5_path_lst, division_json=params.DATASET_DIVISION, shots=param['shots'][0], dataset=dataset_name)
#     selectloader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    
#     classnames = get_classnames_from_division_file(params.DATASET_DIVISION, dataset_name)
#     classnames = ['Normal'] + classnames
    
#     # prompts_lst = load_prompts_combination(param["zero_shot_template_file"], classnames)
#     prompts_lst = load_all_prompts(dataset_name, classnames)
    
#     model = AutoModel.from_pretrained(params.KEEP_PATH, trust_remote_code=True).to(device)
#     tokenizer = AutoTokenizer.from_pretrained(params.KEEP_PATH, trust_remote_code=True)
#     model.eval()
    
#     with torch.no_grad():
#         tokens = [tokenizer(prompts, max_length=256, padding='max_length', truncation=True, return_tensors='pt').to(device) for prompts in prompts_lst]
#         text_features = [model.encode_text(token_input) for token_input in tokens] #(class, num_template*num_classname)
    
#     print('Prompt selecting...')
#     all_logits = [] # (class, num_img, num_template*num_classname)
#     all_labels = [] # (class, num_img)
#     for text_label, class_feature in enumerate(text_features):
#         batch_logits = []
#         batch_labels = []
#         for img_features, img_label in tqdm(selectloader):
#             img_features = img_features.to(device)
#             img_features = img_features / img_features.norm(dim=-1, keepdim=True)
#             class_feature = class_feature / class_feature.norm(dim=-1, keepdim=True)
#             logits = img_features @ class_feature.t()
#             batch_logits.append(logits)
#             batch_labels.append(img_label)

#         all_logits.append(torch.cat(batch_logits).cpu())
#         all_labels.append(torch.cat(batch_labels).cpu())

#     select_prompts = [[] for i in range(len(classnames))]
#     select_features = []
#     for text_label, logits in enumerate(all_logits):
#         img_label = all_labels[text_label]
#         mask = (img_label == text_label)
        
#         positive_logits = logits[mask]
#         negative_logits = -logits[~mask]
        
#         positive_scores = positive_logits.sum(dim=0)
#         negative_scores = negative_logits.sum(dim=0)
        
#         positive_num = mask.sum()
#         negative_num = (~mask).sum()
        
#         scores = (positive_scores / positive_num) + (negative_scores / negative_num)
#         # print(f"Class {text_label} scores: {scores}")
        
#         final_k = min(k,len(scores))
#         topk_ids = torch.topk(scores, final_k).indices
#         select_features.append(text_features[text_label][topk_ids])
#         for i in topk_ids:
#             select_prompts[text_label].append(prompts_lst[text_label][i])
    
#     # Return selected prompts as dictionary
#     meta = {i: select_prompts[i] for i in range(len(classnames))}
#     return meta
