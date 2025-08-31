from torch.utils.data import Dataset, DataLoader
import json
import torch
from utils import load_all_prompts
import params
import random
from tqdm import tqdm 
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

# TODO: Import model-specific libraries here
# Examples:
# from transformers import AutoModel, AutoTokenizer  # For CLIP-like models
# from transformers import XLMRobertaTokenizer       # For MUSK
# import open_clip_CONCH as conch_clip               # For CONCH
# import your_model_library as your_model

def get_classnames_from_division_file(division_file, dataset):
    """Get class names from dataset division file"""
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
    """Generate patch-level labels based on WSI-level label and logits"""
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
    """Predict WSI-level label from patch logits"""
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
    """Sample a fixed number of items from each sublist"""
    sampled_data = []
    for sublist in list_of_lists:
        if len(sublist) >= sample_size:
            sampled = random.sample(sublist, sample_size)
        else:
            sampled = sublist  # If less than sample_size, take all
        sampled_data.append(sampled)
    return sampled_data    

def patch_selector(label_model, label_type, dataset_name, vision_feats, wsi_label, logits_thd=0., text_embedding=None, device='cuda:0'):
    """
    Select patches based on confidence threshold and WSI label
    
    Args:
        label_model: The foundation model for text encoding
        label_type: 'zeroshot' or other training types
        dataset_name: Name of the dataset
        vision_feats: Visual features of patches
        wsi_label: WSI-level ground truth label
        logits_thd: Confidence threshold for patch selection
        text_embedding: Pre-computed text embeddings (optional)
        device: Device to use
    
    Returns:
        patch_labels: List of patch-level labels
    """
    if label_type == 'zeroshot':
        if text_embedding is not None:
            text_features = text_embedding.cpu()
        else:
            # TODO: Implement manual prompt embedding for your model
            # Load your tokenizer and model here
            # Example implementations:
            
            # For CLIP-like models (KEEP, PLIP):
            # tokenizer = AutoTokenizer.from_pretrained(params.YOUR_MODEL_PATH, trust_remote_code=True)
            
            # For MUSK:
            # tokenizer = XLMRobertaTokenizer("./your_model/tokenizer.spm")
            
            # For CONCH:
            # tokenizer = conch_tokenizer.get_tokenizer()
            
            label_model.eval()
            
            classnames = get_classnames_from_division_file(params.DATASET_DIVISION, dataset_name)
            classnames = ['Normal'] + classnames
            prompts_lst = load_all_prompts(dataset_name, classnames)
            
            prompts_sublist = sample_from_sublists(prompts_lst, sample_size=20)
            
            with torch.no_grad():
                # TODO: Implement tokenization for your model
                # Examples:
                
                # For CLIP-like models:
                # tokens = [tokenizer(prompts, max_length=256, padding='max_length', 
                #                   truncation=True, return_tensors='pt').to(device) 
                #          for prompts in prompts_sublist]
                # text_features = [label_model.encode_text(token_input) for token_input in tokens]
                
                # For MUSK:
                # tokens = []
                # paddings = []
                # for prompts in prompts_sublist:
                #     tokenized_prompts = []
                #     tokenized_pads = []
                #     for prompt in prompts:
                #         tps, tpsd = musk_utils.xlm_tokenizer(prompt, tokenizer, max_len=100)
                #         tokenized_prompts.append(torch.tensor(tps).unsqueeze(0))
                #         tokenized_pads.append(torch.tensor(tpsd).unsqueeze(0))
                #     tokens.append(torch.cat(tokenized_prompts).to(device))
                #     paddings.append(torch.cat(tokenized_pads).to(device))
                # text_features = [label_model(text_description=token_input, padding_mask=pad_input)[1] 
                #                 for token_input, pad_input in zip(tokens, paddings)]
                
                # For CONCH:
                # prompt_input = conch_tokenizer.tokenize(tokenizer, prompts_sublist).to(device)
                # text_features = label_model.encode_text(prompt_input, embed_cls=False)
                
                # TODO: Replace with your model's text encoding method
                manual_text_features = None  # Implement your text encoding here
                
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

def topj_pooling(logits, topj):
    """
    Top-j pooling for WSI-level prediction (used by CONCH)
    
    Args:
        logits: N x C logits for each patch
        topj: tuple of the top number of patches to use for pooling
    
    Returns:
        preds: dict of predicted class indices for different j values
    """
    maxj = min(max(topj), logits.size(0))
    values, _ = logits.topk(maxj, 0, True, True)
    preds = {j : values[:min(j, maxj)].sum(dim=0, keepdim=True) for j in topj}
    preds = {key: val.argmax(dim=1) for key, val in preds.items()}
    return preds

def wsi_prompt_selector(base_model, tokenizer, train_wsi_loader, prompt_lst, PPL_model=None, device='cuda:0'):
    """
    Select optimal prompts for WSI-level classification
    
    Args:
        base_model: The foundation model
        tokenizer: Tokenizer for the model
        train_wsi_loader: DataLoader for training WSIs
        prompt_lst: List of prompts for each class
        PPL_model: Prompt-learning model (optional)
        device: Device to use
    
    Returns:
        selected_prompt_embedding: Selected prompt embeddings
    """
    base_model.eval()
    
    with torch.no_grad():
        # TODO: Implement tokenization and text encoding for your model
        # Examples:
        
        # For CLIP-like models (PLIP):
        # tokens = [tokenizer(list(prompts), add_special_tokens=True, max_length=77,
        #                   pad_to_max_length=True, return_tensors='pt').to(device) 
        #          for prompts in prompt_lst]
        # text_features = [base_model.get_text_features(**token_input) for token_input in tokens]
        
        # For KEEP:
        # tokens = [tokenizer(prompts, max_length=256, padding='max_length', 
        #                   truncation=True, return_tensors='pt').to(device) 
        #          for prompts in prompt_lst]
        # text_features = [base_model.encode_text(token_input) for token_input in tokens]
        
        # For MUSK:
        # tokens = []
        # paddings = []
        # for prompts in prompt_lst:
        #     tokenized_prompts = []
        #     tokenized_pads = []
        #     for prompt in prompts:
        #         tps, tpsd = musk_utils.xlm_tokenizer(prompt, tokenizer, max_len=100)
        #         tokenized_prompts.append(torch.tensor(tps).unsqueeze(0))
        #         tokenized_pads.append(torch.tensor(tpsd).unsqueeze(0))
        #     tokens.append(torch.cat(tokenized_prompts).to(device))
        #     paddings.append(torch.cat(tokenized_pads).to(device))
        # text_features = [base_model(text_description=token_input, padding_mask=pad_input)[1] 
        #                 for token_input, pad_input in zip(tokens, paddings)]
        
        # For CONCH:
        # tokens = [conch_clip.tokenize(conch_clip.get_tokenizer(), prompts).to(device) 
        #          for prompts in prompt_lst]
        # text_features = [base_model.encode_text(token_input) for token_input in tokens]
        
        # TODO: Replace with your model's text encoding method
        text_features = []  # Implement your text encoding here
    
    cls_prompt_feats = [[] for _ in text_features]
    
    prompt_num = 200
    select_num = 100
    
    prompt_embeddings = torch.zeros(prompt_num, len(text_features), text_features[0].shape[1], device=device)
    
    for idx in range(prompt_num):
        random.seed(idx)
        prompt_feats = [random.choice(cls_feats) for cls_feats in text_features]
        prompt_feats = torch.stack(prompt_feats)
        
        # Optional: Add learned prompt features if PPL_model is provided
        if PPL_model is not None:
            PPL_model.eval()
            with torch.no_grad():
                # TODO: Implement prompt learning integration for your model
                # This part varies significantly between models
                # See KEEP, MUSK implementations for examples
                
                prompts = PPL_model.prompt_learner()
                # Add model-specific prompt processing here
                token_text_features = None  # Implement based on your model
                
                if token_text_features is not None:
                    prompt_feats += token_text_features
                    prompt_feats = prompt_feats / prompt_feats.norm(dim=-1, keepdim=True)
        
        for cls_i in range(len(text_features)):
            cls_prompt_feats[cls_i].append(prompt_feats[cls_i])
        
        prompt_embeddings[idx, :, :] = prompt_feats
    
    # Evaluate prompts on training data
    gt_labels = []
    all_pred_labels = [[] for _ in range(prompt_num)]
    
    for data, _, wsi_label, _ in tqdm(train_wsi_loader):
        data = data.squeeze(0)
        gt_labels.append(wsi_label.item())
        
        all_logits = torch.einsum('nk,mck->mnc', data.float(), prompt_embeddings.cpu().float())
        logits_list = [all_logits[i] for i in range(prompt_num)]
        
        for pt_idx, logits in enumerate(logits_list):
            # TODO: Implement WSI-level prediction logic
            # This can vary based on your model's requirements
            
            # Option 1: Use top-j pooling (like CONCH)
            # preds = topj_pooling(logits[:,1:], topj=[1,5,10,50,100])
            # all_pred_labels[pt_idx].append(preds[100].item())
            
            # Option 2: Use majority voting (like KEEP, MUSK)
            final_pred, _ = predict_wsi_label(logits)
            all_pred_labels[pt_idx].append(final_pred)
    
    # Select best prompts based on balanced accuracy
    bacc_list = []
    for pred_labels in all_pred_labels:
        balanced_acc = balanced_accuracy_score(gt_labels, pred_labels)
        bacc_list.append(balanced_acc)
    
    sorted_indices = np.argsort(bacc_list)[::-1]
    top_k_indices = sorted_indices[:min(select_num, len(bacc_list))]
    
    # Average selected prompt embeddings for each class
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

def patch_prompt_selector(train_dataset, dataset_name, param, device='cuda:0', k=10):
    """
    Select the best prompts for each class (optional function, not commonly used)
    
    Args:
        train_dataset: Training dataset
        dataset_name: Name of the dataset
        param: Parameter dictionary
        device: Device to use
        k: Number of prompts to select for each class
    
    Returns:
        meta: Dictionary containing selected prompts for each class
    """
    # TODO: Implement if needed for your specific use case
    # This function is typically not used in the main pipeline
    # but can be useful for prompt analysis
    
    selectloader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    
    classnames = get_classnames_from_division_file(params.DATASET_DIVISION, dataset_name)
    classnames = ['Normal'] + classnames
    
    prompts_lst = load_all_prompts(dataset_name, classnames)
    
    # TODO: Load your model and tokenizer
    # model = YourModel.from_pretrained(params.YOUR_MODEL_PATH).to(device)
    # tokenizer = YourTokenizer.from_pretrained(params.YOUR_MODEL_PATH)
    # model.eval()
    
    # TODO: Implement text feature extraction
    # with torch.no_grad():
    #     tokens = [tokenizer(...) for prompts in prompts_lst]
    #     text_features = [model.encode_text(token_input) for token_input in tokens]
    
    # TODO: Implement prompt selection logic
    # Similar to existing implementations but adapted for your model
    
    # Return selected prompts as dictionary
    meta = {}  # Implement your prompt selection here
    return meta