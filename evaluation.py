import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
import numpy as np
import warnings
from tqdm import tqdm
import os
import cv2
import openslide
warnings.filterwarnings("ignore", category=UserWarning)

def compute_mean_dice(label, prediction, cls):
    # 计算每列的 TP, FP, FN
    tp = ((label == cls) & (prediction == cls)).sum()
    fp = ((prediction == cls) & (label != cls)).sum()
    fn = ((label == cls) & (prediction != cls)).sum()
    
    # 计算每列 Dice系数（避免除零）
    denominator = 2 * tp + fp + fn
    dice = (2 * tp) / denominator if denominator != 0 else 0

    return dice

def compute_dice(model, data_loader, device, zeroshot = False):
    model.eval()
        
    all_dice = []
    print('computing dice...')

    # for data, wsi_label in tqdm(test_loader):
    for data, patch_label, wsi_label, _ in tqdm(data_loader):
        data = data.float().to(device).squeeze(0)  # Assuming each batch contains one WSI (one wsi, batch = 1)
        patch_label = patch_label.to(device).squeeze(0)
        with torch.no_grad():
            if zeroshot:
                patch_logits = model(data)
            else:
                wsi_logits, patch_logits = model(data)
            patch_pred = torch.argmax(patch_logits, dim=1)
            
            dice = compute_mean_dice(patch_label, patch_pred, wsi_label.item() + 1)
            
            if isinstance(dice, int):
                all_dice.append(dice)
            else:
                all_dice.append(dice.item())
        
        torch.cuda.empty_cache()

    return np.array(all_dice).mean()


def sub_bacc_wf1(model, test_loader, device, normal_ext=False, batch_classifier=None, model_type = 'fewshot', prompt_embedding = None, save_logits = False, vision_only = False):
    model.eval()
    if batch_classifier:
        batch_classifier.eval()
    gt_labels = []
    pred_labels_patch = []
    pred_probs_patch = []
    
    pred_labels_mil = []
    pred_probs_mil = []
    
    all_logits = []
    print('Evaluating...')

    for data, _, wsi_label, _ in tqdm(test_loader):
        data = data.float().squeeze(0)  # (n_patch, 1024)
        wsi_label = wsi_label.item()
        gt_labels.append(wsi_label)
        
        # 分批处理patch数据
        batch_size = 16384
        n_patches = data.shape[0]
        all_patch_logits = []
        
        with torch.no_grad():
            for i in range(0, n_patches, batch_size):
                end_idx = min(i + batch_size, n_patches)
                batch_data = data[i:end_idx].to(device)  # (batch_size, 1024)
                
                if model_type == 'zeroshot':
                    batch_logits = model(batch_data)  # (batch_size, n_cls)
                elif model_type == 'fewshot':
                    batch_logits, _ = model(batch_data)  # (batch_size, n_cls)
                elif model_type == 'prompting':
                    batch_data = batch_data / batch_data.norm(dim=-1, keepdim=True)
                    prompt_embedding = prompt_embedding / prompt_embedding.norm(dim=-1, keepdim=True)
                    batch_logits = batch_data.float() @ prompt_embedding.float().t()
                elif model_type == 'fewshot-mil':
                    wsi_logits, batch_logits = model(batch_data)
                
                all_patch_logits.append(batch_logits)
            
            # 拼接所有batch的结果
            patch_logits = torch.cat(all_patch_logits, dim=0)  # (n_patch, n_cls)
            
            # 对于fewshot-mil的特殊处理
            if model_type == 'fewshot-mil':
                if vision_only:
                    wsi_pred = torch.argmax(wsi_logits, dim=1)
                    wsi_prob = torch.nn.functional.softmax(wsi_logits, dim=1)
                    pred_labels_mil.append(wsi_pred.item())
                    pred_probs_mil.append(wsi_prob.squeeze())
                    
        torch.cuda.empty_cache()
        if patch_logits is not None:  
            all_logits.append(patch_logits.cpu())
            
            if normal_ext:
                patch_logits = patch_logits[:, 1:] #没有normal的wsi
            # wsi_label -= 1 #testset 的label从1开始
            
            # if batch_classifier == None:
            ### patch prediction
            patch_pred = torch.argmax(patch_logits, dim=1)
            class_counts = torch.bincount(patch_pred, minlength=patch_logits.shape[1])
            
            all_normal_flag = class_counts[1:].sum() == 0
            
            max_val = class_counts[1:].max()
            mask = (class_counts[1:] == max_val)
            tumor_equal_flag = mask.sum() > 1
            
            
            if all_normal_flag or tumor_equal_flag:
                patch_pred = torch.argmax(patch_logits[:,1:], dim=1)
                class_counts = torch.bincount(patch_pred, minlength=patch_logits.shape[1]-1)
                patch_prob = class_counts/class_counts.sum()
            else:
                patch_prob = class_counts[1:]/class_counts[1:].sum()
            
            pred_probs_patch.append(patch_prob.squeeze())
            final_pred = torch.argmax(patch_prob).item()
            pred_labels_patch.append(final_pred)
            
            if final_pred != wsi_label:
                # print(final_prob)
                test = 1
            
    # 计算平衡准确率和加权F1分数
    # print(gt_labels, pred_labels)
    
    results = {}
    if len(pred_labels_patch) > 0:
        print('***** patch result:')
        print(confusion_matrix(gt_labels,pred_labels_patch))
        patch_bacc = balanced_accuracy_score(gt_labels, pred_labels_patch)
        report = classification_report(gt_labels, pred_labels_patch, output_dict=True, zero_division=0)
        patch_wf1 = report['weighted avg']['f1-score']
        
        results['patch_bacc'] = patch_bacc
        results['patch_wf1'] = patch_wf1
        results['logits'] = all_logits
    
    if vision_only:
        print('***** wsi cls token result:')
        print(confusion_matrix(gt_labels,pred_labels_mil))
        wsi_bacc = balanced_accuracy_score(gt_labels, pred_labels_mil)
        report = classification_report(gt_labels, pred_labels_mil, output_dict=True, zero_division=0)
        wsi_wf1 = report['weighted avg']['f1-score']
        
        # ensemble_probs = (torch.stack(pred_probs_patch) + torch.stack(pred_probs_mil))/2
        # ensemble_labels = torch.argmax(ensemble_probs, dim=1).tolist()
        
        # print('***** ensemble result:')
        # print(confusion_matrix(gt_labels,ensemble_labels))
        # ensemble_bacc = balanced_accuracy_score(gt_labels, ensemble_labels)
        # report = classification_report(gt_labels, ensemble_labels, output_dict=True, zero_division=0)
        # ensemble_wf1 = report['weighted avg']['f1-score']

        results['wsi_bacc'] = wsi_bacc
        results['wsi_wf1'] = wsi_wf1
        
        # results['ensemble_bacc'] = ensemble_bacc
        # results['ensemble_wf1'] = ensemble_wf1

    return results


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


def sub_bacc_wf1_conch(model, test_loader, device, normal_ext=False, batch_classifier=None, model_type = 'fewshot', prompt_embedding = None, save_logits = False, vision_mil = False):
    model.eval()
    if batch_classifier:
        batch_classifier.eval()
    gt_labels = []
    pred_labels_patch = []
    pred_probs_patch = []
    
    pred_labels_mil = []
    pred_probs_mil = []
    
    all_logits = []
    print('Evaluating...')
    for data, wsi_label in tqdm(test_loader):
        data = data.to(device).squeeze(0)  # Assuming each batch contains one WSI (one wsi, batch = 1)
        wsi_label = wsi_label.item()
        with torch.no_grad():
            if model_type=='zeroshot':
                patch_logits = model(data) # (n_patch, n_cls) or (n_patch, n_cls+1) with normal
            elif model_type=='fewshot':
                patch_logits, _ = model(data) # (n_patch, n_cls) or (n_patch, n_cls+1) with normal

            elif model_type =='prompting':
                data = data / data.norm(dim=-1, keepdim=True)
                prompt_embedding = prompt_embedding / prompt_embedding.norm(dim=-1, keepdim=True)
                patch_logits = data @ prompt_embedding.t()
            elif model_type =='fewshot-mil':
                wsi_logits, patch_logits = model(data)
                
                if vision_mil:
                    wsi_pred = torch.argmax(wsi_logits, dim=1)
                    wsi_prob = torch.nn.functional.softmax(wsi_logits, dim = 1)
                    pred_labels_mil.append(wsi_pred.item())
                    pred_probs_mil.append(wsi_prob.squeeze())
                
        torch.cuda.empty_cache()
        all_logits.append(patch_logits.cpu())
        
        if normal_ext:
            patch_logits = patch_logits[:, 1:] #没有normal的wsi
        # wsi_label -= 1 #testset 的label从1开始
        gt_labels.append(wsi_label)
        
        preds = topj_pooling_conch(patch_logits[:,1:], topj = [1,5,10,50,100])
        pred_labels_patch.append(preds[100].item())
         
    # 计算平衡准确率和加权F1分数
    # print(gt_labels, pred_labels)
    
    results = {}
    print('***** patch result:')
    print(confusion_matrix(gt_labels,pred_labels_patch))
    patch_bacc = balanced_accuracy_score(gt_labels, pred_labels_patch)
    report = classification_report(gt_labels, pred_labels_patch, output_dict=True, zero_division=0)
    patch_wf1 = report['weighted avg']['f1-score']
    
    results['patch_bacc'] = patch_bacc
    results['patch_wf1'] = patch_wf1
    results['logits'] = all_logits
    
    if vision_mil:
        print('***** wsi cls token result:')
        print(confusion_matrix(gt_labels,pred_labels_mil))
        wsi_bacc = balanced_accuracy_score(gt_labels, pred_labels_mil)
        report = classification_report(gt_labels, pred_labels_mil, output_dict=True, zero_division=0)
        wsi_wf1 = report['weighted avg']['f1-score']
        
        ensemble_probs = (torch.stack(pred_probs_patch) + torch.stack(pred_probs_mil))/2
        ensemble_labels = torch.argmax(ensemble_probs, dim=1).tolist()
        
        print('***** ensemble result:')
        print(confusion_matrix(gt_labels,ensemble_labels))
        ensemble_bacc = balanced_accuracy_score(gt_labels, ensemble_labels)
        report = classification_report(gt_labels, ensemble_labels, output_dict=True, zero_division=0)
        ensemble_wf1 = report['weighted avg']['f1-score']

        results['wsi_bacc'] = wsi_bacc
        results['wsi_wf1'] = wsi_wf1
        
        results['ensemble_bacc'] = ensemble_bacc
        results['ensemble_wf1'] = ensemble_wf1

    return results

def simple_dice_auc(model, test_loader, device, batch_classifier=None):
    model.eval()
    wf1_list = []
    bacc_list = []
    
    gt_label = []
    pred_label = []
    for data, label in test_loader:
        data = data.squeeze(0).to(device)
        label = label.squeeze(0).cpu().numpy()
        with torch.no_grad():
            logit, _ = model(data)
        
        patch_pred = torch.argmax(logit, dim=-1).cpu().numpy()
        
        cm = confusion_matrix(label,patch_pred)
        # print(cm)
        bacc = balanced_accuracy_score(label, patch_pred)
        report = classification_report(label, patch_pred, output_dict=True, zero_division=0)
        weighted_f1 = report['weighted avg']['f1-score']
        
        # inter_num = np.count_nonzero(label * patch_pred)
        # all_num = np.count_nonzero(label) + np.count_nonzero(patch_pred)
        # dice_score = 2 * inter_num / all_num
        # auc_score = roc_auc_score(label, logit.cpu().numpy()[:, 1])
        
        wf1_list.append(weighted_f1)
        bacc_list.append(bacc)

        gt_label.extend(list(label))
        pred_label.extend(list(patch_pred))
    
    mean_bacc = sum(bacc_list) / len(bacc_list)
    mean_wf1 = sum(wf1_list) / len(wf1_list)

    # print(gt_labels)
    # print(pred_labels)
    
    overall_cm = confusion_matrix(gt_label,pred_label)
    overall_acc = np.diag(overall_cm).sum()/overall_cm.sum()
    print(overall_cm)
    print('overall acc is %.4f.'% (overall_acc))
    
    return [mean_bacc, mean_wf1]


def blocks_to_image_fast(coords, predictions, block_size=224):
    """
    快速向量化版本
    """
    coords = np.array(coords)
    predictions = np.array(predictions)
    
    # 向量化计算块坐标
    block_coords = coords // block_size
    
    # 计算偏移量和图像尺寸
    min_coords = block_coords.min(axis=0)
    max_coords = block_coords.max(axis=0)
    
    height = max_coords[1] - min_coords[1] + 1
    width = max_coords[0] - min_coords[0] + 1
    
    # 计算相对坐标
    rel_coords = block_coords - min_coords
    
    # 向量化填充
    image = np.zeros((height, width), dtype=np.float32)
    image[rel_coords[:, 1], rel_coords[:, 0]] = predictions
    
    return image


def blocks_to_image_fine(coords, predictions, ori_size, block_size=224):
    """
    可视化版本
    """
    coords = np.array(coords)
    predictions = np.array(predictions)
    
    # 向量化计算块坐标
    block_coords = coords // block_size

    height = (ori_size[1] // block_size) + 1
    width = (ori_size[0] // block_size) + 1
    
    # 向量化填充
    image = np.zeros((height, width), dtype=np.float32)
    # print(block_coords.shape, predictions.shape)
    image[block_coords[:, 1], block_coords[:, 0]] = predictions
    
    return image


def draw_mask(model, test_loader, device, model_type='fewshot', prompt_embedding=None, save_mask=True, save_path=None):
    model.eval()
    dice_scores = []
    
    test_loader.dataset.seg_flag = True  # Enable segmentation flag for WSI data loader
    
    for data, coords, labels, wsi_label, slide_id in tqdm(test_loader):
        labels_squeezed = labels.squeeze(0)
        if labels_squeezed.numel() == 1 and labels_squeezed.item() == -1:
            print(f"Warning: No labels for slide {slide_id}. Skipping this slide.")
            continue
        data = data.float().squeeze(0)
        labels = labels.squeeze(0).cpu().numpy()
        coords = coords.squeeze(0).cpu().numpy()
        wsi_label = wsi_label.item() if isinstance(wsi_label, torch.Tensor) else wsi_label
        wsi_label += 1
        # print(wsi_label, slide_id)
        if np.sum(labels == wsi_label) == 0:
            print(f"Error: No patches with label {wsi_label} in slide {slide_id}.")
            continue
        
        # print(data.shape, labels.shape, coords.shape)
        # print({0: np.sum(labels == 0), 1: np.sum(labels == 1)})
        
        # 设置批次大小
        batch_size = 16384  # 你可以根据显存情况调整这个值
        n_patches = data.shape[0]
        all_logits = []
        
        with torch.no_grad():
            for i in range(0, n_patches, batch_size):
                end_idx = min(i + batch_size, n_patches)
                data_batch = data[i:end_idx].to(device)
                
                if model_type=='zeroshot':
                    logits_batch = model(data_batch) # (batch_size, n_cls) or (batch_size, n_cls+1) with normal
                elif model_type=='fewshot':
                    logits_batch, _ = model(data_batch) # (batch_size, n_cls) or (batch_size, n_cls+1) with normal
                elif model_type =='prompting':
                    data_batch = data_batch / data_batch.norm(dim=-1, keepdim=True)
                    prompt_embedding = prompt_embedding / prompt_embedding.norm(dim=-1, keepdim=True)
                    logits_batch = data_batch @ prompt_embedding.t()
                    logits_batch = torch.nn.functional.softmax(logits_batch*10,-1)
                elif model_type =='fewshot-mil':
                    _, logits_batch = model(data_batch)
                
                all_logits.append(logits_batch.cpu())
                torch.cuda.empty_cache()
                
            # 拼接所有批次的结果
            logits = torch.cat(all_logits, dim=0)
            
            # print(logits.shape)
        # patch_size = test_loader.dataset.patch_size
        patch_size = 512
        thd = test_loader.dataset.thd
        
        # print(logits.shape, coords.shape, labels.shape, wsi_label, slide_id)
        # predictions = logits[:,int(wsi_label)].cpu().numpy() > thd
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        if not save_mask:
            pred_mask = blocks_to_image_fast(coords, predictions, block_size=patch_size)
            gt_mask = blocks_to_image_fast(coords, labels, block_size=patch_size)
        else:
            slide_name = slide_id[0] if isinstance(slide_id, (list, tuple)) else slide_id
            wsi_path = test_loader.dataset.get_wsi_path(slide_name)
            ori_size = openslide.open_slide(wsi_path).dimensions
            # print(coords.shape, predictions.shape, labels.shape, ori_size, patch_size)
            pred_mask = blocks_to_image_fine(coords, predictions, ori_size, block_size=patch_size)
            gt_mask = blocks_to_image_fine(coords, labels, ori_size, block_size=patch_size)
        
        gt_mask_binary = (gt_mask == wsi_label).astype(np.float32)
        pred_mask_binary = (pred_mask == wsi_label).astype(np.float32)
        inter_mask = gt_mask_binary*pred_mask_binary
        intersection_sum = np.count_nonzero(inter_mask)
        pred_sum = np.count_nonzero(pred_mask_binary)
        gt_sum = np.count_nonzero(gt_mask_binary)

        # safe_divide = gt_sum + pred_sum
        # assert safe_divide != 0, 'mask_sum and pred_sum are both zero!'
        # if safe_divide == 0:
        #     # print(f"Warning: Both gt_sum and pred_sum are zero for slide {slide_id} with label {wsi_label}. Skipping this slide.")
        #     continue

        dice_per_slide = 2*intersection_sum/(gt_sum + pred_sum)
        dice_scores.append(dice_per_slide)
        
        if save_mask:
            num_classes = test_loader.dataset.num_classes # subtype number, no normal
            scale = 255 // num_classes
            dice_str = f"{dice_per_slide:.3f}"
            slide_name = slide_id[0] if isinstance(slide_id, (list, tuple)) else slide_id
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            # Save ground truth mask
            if model_type == 'zeroshot':
                gt_path = os.path.join(save_path, 'gt')
                if not os.path.exists(gt_path):
                    os.makedirs(gt_path)
                gt_filename = os.path.join(gt_path, f"{slide_name}_gt.png")
                cv2.imwrite(gt_filename, (gt_mask * scale).astype(np.uint8))
            
            # Save prediction mask
            pred_path = os.path.join(save_path, model_type)
            if not os.path.exists(pred_path):
                os.makedirs(pred_path)
            pred_filename = os.path.join(pred_path, f"{dice_str}_{slide_name}_pred.png")
            cv2.imwrite(pred_filename, (pred_mask * scale).astype(np.uint8))
        
    test_loader.dataset.seg_flag = False  # Disable segmentation flag after evaluation

    return sum(dice_scores)/len(dice_scores)