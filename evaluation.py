import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
import numpy as np
import warnings
from tqdm import tqdm
import os
warnings.filterwarnings("ignore", category=UserWarning)

def sub_bacc_wf1(model, test_loader, device, normal_ext=False, batch_classifier=None, model_type = 'fewshot', prompt_embedding = None, save_logits=True, save_dir = None, vision_only = False):
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
        
        # batch processing
        batch_size = 50000
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
            
            # concatenate all patch logits
            patch_logits = torch.cat(all_patch_logits, dim=0)  # (n_patch, n_cls)
            
            # for fewshot-mil
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
                patch_logits = patch_logits[:, 1:] # no normal WSI
            # wsi_label -= 1 # label in testset start from 1
            
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
            
            
    # BACC & WF1
    
    results = {}
    if len(pred_labels_patch) > 0:
        print('***** patch result:')
        cm = confusion_matrix(gt_labels,pred_labels_patch)
        print(cm)
        patch_bacc = balanced_accuracy_score(gt_labels, pred_labels_patch)
        report = classification_report(gt_labels, pred_labels_patch, output_dict=True, zero_division=0)
        patch_wf1 = report['weighted avg']['f1-score']
        
        results['patch_bacc'] = patch_bacc
        results['patch_wf1'] = patch_wf1
        results['logits'] = all_logits

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.save(save_dir + '/gt_labels.npy', np.array(gt_labels))
            np.save(save_dir + '/pred_labels.npy', np.array(pred_labels_patch))
            np.save(save_dir + '/confusion_matrix_patch.npy', cm)
            if model_type != 'zeroshot':
                torch.save(model.prompt_learner.ctx, save_dir + '/prompt_embedding.pt')
                torch.save(model.mlp.state_dict(), save_dir + '/spatial_aware_module.pt')
    
    if vision_only:
        print('***** wsi cls token result:')
        cm = confusion_matrix(gt_labels,pred_labels_mil)
        print(cm)
        wsi_bacc = balanced_accuracy_score(gt_labels, pred_labels_mil)
        report = classification_report(gt_labels, pred_labels_mil, output_dict=True, zero_division=0)
        wsi_wf1 = report['weighted avg']['f1-score']

        results['wsi_bacc'] = wsi_bacc

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
        
        wf1_list.append(weighted_f1)
        bacc_list.append(bacc)

        gt_label.extend(list(label))
        pred_label.extend(list(patch_pred))
    
    mean_bacc = sum(bacc_list) / len(bacc_list)
    mean_wf1 = sum(wf1_list) / len(wf1_list)
    
    overall_cm = confusion_matrix(gt_label,pred_label)
    overall_acc = np.diag(overall_cm).sum()/overall_cm.sum()
    print(overall_cm)
    print('overall acc is %.4f.'% (overall_acc))
    
    return [mean_bacc, mean_wf1] 