import os
import torch
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader
from PathPL_model_KEEP import OriginKEEP, CustomKEEP, PPLKEEP
import torch.nn.functional as F
from WSI_dataset import WSI_Data
import random
import utils
import numpy as np
import evaluation
import params
import logging
from transformers import AutoModel, AutoTokenizer
from datetime import datetime
from tqdm import tqdm
from wsi_selecter_KEEP import patch_selector, wsi_prompt_selector
from torchsampler import ImbalancedDatasetSampler
import json
from data import SelectDataset
from collections import Counter
from torch.cuda.amp import autocast, GradScaler
from loss import calpatch_loss, PatchSSLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main_subtyping(dataset_name, cfg = params.PromptLearnerConfig(input_size=256),  metric = evaluation.simple_dice_auc, save_name = None, given_param=None):
    
    if given_param is not None:
        param = given_param
    else:
        param = params.subtype_params[dataset_name]

    keep_path = params.KEEP_PATH
    # zero_shot_template_file = param['zero_shot_template_file']
    with open(params.DATASET_DIVISION, 'r') as f:
        meta = json.load(f)[dataset_name.upper()]
    name2label = meta['name2label']
    subtype_classnames = sorted(name2label.keys(), key=lambda x: name2label[x])
    # subtype_classnames = gen_loader.get_classnames()
    subtype_classnames = ['Normal'] + subtype_classnames
    print(subtype_classnames)
    # csv_classnames = ['Normal', 'IDC', 'ILC'] #load_prompts_from_template识别不了首字母缩写
    zeroshot_prompt_lst,classnames_list = utils.load_prompts(dataset_name, subtype_classnames)
    
    multiple_trains_and_eval(
        cfg=cfg,
        ckpt_path=keep_path,
        zeroshot_prompt_lst=zeroshot_prompt_lst,
        metric=metric,
        classnames_list=classnames_list,
        param = param,
        train_info = meta,
    )

    # save_root = './results/'
    # if not os.path.exists(save_root):
    #     os.makedirs(save_root)

    # all_results['params'] = param
    # json_str = json.dumps(all_results, indent=2)
    # with open(save_root + save_name + '.json', 'w') as json_file:
    #     json_file.write(json_str)
    
    # zeroshot_bacc = np.array(all_results['zero-shot'])[:,0]
    # q1 = np.percentile(zeroshot_bacc, 25)
    # median = np.percentile(zeroshot_bacc, 50)
    # q3 = np.percentile(zeroshot_bacc, 75)
    # print('zeroshot results: %.4f (%.4f, %.4f)'%(median, q1, q3))
    
    # fewshot_bacc = np.array(all_results[str(shot) + '-shot'])[:,0]
    # q1 = np.percentile(fewshot_bacc, 25)
    # median = np.percentile(fewshot_bacc, 50)
    # q3 = np.percentile(fewshot_bacc, 75)
    # print('fewshot results: %.4f (%.4f, %.4f)'%(median, q1, q3))


def multiple_trains_and_eval(cfg, 
                             ckpt_path, 
                             zeroshot_prompt_lst, 
                             metric, 
                             classnames_list=None, 
                             param = None,
                             train_info = None): 
    
    repeats = param['repeats']
    device = param['device']
    val_kwargs = {
                  'wsi_dir_path': param['wsi_dir_path'],
                  'patch_size': 256,
                  'thd': 0.5}
    
    keep_model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True)
    keep_tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    keep_model.to(device)
    
    # conch_model.to(device)
    zeroshot_results = []
    fewshot_results = []
    print(device)

    for i in range(repeats):
        
        ## data loader
        train_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                feature_path = param['keep_feature_root'],
                                train_path = param["train_dir_path"],
                                fold = i,
                                patch_num=param['patch_num'],
                                state='train',
                                val_kwargs=val_kwargs,)
        train_loader = DataLoader(train_dataset, batch_size=param['batch_size'], num_workers=8, shuffle=True)
        
        train_wsi_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                feature_path = param['keep_feature_root'],
                                train_path = param["train_dir_path"],
                                fold = i,
                                state='train',
                                val_kwargs=val_kwargs,)
        train_wsi_loader = DataLoader(train_wsi_dataset, batch_size=1, num_workers=8, shuffle=True)
        
        val_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                feature_path = param['keep_feature_root'],
                                train_path = param["train_dir_path"],
                                fold = i,
                                # patch_num=param['patch_num'],
                                state='val',
                                train_info = train_info,
                                val_kwargs=val_kwargs)
        val_wsi_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, shuffle=True)
        
        test_wsi_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                feature_path = param['keep_feature_root'],
                                fold = i,
                                state='test',
                                val_kwargs=val_kwargs)
        test_wsi_loader = DataLoader(test_wsi_dataset, batch_size=1, num_workers=8, shuffle=False)
        
        
        logging.info("======== ↓↓ Experiment No.{} ↓↓ ========".format(i))
        
        # unique_classnames_lst = utils.unique_classnames(classnames_lst)
        # print(f'Ensemble classnames: {classnames_lst}')
        # origin_model = CustomCONCH(cfg, unique_classnames_lst, conch_model, device, mode)
        origin_model = CustomKEEP(cfg, classnames_list, keep_model, keep_tokenizer, device, param, vfeat_dim=768)
        
        # test_loader = dataloader_generator.gen_testloader(metric)
        # train_feat_paths, val_feat_paths = dataloader_generator.split_train_val(seed = i)
        # train_wsi_loader = dataloader_generator.gen_wsiloader(train_feat_paths)
        # val_wsi_loader = dataloader_generator.gen_wsiloader(val_feat_paths)
        
        selected_prompt_embedding = None
        if param['prompt_select']:
            # train_dataset_list = dataloader_generator.trainset.get_example_paths()
            print('step 1. Selecting prompts...')
            selected_prompt_embedding = wsi_prompt_selector(train_wsi_loader, zeroshot_prompt_lst, device=device)

            train_results_dict = metric(origin_model, train_wsi_loader, device, batch_classifier=None, model_type = 'prompting', prompt_embedding = selected_prompt_embedding, save_logits = True)
            prompting_train_result = np.array([train_results_dict['patch_bacc'],train_results_dict['patch_wf1']])
            logging.info(f"Prompting train result{prompting_train_result}")
            
            test_results_dict = metric(origin_model, test_wsi_loader, device, batch_classifier=None, model_type = 'prompting', prompt_embedding = selected_prompt_embedding)
            prompting_test_result = np.array([test_results_dict['patch_bacc'],test_results_dict['patch_wf1']])
            # print(f'Prompting result{result}')
            logging.info(f"Prompting test result{prompting_test_result}")
        
        random.seed(i)
        zeroshot_prompt = []
        for cls_prompt in zeroshot_prompt_lst:
            index = random.randint(0, len(cls_prompt) - 1)
            zeroshot_prompt.append(cls_prompt[index])
        # print(f'zeroshot prompt:{prompt}')
        # zero_shot_model = OriginCONCH(prompt, conch_model, device)
        
        zero_shot_model = OriginKEEP(zeroshot_prompt, keep_model, keep_tokenizer, device)
        zero_shot_results_dict = metric(zero_shot_model, test_wsi_loader, device, batch_classifier=None, model_type = 'zeroshot')
            # zeroshot_results.append(zero_shot_result)
            # print(f'zeroshot result{zero_shot_result}')
        zero_shot_result = np.array([zero_shot_results_dict['patch_bacc'],zero_shot_results_dict['patch_wf1']])
        logging.info(f"Zeroshot result{zero_shot_result}")
        
        # zeroshot_train_dice = evaluation.compute_dice(zero_shot_model, train_wsi_loader, device=device, zeroshot=True)
        # zeroshot_val_dice = evaluation.compute_dice(zero_shot_model, val_wsi_loader, device=device, zeroshot=True)
        zeroshot_train_dice = evaluation.draw_mask(zero_shot_model, train_wsi_loader, device=device, model_type='zeroshot', save_path=f'mask/{param["dataset_name"]}/{param["log_name"]}/train')
        zeroshot_val_dice = evaluation.draw_mask(zero_shot_model, val_wsi_loader, device=device, model_type='zeroshot', save_path=f'mask/{param["dataset_name"]}/{param["log_name"]}/val')
        logging.info(f"Zeroshot dice{[zeroshot_train_dice, zeroshot_val_dice]}")

        logging.info('>>>>>>>>> Shots: ' + str(param['shot']) + ' <<<<<<<<<')
        
        ## train & validadte the model ###
        print('step 3. Prompt learning...')
        torch.cuda.empty_cache()
        train_anno(cfg,classnames_list,
                    keep_model, 
                    keep_tokenizer,
                    train_loader, 
                    train_wsi_loader,
                    val_wsi_loader,
                    test_wsi_loader, 
                    metric, 
                    param = param,
                    selected_prompt_embedding = selected_prompt_embedding, 
                    device=device,
                )
  
  
        ### test the model ###
        # best_model.eval()
        # with torch.no_grad():
        #     fresult = metric(best_model, test_wsi_loader, device=device, batch_classifier=None)
        # # print(f'result{result}')
        # logging.info(f"Best result{fresult}")
        # fewshot_results.append(fresult)

    # all_results = dict()
    # all_results['zero-shot'] = zeroshot_results
    # all_results[str(param['shot'])+'-shot'] = fewshot_results
    
    # return all_results


def train_anno(cfg, 
               classnames_list,
               keep_model,
               keep_tokenizer,
               train_loader, 
               train_wsi_loader,
               val_wsi_loader,
               test_loader, 
               metric, 
               param,
               selected_prompt_embedding = None, 
               device="cuda:0", 
            ):
    
    ## first step
    init_model, optimizer, scheduler = model_init(cfg, classnames_list, keep_model, keep_tokenizer, device, param, vfeat_dim = 768)

    print('******* Setp 1 ********')
    model_v1 = train_one_step(param['epochs'], 
                              init_model, 
                              device, 
                              param, 
                              train_loader, 
                              train_wsi_loader,
                              val_wsi_loader,
                              keep_model, 
                              'zeroshot', 
                              optimizer, 
                              scheduler, 
                              metric, 
                              test_loader, 
                              selected_prompt_embedding = selected_prompt_embedding,
                              enable_pseudo = param['enable_pseudo'])
    
    # ## second step
    # init_model, optimizer, scheduler = model_init(cfg, classnames_list, keep_model, keep_tokenizer, device, param, vfeat_dim = 768)

    # print('******* Setp 2 ********')
    # model_v2 = train_one_step(param['epochs'], 
    #                             init_model, device, 
    #                             param, 
    #                             train_loader, 
    #                             model_v1, 
    #                             'fewshot-mil',
    #                             optimizer, 
    #                             scheduler, 
    #                             metric, 
    #                             test_loader, 
    #                             selected_prompt_embedding = None,
    #                             enable_pseudo = True)
    
    
def model_init(cfg, classnames_list, keep_model, keep_tokenizer, device, param, vfeat_dim):
    model = PPLKEEP(cfg, classnames_list, keep_model, keep_tokenizer, device, param, vfeat_dim)
    
    ### Freeze all parameters except prompt_learner.ctx ###
    for name, parameters in model.named_parameters():
        if not (name.startswith('prompt_learner.ctx')
                or name.startswith('mlp.')
                or name.startswith('prompt_')
                or name.startswith('mil.')):
            parameters.requires_grad = False
            
    ### Set optimizer and scheduler ###
    optimizer = optim.Adam(model.parameters(), lr=param['lr'])
    warm_up_iter = int(param['epochs'] * 0.1)
    lambda0 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
            0.5 * (1.0 + np.cos(np.pi * (cur_iter - warm_up_iter) / (param['epochs'] - warm_up_iter)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    
    return model, optimizer, scheduler

def compute_mean_iou(label, prediction, cls):
    # 计算每列的 TP, FP, FN
    all_iou = []
    for i in range(len(cls)):
        cls_i = cls[i]
        label_i = label[i,:]
        pred_i = prediction[i,:]
        tp = ((label_i == cls_i) & (pred_i == cls_i)).sum()
        fp = ((pred_i == cls_i) & (label_i != cls_i)).sum()
        fn = ((label_i == cls_i) & (pred_i != cls_i)).sum()
        
        # 计算每列 IoU（避免除零）
        denominator = tp + fp + fn
        iou = tp/denominator if denominator !=0 else 0
        # iou = np.divide(tp, denominator, out=np.zeros_like(tp, dtype=float), where=(denominator != 0))
    # iou = np.where((tp + fp + fn) > 0, tp / (tp + fp + fn), 0.0)
        all_iou.append(iou)
        
    # 返回平均 IoU 和每列 IoU
    return np.mean(np.array(all_iou)), all_iou

def compute_mean_dice(label, prediction, cls):
    # 计算每列的 TP, FP, FN
    all_dice = []
    for i in range(len(cls)):
        cls_i = cls[i]
        label_i = label[i,:]
        pred_i = prediction[i,:]
        tp = ((label_i == cls_i) & (pred_i == cls_i)).sum()
        fp = ((pred_i == cls_i) & (label_i != cls_i)).sum()
        fn = ((label_i == cls_i) & (pred_i != cls_i)).sum()
        
        # 计算每列 Dice系数（避免除零）
        denominator = 2 * tp + fp + fn
        dice = (2 * tp) / denominator if denominator != 0 else 0
        all_dice.append(dice)
        
    # 返回平均 Dice 和每列 Dice
    return np.mean(np.array(all_dice)), all_dice
    
def train_one_step(epoch, model, device, param, train_loader, train_wsi_loader, val_wsi_loader, label_model, label_type, optimizer, scheduler, metric, test_loader, selected_prompt_embedding = None, enable_pseudo = True):
    tumor_dice = []
    for i in range(epoch):
        ### train the model ###
        # dataloader.dataset.shuffle_data()
        model.train()
        model.to(device)
        
        running_loss = torch.zeros(1, device=device)

        accumulation_steps = param['accumulation_steps']  # 累积梯度再更新
        scaler = GradScaler()
        
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {i + 1}/{epoch}",
            ncols=100,  # 进度条宽度
            leave=True  # 保留进度条痕迹
        )
        
        # epoch_dice = []
        # epoch_idx = []
        for step, (data, manual_labels, label, idx) in progress_bar:
            # change classnames
            # model.change_classnames()
            
            # print('step 2. Patch labeling...')
            if not param['vision_only']:
                patch_labels = patch_selector(label_model=label_model,
                                            label_type = label_type,
                                                dataset_name=param['dataset_name'], 
                                                vision_feats = data, 
                                                wsi_label = label,
                                                logits_thd = param['logits_thd'], 
                                                text_embedding = selected_prompt_embedding,
                                                device=device
                                                )
                patch_labels = torch.tensor(patch_labels).to(device)
            
            torch.cuda.empty_cache()
            
            # 统计每种标签的数量
            # view_labels = list(patch_labels.view(-1).cpu().numpy())
            # label_counts = Counter(view_labels)
            # for patch_label, count in label_counts.items():
            #     print(f"label {patch_label}: {count} instances")
            
            data = data.to(device)
            label = label.to(device)
            
            ## merge pesudo_labels and manual_labels
            # manual_labels = manual_labels.to(device)
            # mask = (manual_labels.squeeze() >= 0) & (patch_labels >= 0) & (manual_labels.squeeze() == patch_labels)
            # if len(label) > 1:
            #     replace_label = label.view(len(label),1)
            # else:
            #     replace_label = label.item()
            # patch_labels = torch.where(mask, patch_labels, -1*(replace_label+1))
            
            # for iii in range(patch_labels.shape[0]):
            #     aaa = patch_labels[iii,:].cpu().tolist()
            #     label_counts = Counter(aaa)
            #     for p_label, count in label_counts.items():
            #         print(f"label {p_label}: {count} instances")
            
            # view_labels = list(patch_labels.view(-1).cpu().numpy())
            # label_counts = Counter(view_labels)
            # for patch_label, count in label_counts.items():
            #     print(f"label {patch_label}: {count} instances")
                        
            with autocast(): 
                wsi_logits, patch_logits = model(data)
                if patch_logits is not None:
                    # patch_loss = balanced_ce_loss(patch_logits, patch_labels)
                    patch_loss = PatchSSLoss(patch_logits, patch_labels, epoch=i, total_epoch=epoch, weights=param['loss_weight'], balance=param['balance'], vision_only = param['vision_only'], pseudo_loss = enable_pseudo)

                    # compute dice with manual labels
                    # if 'return_labels' not in patch_loss:
                    #     _, batch_dice = compute_mean_dice(manual_labels.cpu().numpy(), patch_labels.view(manual_labels.shape[0], manual_labels.shape[1]).cpu().numpy(), label.cpu().numpy()+1)
                    # else:
                    #     return_labels = patch_loss['return_labels']
                    #     return_labels = return_labels.view(manual_labels.shape[0], manual_labels.shape[1]).cpu().numpy()
                    #     _, batch_dice = compute_mean_dice(manual_labels.cpu().numpy(), return_labels, label.cpu().numpy()+1)
                    # epoch_dice.extend(batch_dice)
                    # epoch_idx.extend(idx.cpu().tolist())
                
                if param['vision_only']:
                    loss = F.cross_entropy(wsi_logits, label)
                else:
                    loss = patch_loss
                
                if isinstance(loss,dict):
                    loss = loss['loss']/accumulation_steps
                else:
                    loss = loss/accumulation_steps
                    
            torch.cuda.empty_cache()
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                running_loss += loss.item()
                torch.cuda.empty_cache()

        # if len(epoch_dice) > 0:
        #     combined = sorted(zip(epoch_idx, epoch_dice), key=lambda x: x[0])
        #     # 解压排序后的结果
        #     sorted_epoch_idx, sorted_epoch_dice = zip(*combined)
        #     # print(sorted_epoch_dice)
        #     mean_dice = np.array(sorted_epoch_dice).mean()
        #     tumor_dice.append(mean_dice)
        
        scheduler.step()
        ### evaluate the model ###
        model.eval()
        current_lr = scheduler.get_last_lr()[0]  # 获取当前学习率
        with torch.no_grad():
            # if val_wsi_loader and metric is evaluation.sub_bacc_wf1:
            #     select_result = np.array(metric(model, val_wsi_loader, device=device, batch_classifier=None))
            #     print(f'select_val:{select_result.tolist()}')
            #     if select_result_max <= select_result[0]:
            #         select_result_max = select_result[0]
            #         best_model = deepcopy(model)
            #         print(f'best epoch find: {i}')
            if isinstance(metric,dict):
                train_patch_result_dict = metric['patch'](model, test_loader['patch'], device=device, batch_classifier=None)
                val_result_dict = metric['wsi'](model, test_loader['wsi'], device=device, batch_classifier=None)
            else:
                val_result_dict = metric(model, test_loader, device=device, batch_classifier=None, model_type = 'fewshot-mil', vision_only = param['vision_only'])

                ## compute dice on train and val
                # train_dice = evaluation.compute_dice(model, train_wsi_loader, device=device)
                # val_dice = evaluation.compute_dice(model, val_wsi_loader, device=device)
                train_dice = evaluation.draw_mask(model, train_wsi_loader, device=device, model_type='fewshot-mil', save_mask=(i==epoch-1),save_path=f'mask/{param["dataset_name"]}/{param["log_name"]}/train')
                val_dice = evaluation.draw_mask(model, val_wsi_loader, device=device, model_type='fewshot-mil', save_mask=(i==epoch-1),save_path=f'mask/{param["dataset_name"]}/{param["log_name"]}/val')

        
        # print(f"Epoch {i} loss: {running_loss.item() / len(dataloader)}, val:{val_result.tolist()}, LR: {current_lr:.8f}")        
        if param['vision_only']:
            # patch_result = np.array([val_result_dict['patch_bacc'],val_result_dict['patch_wf1']])
            # patch_rounded = np.around(patch_result, decimals=4).tolist()
            wsi_result = np.array([val_result_dict['wsi_bacc'],val_result_dict['wsi_wf1']])
            wsi_rounded = np.around(wsi_result, decimals=4).tolist()
            # ensemble_result = np.array([val_result_dict['ensemble_bacc'],val_result_dict['ensemble_wf1']])
            # ensemble_rounded = np.around(ensemble_result, decimals=4).tolist()
            logging.info(f"Epoch {i} loss: {running_loss.item()*accumulation_steps / len(train_loader):.4f}, LR: {current_lr:.8f}, wsi:{wsi_rounded}")
        else:
            patch_result = np.array([val_result_dict['patch_bacc'],val_result_dict['patch_wf1']])
            patch_rounded = np.around(patch_result, decimals=4).tolist()
            logging.info(f"Epoch {i} loss: {running_loss.item()*accumulation_steps / len(train_loader):.4f}, LR: {current_lr:.8f}, patch_test:{patch_rounded}, train_dice:{train_dice:.4f}, val_dice:{val_dice:.4f}")
            # logging.info(f"Epoch {i} loss: {running_loss.item()*accumulation_steps / len(train_loader):.4f}, LR: {current_lr:.8f}, patch_test:{patch_rounded}")
        
        # torch.save(model, os.path.join(params.SAVE_DIR, f'epoch_latest.pt'))
    # timestamp = datetime.now().strftime("%m%d%H%M%S")
    # torch.save(model.prompt_learner.ctx, os.path.join(params.SAVE_DIR, f'pt{timestamp}.pt'))
    # print(f'best prompt saved as pt{timestamp}.pt')
    return model
