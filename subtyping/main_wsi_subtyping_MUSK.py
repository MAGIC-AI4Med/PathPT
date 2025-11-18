import os
import torch
import torch.optim as optim
import timm
import musk.utils as musk_utils
import musk.modeling as musk_modeling
from transformers import XLMRobertaTokenizer
from torch.utils.data import DataLoader
from models.PathPT_model_MUSK import OriginMUSK, CustomMUSK, PPTMUSK
import torch.nn.functional as F
from WSI_dataset import WSI_Data
import random
import utils
import numpy as np
import evaluation
import params
import logging
from tqdm import tqdm
from wsi_selecters.wsi_selecter_MUSK import patch_selector, wsi_prompt_selector
import json
from torch.cuda.amp import autocast, GradScaler
from loss import calpatch_loss, PatchSSLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main_subtyping(dataset_name, cfg = params.PromptLearnerConfig(input_size=256),  metric = evaluation.simple_dice_auc, given_param=None):
    
    if given_param is not None:
        param = given_param
    else:
        param = params.subtype_params[dataset_name]

    musk_path = params.MUSK_PATH
    with open(params.DATASET_DIVISION, 'r') as f:
        meta = json.load(f)[dataset_name.upper()]
    name2label = meta['name2label']
    subtype_classnames = sorted(name2label.keys(), key=lambda x: name2label[x])
    subtype_classnames = ['Normal'] + subtype_classnames
    print(dataset_name)
    print(subtype_classnames)
    zeroshot_prompt_lst,classnames_list = utils.load_prompts(dataset_name, subtype_classnames)
    
    multiple_trains_and_eval(
        cfg=cfg,
        ckpt_path=musk_path,
        zeroshot_prompt_lst=zeroshot_prompt_lst,
        metric=metric,
        classnames_list=classnames_list,
        param = param,
        train_info = meta,
    )


def multiple_trains_and_eval(cfg, 
                             ckpt_path, 
                             zeroshot_prompt_lst, 
                             metric, 
                             classnames_list=None, 
                             param = None,
                             train_info = None): 
    
    repeats = param['repeats']
    device = param['device']
    musk_model = timm.create_model("musk_large_patch16_384")
    musk_utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", musk_model, 'model|module', '', local_dir = ckpt_path)
    musk_model.to(device=device, dtype=torch.float32)
    musk_tokenizer = XLMRobertaTokenizer("./musk/models/tokenizer.spm")

    print(device)

    for i in range(repeats):
        
        ## data loader
        train_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                feature_path = param['musk_feature_root'],
                                fold = i,
                                patch_num=param['patch_num'],
                                state='train')
        train_loader = DataLoader(train_dataset, batch_size=param['batch_size'], num_workers=8, shuffle=True)
        
        train_wsi_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                feature_path = param['musk_feature_root'],
                                fold = i,
                                state='train')
        train_wsi_loader = DataLoader(train_wsi_dataset, batch_size=1, num_workers=8, shuffle=True)
        
        val_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                feature_path = param['musk_feature_root'],
                                fold = i,
                                # patch_num=param['patch_num'],
                                state='val',
                                train_info = train_info)
        val_wsi_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, shuffle=True)
        
        test_wsi_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                feature_path = param['musk_feature_root'],
                                fold = i,
                                state='test')
        test_wsi_loader = DataLoader(test_wsi_dataset, batch_size=1, num_workers=8, shuffle=False)
        
        
        logging.info("======== ↓↓ Experiment No.{} ↓↓ ========".format(i))
        
        selected_prompt_embedding = None
        if param['prompt_select']:
            print('Manual prompt selecting...')
            selected_prompt_embedding = wsi_prompt_selector(musk_model, musk_tokenizer, train_wsi_loader, zeroshot_prompt_lst, device=device)

            train_results_dict = metric(musk_model, train_wsi_loader, device, batch_classifier=None, model_type = 'prompting', prompt_embedding = selected_prompt_embedding, save_logits = True)
            prompting_train_result = np.array([train_results_dict['patch_bacc'],train_results_dict['patch_wf1']])
            logging.info(f"Manual prompting on trainset{prompting_train_result}")
            
            test_results_dict = metric(musk_model, test_wsi_loader, device, batch_classifier=None, model_type = 'prompting', prompt_embedding = selected_prompt_embedding)
            prompting_test_result = np.array([test_results_dict['patch_bacc'],test_results_dict['patch_wf1']])
            logging.info(f"Manual prompting on testset{prompting_test_result}")
        
        random.seed(i)
        zeroshot_prompt = []
        for cls_prompt in zeroshot_prompt_lst:
            index = random.randint(0, len(cls_prompt) - 1)
            zeroshot_prompt.append(cls_prompt[index])
        
        zero_shot_model = OriginMUSK(zeroshot_prompt, musk_model, musk_tokenizer, device)
        zero_shot_results_dict = metric(zero_shot_model, test_wsi_loader, device, batch_classifier=None, model_type = 'zeroshot', save_dir=f'./fewshot_results/{os.path.basename(param["log_name"])[:-4]}/zeroshot')
        zero_shot_result = np.array([zero_shot_results_dict['patch_bacc'],zero_shot_results_dict['patch_wf1']])
        logging.info(f"Zeroshot result{zero_shot_result}")

        logging.info('>>>>>>>>> Shots: ' + str(param['shot']) + ' <<<<<<<<<')
        
        ## train & validadte the model ###
        print('step 3. Prompt learning...')
        torch.cuda.empty_cache()
        train_anno(cfg,classnames_list,
                    musk_model, 
                    musk_tokenizer,
                    train_loader, 
                    train_wsi_loader,
                    val_wsi_loader,
                    test_wsi_loader, 
                    metric, 
                    param = param,
                    selected_prompt_embedding = selected_prompt_embedding, 
                    device=device,
                    save_dir = f'./fewshot_results/{os.path.basename(param["log_name"])[:-4]}/fold{i}'
                )


def train_anno(cfg, 
               classnames_list,
               musk_model,
               musk_tokenizer,
               train_loader, 
               train_wsi_loader,
               val_wsi_loader,
               test_loader, 
               metric, 
               param,
               selected_prompt_embedding = None, 
               device="cuda:0", 
               save_dir=None
            ):
    
    init_model, optimizer, scheduler = model_init(cfg, classnames_list, musk_model, musk_tokenizer, device, param, vfeat_dim = 1024)
    model_v1 = train_one_step(param['epochs'], 
                              init_model, 
                              device, 
                              param, 
                              train_loader, 
                              train_wsi_loader,
                              val_wsi_loader,
                              musk_model, 
                              'zeroshot', 
                              optimizer, 
                              scheduler, 
                              metric, 
                              test_loader, 
                              selected_prompt_embedding = selected_prompt_embedding,
                              enable_pseudo = param['enable_pseudo'],
                              save_dir=save_dir
                              )

    
    
def model_init(cfg, classnames_list, musk_model, musk_tokenizer, device, param, vfeat_dim):
    model = PPTMUSK(cfg, classnames_list, musk_model, musk_tokenizer, device, param, vfeat_dim)
    
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


    
def train_one_step(epoch, model, device, param, train_loader, train_wsi_loader, val_wsi_loader, label_model, label_type, optimizer, scheduler, metric, test_loader, selected_prompt_embedding = None, enable_pseudo = True, save_dir=None):
    for i in range(epoch):
        ### train the model ###
        model.train()
        model.to(device)
        
        running_loss = torch.zeros(1, device=device)

        accumulation_steps = param['accumulation_steps']
        scaler = GradScaler()
        
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {i + 1}/{epoch}",
            ncols=100,
            leave=True
        )

        for step, (data, manual_labels, label, idx) in progress_bar:
            if not param['vision_only']:
                patch_labels = patch_selector(label_model=label_model,
                                            label_type = label_type,
                                                dataset_name=param['dataset_name'], 
                                                vision_feats = data.float(), 
                                                wsi_label = label,
                                                logits_thd = param['logits_thd'], 
                                                text_embedding = selected_prompt_embedding,
                                                device=device
                                                )
                patch_labels = torch.tensor(patch_labels).to(device)
            
            torch.cuda.empty_cache()
            
            data = data.float().to(device)
            label = label.to(device)
                        
            with autocast(): 
                wsi_logits, patch_logits = model(data)
                if patch_logits is not None:
                    patch_loss = PatchSSLoss(patch_logits, patch_labels, epoch=i, total_epoch=epoch, weights=param['loss_weight'], balance=param['balance'], vision_only = param['vision_only'], pseudo_loss = enable_pseudo)
                
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

                running_loss += loss.item()
                torch.cuda.empty_cache()

        
        scheduler.step()
        ### evaluate the model ###
        model.eval()
        current_lr = scheduler.get_last_lr()[0]
        with torch.no_grad():
            if isinstance(metric,dict):
                train_patch_result_dict = metric['patch'](model, test_loader['patch'], device=device, batch_classifier=None)
                val_result_dict = metric['wsi'](model, test_loader['wsi'], device=device, batch_classifier=None, save_dir=save_dir)
            else:
                val_result_dict = metric(model, test_loader, device=device, batch_classifier=None, model_type = 'fewshot-mil', vision_only = param['vision_only'], save_dir=save_dir)
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
            
            logging.info(f"Epoch {i} loss: {running_loss.item()*accumulation_steps / len(train_loader):.4f}, LR: {current_lr:.8f}, patch_test:{patch_rounded}")
        
    return model