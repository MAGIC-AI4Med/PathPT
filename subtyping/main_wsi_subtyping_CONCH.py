import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.PathPT_model_CONCH import OriginCONCH, PPTCONCH
import torch.nn.functional as F
from WSI_dataset import WSI_Data
import random
import utils
import numpy as np
import evaluation
import params
import logging
from tqdm import tqdm
import wsi_selecters.wsi_selecter_CONCH as wsi_selecter_CONCH
import json
from torch.cuda.amp import autocast, GradScaler
from loss import calpatch_loss, PatchSSLoss
import open_clip_CONCH as conch_clip

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main_subtyping(dataset_name, cfg = params.PromptLearnerConfig(input_size=256),  metric = evaluation.simple_dice_auc, given_param=None):
    
    if given_param is not None:
        param = given_param
    else:
        param = params.subtype_params[dataset_name]

    model_path = params.CONCH_PATH
    with open(params.DATASET_DIVISION, 'r') as f:
        meta = json.load(f)[dataset_name.upper()]
    name2label = meta['name2label']
    subtype_classnames = sorted(name2label.keys(), key=lambda x: name2label[x])
    subtype_classnames = ['Normal'] + subtype_classnames
    print(subtype_classnames)
    zeroshot_prompt_lst,classnames_list = utils.load_prompts(dataset_name, subtype_classnames)
    
    multiple_trains_and_eval(
        cfg=cfg,
        ckpt_path=model_path,
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
    
    base_model, _ = conch_clip.create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=ckpt_path)
    base_model.to(device)
    print(device)

    for i in range(repeats):
        
        ## data loader
        train_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                feature_path = param['conch_feature_root'],
                                fold = i,
                                patch_num=param['patch_num'],
                                state='train')
        train_loader = DataLoader(train_dataset, batch_size=param['batch_size'], num_workers=8, shuffle=True)
        
        train_wsi_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                feature_path = param['conch_feature_root'],
                                fold = i,
                                state='train')
        train_wsi_loader = DataLoader(train_wsi_dataset, batch_size=1, num_workers=8, shuffle=True)
        
        
        val_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                        feature_path = param['conch_feature_root'],
                        fold = i,
                        state='val',
                        train_info = train_info)
        val_wsi_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, shuffle=True)
        
        test_wsi_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                feature_path = param['conch_feature_root'],
                                fold = i,
                                state='test')
        test_wsi_loader = DataLoader(test_wsi_dataset, batch_size=1, num_workers=8, shuffle=False)
        
        
        logging.info("======== ↓↓ Experiment No.{} ↓↓ ========".format(i))
        
        selected_prompt_embedding = None
        if param['prompt_select']:
            print('Manual prompt selecting...')
            selected_prompt_embedding = wsi_selecter_CONCH.wsi_prompt_selector(base_model, train_wsi_loader, zeroshot_prompt_lst, device=device)
            # selected_prompt_embedding = wsi_selecter.wsi_prompt_selector(train_wsi_loader, zeroshot_prompt_lst, device=device)
            
            train_results_dict = metric(base_model, train_wsi_loader, device, batch_classifier=None, model_type = 'prompting', prompt_embedding = selected_prompt_embedding, save_logits = True)
            prompting_train_result = np.array([train_results_dict['patch_bacc'],train_results_dict['patch_wf1']])
            logging.info(f"Manual prompting on trainset{prompting_train_result}")
            
            test_results_dict = metric(base_model, test_wsi_loader, device, batch_classifier=None, model_type = 'prompting', prompt_embedding = selected_prompt_embedding)
            prompting_test_result = np.array([test_results_dict['patch_bacc'],test_results_dict['patch_wf1']])
            # print(f'Prompting result{result}')
            logging.info(f"Manual prompting on testset{prompting_test_result}")
        
        random.seed(i)
        zeroshot_prompt = []
        for cls_prompt in zeroshot_prompt_lst:
            index = random.randint(0, len(cls_prompt) - 1)
            zeroshot_prompt.append(cls_prompt[index])

        zero_shot_model = OriginCONCH(zeroshot_prompt, base_model, device)
        zero_shot_results_dict = metric(zero_shot_model, test_wsi_loader, device, batch_classifier=None, model_type = 'zeroshot', save_dir=f'./fewshot_results/{os.path.basename(param["log_name"])[:-4]}/zeroshot')
        zero_shot_result = np.array([zero_shot_results_dict['patch_bacc'],zero_shot_results_dict['patch_wf1']])
        logging.info(f"Zeroshot result{zero_shot_result}")

        logging.info('>>>>>>>>> Shots: ' + str(param['shot']) + ' <<<<<<<<<')
        
        ### train & validadte the model ###
        print('Prompt learning...')
        torch.cuda.empty_cache()
        train_anno(cfg,classnames_list,
                    base_model,
                    zero_shot_model, 
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
               base_model,
               label_model,
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
    init_model, optimizer, scheduler = model_init(cfg, classnames_list, base_model, device, param, vfeat_dim = 512)
    model_v1 = train_one_step(param['epochs'], 
                              init_model, 
                              device, 
                              param, 
                              train_loader,
                              train_wsi_loader,
                              val_wsi_loader,
                              label_model, 
                              'zeroshot', 
                              optimizer, 
                              scheduler, 
                              metric, 
                              test_loader, 
                              selected_prompt_embedding = selected_prompt_embedding,
                              enable_pseudo = param['enable_pseudo'],
                              save_dir=save_dir
                              )
    
    
def model_init(cfg, classnames_list, base_model, device, param, vfeat_dim):
    model = PPTCONCH(cfg, classnames_list, base_model, device, param, vfeat_dim)
    
    ### Freeze all parameters except prompt_learner.ctx ###
    for name, parameters in model.named_parameters():
        parameters.requires_grad = False
        if name.startswith('prompt_learner.ctx') or name.startswith('mlp.') or name.startswith('prompt_') or name.startswith('mil.'):
            parameters.requires_grad = True   
            
            
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
        
        for step, (data, _, label, _) in progress_bar:
            patch_labels = wsi_selecter_CONCH.patch_selector(label_model=label_model,
                                                    label_type = label_type,
                                                    dataset_name=param['dataset_name'], 
                                                    vision_feats = data, 
                                                    wsi_label = label,
                                                    logits_thd = param['logits_thd'], 
                                                    text_embedding = selected_prompt_embedding,
                                                    device=device
                                                    )
            torch.cuda.empty_cache()
            data = data.to(device)
            label = label.to(device)
            patch_labels = torch.tensor(patch_labels).to(device)
                        
            with autocast(): 
                wsi_logits, patch_logits = model(data)
                patch_loss = PatchSSLoss(patch_logits, patch_labels, epoch=i, total_epoch=epoch, weights=param['loss_weight'], balance=param['balance'], vision_only = param['vision_only'], pseudo_loss = enable_pseudo)

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
                
                ## compute dice on train and val
                # train_dice = evaluation.compute_dice(model, train_wsi_loader, device=device)
                # val_dice = evaluation.compute_dice(model, val_wsi_loader, device=device)
                
        # print(f"Epoch {i} loss: {running_loss.item() / len(dataloader)}, val:{val_result.tolist()}, LR: {current_lr:.8f}")        
        patch_result = np.array([val_result_dict['patch_bacc'],val_result_dict['patch_wf1']])
        patch_rounded = np.around(patch_result, decimals=4).tolist()
        logging.info(f"Epoch {i} loss: {running_loss.item()*accumulation_steps / len(train_loader):.4f}, LR: {current_lr:.8f}, patch_test:{patch_rounded}")
        # torch.save(model, os.path.join(params.SAVE_DIR, f'epoch_latest.pt'))
    # timestamp = datetime.now().strftime("%m%d%H%M%S")
    # torch.save(model.prompt_learner.ctx, os.path.join(params.SAVE_DIR, f'pt{timestamp}.pt'))
    # print(f'best prompt saved as pt{timestamp}.pt')
    return model
