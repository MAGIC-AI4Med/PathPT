import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from PathPL_model_CONCH import OriginCONCH, PPLCONCH
import torch.nn.functional as F
from WSI_dataset import WSI_Data
import random
import utils
import numpy as np
import evaluation
import params
import logging
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import wsi_selecter_CONCH
import json
from torch.cuda.amp import autocast, GradScaler
from loss import calpatch_loss, PatchSSLoss
import open_clip_CONCH as conch_clip


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main_subtyping(dataset_name, cfg = params.PromptLearnerConfig(input_size=256),  metric = evaluation.simple_dice_auc, save_name = None, given_param=None):
    if given_param is not None:
        param = given_param
    else:
    
        param = params.subtype_params[dataset_name]
    
    model_path = params.conch_PATH
    keep_path = params.KEEP_PATH
    with open(params.DATASET_DIVISION, 'r') as f:
        meta = json.load(f)[dataset_name.upper()]
    name2label = meta['name2label']
    subtype_classnames = sorted(name2label.keys(), key=lambda x: name2label[x])
    # subtype_classnames = gen_loader.get_classnames()
    subtype_classnames = ['Normal'] + subtype_classnames
    print(subtype_classnames)
    
    # zero_shot_template_file = param['zero_shot_template_file']
    # with open(zero_shot_template_file, 'r') as f:
    #     zeroshot_prompt_file = json.load(f)
    
    # zeroshot_prompt_lst = []
    # for k,v in zeroshot_prompt_file.items():
    #     prompt_result = [
    #                 v["templates"].replace("CLASSNAME", v["classnames"][class_key])
    #                 for class_key in subtype_classnames
    #             ]
    #     zeroshot_prompt_lst.append(prompt_result)

    # csv_classnames = ['Normal', 'IDC', 'ILC'] #load_prompts_from_template识别不了首字母缩写
    zeroshot_prompt_lst,classnames_list = utils.load_prompts(dataset_name, subtype_classnames)
    
    multiple_trains_and_eval(
        cfg=cfg,
        ckpt_path=model_path,
        keep_path=keep_path,
        zeroshot_prompt_lst=zeroshot_prompt_lst,
        metric=metric,
        classnames_list=classnames_list,
        param = param,
        train_info = meta,
    )    


def multiple_trains_and_eval(cfg, 
                             ckpt_path, 
                             keep_path,
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
    
    keep_model = AutoModel.from_pretrained(keep_path, trust_remote_code=True)
    # keep_tokenizer = AutoTokenizer.from_pretrained(keep_path, trust_remote_code=True)
    keep_model.to(device)
    
    base_model, _ = conch_clip.create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=ckpt_path)
    base_model.to(device)
    
    # conch_model.to(device)
    zeroshot_results = []
    fewshot_results = []
    print(device)

    for i in range(repeats):
        
        ## data loader
        train_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                feature_path = param['conch_feature_root'],
                                train_path = param["train_dir_path"],
                                fold = i,
                                patch_num=param['patch_num'],
                                state='train',
                                val_kwargs=val_kwargs,)
        train_loader = DataLoader(train_dataset, batch_size=param['batch_size'], num_workers=8, shuffle=True)
        
        train_wsi_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                feature_path = param['conch_feature_root'],
                                train_path = param["train_dir_path"],
                                fold = i,
                                state='train',
                                val_kwargs=val_kwargs,)
        train_wsi_loader = DataLoader(train_wsi_dataset, batch_size=1, num_workers=8, shuffle=True)
        
        
        val_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                        feature_path = param['conch_feature_root'],
                        train_path = param["train_dir_path"],
                        fold = i,
                        # patch_num=param['patch_num'],
                        state='val',
                        train_info = train_info,
                        val_kwargs=val_kwargs)
        val_wsi_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, shuffle=True)
        ## for keep label
        # label_wsi_dataset = WSI_Data(dataset_path=param['dataset_path'], 
        #                         feature_path = param['keep_feature_root'],
        #                         fold = i,
        #                         patch_num=param['patch_num'],
        #                         state='train')
        # label_wsi_loader = DataLoader(label_wsi_dataset, batch_size=1, num_workers=8, shuffle=True)
        
        test_wsi_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                feature_path = param['conch_feature_root'],
                                fold = i,
                                state='test',
                                val_kwargs=val_kwargs)
        # test_wsi_dataset = WSI_Data(dataset_path=param['dataset_path'],
        test_wsi_loader = DataLoader(test_wsi_dataset, batch_size=1, num_workers=8, shuffle=False)
        
        
        logging.info("======== ↓↓ Experiment No.{} ↓↓ ========".format(i))
        
        # unique_classnames_lst = utils.unique_classnames(classnames_lst)
        # print(f'Ensemble classnames: {classnames_lst}')
        # origin_model = CustomCONCH(cfg, unique_classnames_lst, conch_model, device, mode)
        # origin_model = CustomKEEP(cfg, classnames_list, keep_model, keep_tokenizer, device, param, vfeat_dim=768)
        
        # test_loader = dataloader_generator.gen_testloader(metric)
        # train_feat_paths, val_feat_paths = dataloader_generator.split_train_val(seed = i)
        # train_wsi_loader = dataloader_generator.gen_wsiloader(train_feat_paths)
        # val_loader = dataloader_generator.gen_wsiloader(val_feat_paths)
        
        selected_prompt_embedding = None
        if param['prompt_select']:
            # train_dataset_list = dataloader_generator.trainset.get_example_paths()
            print('step 1. Selecting prompts...')
            selected_prompt_embedding = wsi_selecter_CONCH.wsi_prompt_selector(base_model, train_wsi_loader, zeroshot_prompt_lst, device=device)
            # selected_prompt_embedding = wsi_selecter.wsi_prompt_selector(train_wsi_loader, zeroshot_prompt_lst, device=device)
            
            train_results_dict = metric(base_model, train_wsi_loader, device, batch_classifier=None, model_type = 'prompting', prompt_embedding = selected_prompt_embedding, save_logits = True)
            prompting_train_result = np.array([train_results_dict['patch_bacc'],train_results_dict['patch_wf1']])
            logging.info(f"Prompting train result{prompting_train_result}")
            
            test_results_dict = metric(base_model, test_wsi_loader, device, batch_classifier=None, model_type = 'prompting', prompt_embedding = selected_prompt_embedding)
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

        zero_shot_model = OriginCONCH(zeroshot_prompt, base_model, device)
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
        
        ### train & validadte the model ###
        print('step 3. Prompt learning...')
        torch.cuda.empty_cache()
        train_anno(cfg,classnames_list,
                   base_model,
                    keep_model, 
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
            ):
    
    ## first step
    init_model, optimizer, scheduler = model_init(cfg, classnames_list, base_model, device, param, vfeat_dim = 512)

    print('******* Setp 1 ********')
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
                              enable_pseudo = param['enable_pseudo'])
    
    
def model_init(cfg, classnames_list, base_model, device, param, vfeat_dim):
    model = PPLCONCH(cfg, classnames_list, base_model, device, param, vfeat_dim)
    
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
    
    
def train_one_step(epoch, model, device, param, train_loader, train_wsi_loader, val_wsi_loader, label_model, label_type, optimizer, scheduler, metric, test_loader, selected_prompt_embedding = None, enable_pseudo = True):
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
        
        for step, (data, _, label, _) in progress_bar:
            # change classnames
            # model.change_classnames()
            
            # print('step 2. Patch labeling...')
            patch_labels = wsi_selecter_CONCH.patch_selector(label_model=label_model,
                                                    label_type = label_type,
                                                    dataset_name=param['dataset_name'], 
                                                    vision_feats = data, 
                                                    wsi_label = label,
                                                    logits_thd = param['logits_thd'], 
                                                    text_embedding = selected_prompt_embedding,
                                                    device=device
                                                    )
            
            # vision_feats = []
            # for ii in range(len(idx)):
            #     slide_id = label_wsi_loader.dataset.data[idx[ii].item()]
            #     h5file_path = Path(label_wsi_loader.dataset.feature_dir) / f'{slide_id}.h5'
        
            #     with h5py.File(h5file_path, 'r') as f:
            #         features = f['features'][:]
                
            #     if label_wsi_loader.dataset.patch_num is not None:
            #         vision_feats.append(np.take(features, position_indices[ii], axis=0))
            #     else:
            #         vision_feats.append(features)
                
            # vision_feats = torch.stack([torch.from_numpy(item) for item in vision_feats])
            # if len(vision_feats.shape) < 3:
            #     vision_feats = vision_feats.unsqueeze(0)
                   
            
            # patch_labels = wsi_selecter.patch_selector(label_model=label_model,
            #                                 label_type = label_type,
            #                                 dataset_name=param['dataset_name'], 
            #                                 vision_feats = vision_feats, 
            #                                 wsi_label = label,
            #                                 logits_thd = param['logits_thd'], 
            #                                 text_embedding = selected_prompt_embedding,
            #                                 device=device
            #                                 )
            
            torch.cuda.empty_cache()
            # 统计每种标签的数量
            # view_labels = list(patch_labels.view(-1).numpy())
            # label_counts = Counter(view_labels)
            # for patch_label, count in label_counts.items():
            #     print(f"label {patch_label}: {count} instances")
            
            data = data.to(device)
            
            ## sample N patches 
            # sample_len = min(62500, data.shape[1])
            # start_id = random.randint(0,data.shape[1]-sample_len)
            # data = data[:,start_id:(start_id+sample_len),:]
            # patch_labels = patch_labels[start_id:(start_id+sample_len)]
            
            label = label.to(device)
            patch_labels = torch.tensor(patch_labels).to(device)
                        
            with autocast(): 
                wsi_logits, patch_logits = model(data)
                # patch_loss = balanced_ce_loss(patch_logits, patch_labels)
                patch_loss = PatchSSLoss(patch_logits, patch_labels, epoch=i, total_epoch=epoch, weights=param['loss_weight'], balance=param['balance'], vision_only = param['vision_only'], pseudo_loss = enable_pseudo)

                if param['vision_mil']:
                    wsi_loss = F.cross_entropy(wsi_logits, label)
                    loss = wsi_loss + patch_loss['loss']
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
            
        scheduler.step()
        ### evaluate the model ###
        model.eval()
        current_lr = scheduler.get_last_lr()[0]  # 获取当前学习率
        with torch.no_grad():
            # if val_loader and metric is evaluation.sub_bacc_wf1:
            #     select_result = np.array(metric(model, val_loader, device=device, batch_classifier=None))
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
        if param['vision_mil']:
            patch_result = np.array([val_result_dict['patch_bacc'],val_result_dict['patch_wf1']])
            patch_rounded = np.around(patch_result, decimals=4).tolist()
            wsi_result = np.array([val_result_dict['wsi_bacc'],val_result_dict['wsi_wf1']])
            wsi_rounded = np.around(wsi_result, decimals=4).tolist()
            ensemble_result = np.array([val_result_dict['ensemble_bacc'],val_result_dict['ensemble_wf1']])
            ensemble_rounded = np.around(ensemble_result, decimals=4).tolist()
            logging.info(f"Epoch {i} loss: {running_loss.item()*accumulation_steps / len(train_loader):.4f}, LR: {current_lr:.8f}, patch:{patch_rounded}, wsi:{wsi_rounded}, ensemble:{ensemble_rounded}")
        else:
            patch_result = np.array([val_result_dict['patch_bacc'],val_result_dict['patch_wf1']])
            patch_rounded = np.around(patch_result, decimals=4).tolist()
            logging.info(f"Epoch {i} loss: {running_loss.item()*accumulation_steps / len(train_loader):.4f}, LR: {current_lr:.8f}, patch_test:{patch_rounded}, train_dice:{train_dice:.4f}, val_dice:{val_dice:.4f}")
        
        
        # torch.save(model, os.path.join(params.SAVE_DIR, f'epoch_latest.pt'))
    # timestamp = datetime.now().strftime("%m%d%H%M%S")
    # torch.save(model.prompt_learner.ctx, os.path.join(params.SAVE_DIR, f'pt{timestamp}.pt'))
    # print(f'best prompt saved as pt{timestamp}.pt')
    return model
