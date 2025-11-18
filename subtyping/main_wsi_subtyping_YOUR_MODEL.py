import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from WSI_dataset import WSI_Data
import random
import utils
import numpy as np
import evaluation
import params
import logging
from tqdm import tqdm
import json
from torch.cuda.amp import autocast, GradScaler
from loss import calpatch_loss, PatchSSLoss, balanced_ce_loss

# TODO: Import model-specific libraries here
# Examples:
# from transformers import CLIPModel, AutoModel, AutoTokenizer  # For CLIP-like models
# from transformers import XLMRobertaTokenizer                 # For MUSK-like models
# import timm                                                  # For MUSK
# import open_clip_CONCH as conch_clip                        # For CONCH
# import your_model_library as your_model                     # For custom models
# import your_model.utils as your_model_utils                 # For model-specific utilities

# TODO: Import your model classes
# from models.PathPT_model_YOUR_MODEL import OriginYOUR_MODEL, PPTYOUR_MODEL, CustomYOUR_MODEL

# TODO: Import your WSI selecter
# import wsi_selecters.wsi_selecter_YOUR_MODEL as wsi_selecter_YOUR_MODEL

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main_subtyping(dataset_name, cfg=params.PromptLearnerConfig(input_size=256), metric=evaluation.simple_dice_auc, given_param=None):
    """
    Main function for subtyping with YOUR_MODEL
    
    Args:
        dataset_name: Name of the dataset
        cfg: Configuration for prompt learning
        metric: Evaluation metric function
        save_name: Name for saving logs
        given_param: Optional custom parameters
    """
    if given_param is not None:
        param = given_param
    else:
        param = params.subtype_params[dataset_name]

    # TODO: Set path to your model
    # Examples:
    # model_path = params.YOUR_MODEL_PATH
    # model_path = params.PLIP_PATH
    # model_path = params.KEEP_PATH
    # model_path = params.CONCH_PATH
    # model_path = params.MUSK_PATH
    model_path = None  # TODO: Set your model path here
    
    with open(params.DATASET_DIVISION, 'r') as f:
        meta = json.load(f)[dataset_name.upper()]
    name2label = meta['name2label']
    subtype_classnames = sorted(name2label.keys(), key=lambda x: name2label[x])
    subtype_classnames = ['Normal'] + subtype_classnames
    print(dataset_name)
    print(subtype_classnames)
    
    zeroshot_prompt_lst, classnames_list = utils.load_prompts(dataset_name, subtype_classnames)
    
    multiple_trains_and_eval(
        cfg=cfg,
        ckpt_path=model_path,
        zeroshot_prompt_lst=zeroshot_prompt_lst,
        metric=metric,
        classnames_list=classnames_list,
        param=param,
        train_info=meta,
    )

def multiple_trains_and_eval(cfg, 
                             ckpt_path, 
                             zeroshot_prompt_lst, 
                             metric, 
                             classnames_list=None, 
                             param=None,
                             train_info=None): 
    """
    Run multiple training and evaluation rounds
    """
    repeats = param['repeats']
    device = param['device']
    
    # TODO: Load your base model and tokenizer
    # Examples:
    
    # For PLIP:
    # base_model = CLIPModel.from_pretrained(ckpt_path).to(device)
    # base_tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    
    # For KEEP:
    # base_model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True)
    # base_tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    # base_model.to(device)
    
    # For MUSK:
    # base_model = timm.create_model("musk_large_patch16_384")
    # musk_utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", base_model, 'model|module', '', local_dir=ckpt_path)
    # base_model.to(device=device, dtype=torch.float32)
    # base_tokenizer = XLMRobertaTokenizer("./musk/models/tokenizer.spm")
    
    # For CONCH:
    # base_model, _ = conch_clip.create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=ckpt_path)
    # base_model.to(device)
    # base_tokenizer = None  # CONCH uses its own tokenizer
    
    # TODO: Replace with your model loading code
    base_model = None  # TODO: Load your base model here
    base_tokenizer = None  # TODO: Load your tokenizer here (if needed)
    
    print(f"Device: {device}")

    for i in range(repeats):
        
        ## Create data loaders
        # TODO: Update feature_path to match your model's feature directory
        # Examples:
        # feature_path = param['plip_feature_root']     # For PLIP
        # feature_path = param['keep_feature_root']     # For KEEP
        # feature_path = param['musk_feature_root']     # For MUSK
        # feature_path = param['conch_feature_root']    # For CONCH
        feature_path = param['your_model_feature_root']  # TODO: Set your feature path key
        
        train_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                feature_path=feature_path,
                                fold=i,
                                patch_num=param['patch_num'],
                                state='train')
        train_loader = DataLoader(train_dataset, batch_size=param['batch_size'], num_workers=8, shuffle=True)
        
        train_wsi_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                    feature_path=feature_path,
                                    fold=i,
                                    state='train')
        train_wsi_loader = DataLoader(train_wsi_dataset, batch_size=1, num_workers=8, shuffle=True)
        
        val_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                              feature_path=feature_path,
                              fold=i,
                              state='val',
                              train_info=train_info)
        val_wsi_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, shuffle=True)
        
        test_wsi_dataset = WSI_Data(dataset_path=param['dataset_path'], 
                                   feature_path=feature_path,
                                   fold=i,
                                   state='test')
        test_wsi_loader = DataLoader(test_wsi_dataset, batch_size=1, num_workers=8, shuffle=False)
        
        logging.info("======== ↓↓ Experiment No.{} ↓↓ ========".format(i))
        
        # Prompt selection phase
        selected_prompt_embedding = None
        if param['prompt_select']:
            print('Manual prompt selecting...')
            
            # TODO: Use your model's prompt selector
            # Examples:
            # selected_prompt_embedding = wsi_selecter_PLIP.wsi_prompt_selector(base_model, base_tokenizer, train_wsi_loader, zeroshot_prompt_lst, device=device)
            # selected_prompt_embedding = wsi_selecter_KEEP.wsi_prompt_selector(train_wsi_loader, zeroshot_prompt_lst, device=device)
            # selected_prompt_embedding = wsi_selecter_MUSK.wsi_prompt_selector(base_model, base_tokenizer, train_wsi_loader, zeroshot_prompt_lst, device=device)
            # selected_prompt_embedding = wsi_selecter_CONCH.wsi_prompt_selector(base_model, train_wsi_loader, zeroshot_prompt_lst, device=device)
            
            # TODO: Replace with your model's prompt selector
            selected_prompt_embedding = None  # TODO: Implement prompt selection here
            
            # Evaluate prompt selection on training set
            train_results_dict = metric(base_model, train_wsi_loader, device, batch_classifier=None, 
                                      model_type='prompting', prompt_embedding=selected_prompt_embedding, save_logits=True)
            prompting_train_result = np.array([train_results_dict['patch_bacc'], train_results_dict['patch_wf1']])
            logging.info(f"Manual prompting on trainset{prompting_train_result}")
            
            # Evaluate prompt selection on test set
            test_results_dict = metric(base_model, test_wsi_loader, device, batch_classifier=None, 
                                     model_type='prompting', prompt_embedding=selected_prompt_embedding)
            prompting_test_result = np.array([test_results_dict['patch_bacc'], test_results_dict['patch_wf1']])
            logging.info(f"Manual prompting on testset{prompting_test_result}")
        
        # Zero-shot evaluation
        random.seed(i)
        zeroshot_prompt = []
        for cls_prompt in zeroshot_prompt_lst:
            index = random.randint(0, len(cls_prompt) - 1)
            zeroshot_prompt.append(cls_prompt[index])
        
        # TODO: Create zero-shot model with your implementation
        # Examples:
        # zero_shot_model = OriginPLIP(zeroshot_prompt, base_model, base_tokenizer, device)
        # zero_shot_model = OriginKEEP(zeroshot_prompt, base_model, base_tokenizer, device)
        # zero_shot_model = OriginMUSK(zeroshot_prompt, base_model, base_tokenizer, device)
        # zero_shot_model = OriginCONCH(zeroshot_prompt, base_model, device)
        
        # TODO: Replace with your zero-shot model
        zero_shot_model = None  # TODO: Create your zero-shot model here
        
        zero_shot_results_dict = metric(zero_shot_model, test_wsi_loader, device, batch_classifier=None, model_type='zeroshot', save_dir=f'./fewshot_results/{os.path.basename(param["log_name"])[:-4]}/zeroshot')
        zero_shot_result = np.array([zero_shot_results_dict['patch_bacc'], zero_shot_results_dict['patch_wf1']])
        logging.info(f"Zeroshot result{zero_shot_result}")

        logging.info('>>>>>>>>> Shots: ' + str(param['shot']) + ' <<<<<<<<<')
        
        # Training phase
        print('Prompt learning...')
        torch.cuda.empty_cache()
        train_anno(cfg, classnames_list,
                   base_model,
                   base_tokenizer,
                   base_model,  # label_model (can be different from base_model)
                   train_loader, 
                   train_wsi_loader,
                   val_wsi_loader,
                   test_wsi_loader, 
                   metric, 
                   param=param,
                   selected_prompt_embedding=selected_prompt_embedding, 
                   device=device,
                   save_dir = f'./fewshot_results/{os.path.basename(param["log_name"])[:-4]}/fold{i}'
                   )

def train_anno(cfg, 
               classnames_list,
               base_model,
               base_tokenizer,
               label_model,
               train_loader, 
               train_wsi_loader,
               val_wsi_loader,
               test_loader, 
               metric, 
               param,
               selected_prompt_embedding=None, 
               device="cuda:0",
               save_dir=None
               ):
    """
    Training and annotation function
    """
    # TODO: Set vfeat_dim based on your model's feature dimension
    # Examples:
    # vfeat_dim = 512   # For PLIP, CONCH
    # vfeat_dim = 768   # For KEEP
    # vfeat_dim = 1024  # For MUSK
    vfeat_dim = 512  # TODO: Set your model's feature dimension
    
    init_model, optimizer, scheduler = model_init(cfg, classnames_list, base_model, base_tokenizer, device, param, vfeat_dim)
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
                              selected_prompt_embedding=selected_prompt_embedding,
                              enable_pseudo=param['enable_pseudo'],
                              save_dir=save_dir
                              )

def model_init(cfg, classnames_list, base_model, base_tokenizer, device, param, vfeat_dim):
    """
    Initialize the model, optimizer, and scheduler
    """
    # TODO: Initialize your PathPT model
    # Examples:
    # model = PPTPLIP(cfg, classnames_list, base_model, base_tokenizer, device, param, vfeat_dim)
    # model = PPTKEEP(cfg, classnames_list, base_model, base_tokenizer, device, param, vfeat_dim)
    # model = PPTMUSK(cfg, classnames_list, base_model, base_tokenizer, device, param, vfeat_dim)
    # model = PPTCONCH(cfg, classnames_list, base_model, device, param, vfeat_dim)  # Note: CONCH doesn't need tokenizer
    
    # TODO: Replace with your model initialization
    model = None  # TODO: Initialize your PathPT model here
    
    # Freeze all parameters except prompt_learner.ctx and other trainable components
    for name, parameters in model.named_parameters():
        parameters.requires_grad = False
        if (name.startswith('prompt_learner.ctx') or 
            name.startswith('mlp.') or 
            name.startswith('prompt_') or 
            name.startswith('mil.')):
            parameters.requires_grad = True   
    
    # Set optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=param['lr'])
    warm_up_iter = int(param['epochs'] * 0.1)
    lambda0 = lambda cur_iter: cur_iter / warm_up_iter if cur_iter < warm_up_iter else \
            0.5 * (1.0 + np.cos(np.pi * (cur_iter - warm_up_iter) / (param['epochs'] - warm_up_iter)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    
    return model, optimizer, scheduler

def train_one_step(epoch, model, device, param, train_loader, train_wsi_loader, val_wsi_loader, 
                   label_model, label_type, optimizer, scheduler, metric, test_loader, 
                   selected_prompt_embedding=None, enable_pseudo=True, save_dir=None):
    """
    Training loop for one step
    """
    for i in range(epoch):
        # Training phase
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
            
            # Generate patch labels if not in vision-only mode
            if not param['vision_only']:
                # TODO: Use your model's patch selector
                # Examples:
                # patch_labels = wsi_selecter_PLIP.patch_selector(...)
                # patch_labels = wsi_selecter_KEEP.patch_selector(...)
                # patch_labels = wsi_selecter_MUSK.patch_selector(...)
                # patch_labels = wsi_selecter_CONCH.patch_selector(...)
                
                # TODO: Replace with your patch selector
                patch_labels = None  # TODO: Implement patch selection here
                
                patch_labels = torch.tensor(patch_labels).to(device)
            
            torch.cuda.empty_cache()
            
            # Prepare data
            data = data.to(device)
            if hasattr(data, 'float'):  # For MUSK, data needs to be float
                data = data.float()
            label = label.to(device)
            
            # Forward pass with mixed precision
            with autocast(): 
                wsi_logits, patch_logits = model(data)
                
                # Calculate loss based on mode
                if param['vision_only']:
                    loss = F.cross_entropy(wsi_logits, label)
                else:
                    patch_loss = PatchSSLoss(patch_logits, patch_labels, epoch=i, total_epoch=epoch, 
                                           weights=param['loss_weight'], balance=param['balance'], 
                                           vision_only=param['vision_only'], pseudo_loss=enable_pseudo)
                    loss = patch_loss
                
                # Handle different loss formats
                if isinstance(loss, dict):
                    loss = loss['loss'] / accumulation_steps
                else:
                    loss = loss / accumulation_steps
                    
            torch.cuda.empty_cache()
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                running_loss += loss.item()
                torch.cuda.empty_cache()
        
        scheduler.step()
        
        # Evaluation phase
        model.eval()
        current_lr = scheduler.get_last_lr()[0]
        with torch.no_grad():
            if isinstance(metric, dict):
                train_patch_result_dict = metric['patch'](model, test_loader['patch'], device=device, batch_classifier=None)
                val_result_dict = metric['wsi'](model, test_loader['wsi'], device=device, batch_classifier=None, save_dir=save_dir)
            else:
                val_result_dict = metric(model, test_loader, device=device, batch_classifier=None, 
                                       model_type='fewshot-mil', vision_only=param['vision_only'], save_dir=save_dir)
            
            # TODO: Add model-specific evaluation metrics if needed
            # For example, CONCH computes dice scores:
            # if hasattr(evaluation, 'compute_dice'):
            #     train_dice = evaluation.compute_dice(model, train_wsi_loader, device=device)
            #     val_dice = evaluation.compute_dice(model, val_wsi_loader, device=device)
        
        # Logging
        if param['vision_only']:
            wsi_result = np.array([val_result_dict['wsi_bacc'], val_result_dict['wsi_wf1']])
            wsi_rounded = np.around(wsi_result, decimals=4).tolist()
            logging.info(f"Epoch {i} loss: {running_loss.item()*accumulation_steps / len(train_loader):.4f}, "
                        f"LR: {current_lr:.8f}, wsi:{wsi_rounded}")
        else:
            if 'wsi_bacc' in val_result_dict:
                # Multi-level evaluation
                patch_result = np.array([val_result_dict['patch_bacc'], val_result_dict['patch_wf1']])
                patch_rounded = np.around(patch_result, decimals=4).tolist()
                wsi_result = np.array([val_result_dict['wsi_bacc'], val_result_dict['wsi_wf1']])
                wsi_rounded = np.around(wsi_result, decimals=4).tolist()
                ensemble_result = np.array([val_result_dict['ensemble_bacc'], val_result_dict['ensemble_wf1']])
                ensemble_rounded = np.around(ensemble_result, decimals=4).tolist()
                logging.info(f"Epoch {i} loss: {running_loss.item()*accumulation_steps / len(train_loader):.4f}, "
                           f"LR: {current_lr:.8f}, patch:{patch_rounded}, wsi:{wsi_rounded}, ensemble:{ensemble_rounded}")
            else:
                # Patch-level evaluation only
                patch_result = np.array([val_result_dict['patch_bacc'], val_result_dict['patch_wf1']])
                patch_rounded = np.around(patch_result, decimals=4).tolist()
                logging.info(f"Epoch {i} loss: {running_loss.item()*accumulation_steps / len(train_loader):.4f}, "
                           f"LR: {current_lr:.8f}, patch_test:{patch_rounded}")
                
                # TODO: Add model-specific logging if needed
                # For example, for CONCH with dice scores:
                # logging.info(f"..., train_dice:{train_dice:.4f}, val_dice:{val_dice:.4f}")
    
    return model