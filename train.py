##### checklist before you start #####
# 1. download your base model in the base_model folder
# 2. set your dataset and extract features (frozen) in the features folder
# 3. divide your dataset into train/test sets in dataset_division.json
# 4. make multifold division csv files in the multifold folder
# 5. modify params.py and this file
# 6. run this file
##### -------------------------- #####

import os
import sys
import logging
import torch
import random
import numpy as np
import params
from subtyping.main_wsi_subtyping_KEEP import main_subtyping # change your base model here
import evaluation
from pathlib import Path

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

proc_tumor = 'ucs' # your few-shot dataset

### set seed ###
SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if __name__ == "__main__":
    
    log_name = params.subtype_params[proc_tumor]['log_name']
    eval_type = params.subtype_params[proc_tumor]['eval_type']
    if eval_type == 'patch':
        eval_metric = evaluation.simple_dice_auc  
    elif eval_type == 'wsi':
        eval_metric = evaluation.sub_bacc_wf1
    elif eval_type == 'both':
        eval_metric = dict()
        eval_metric['patch'] = evaluation.simple_dice_auc
        eval_metric['wsi'] = evaluation.sub_bacc_wf1

    save_path = './logs/' + log_name + ".log"
    log_file = Path(save_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(save_path),
            logging.StreamHandler(sys.stdout)
        ],
    )
    logging.info('******************** new start *******************')
    
    main_subtyping(proc_tumor, params.PromptLearnerConfig(n_ctx=32, init = params.subtype_params[proc_tumor]['init']),metric=eval_metric, save_name = log_name)
