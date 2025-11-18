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
from subtyping.main_wsi_subtyping_KEEP import main_subtyping as main_KEEP
from subtyping.main_wsi_subtyping_CONCH import main_subtyping as main_CONCH
from subtyping.main_wsi_subtyping_MUSK import main_subtyping as main_MUSK
from subtyping.main_wsi_subtyping_PLIP import main_subtyping as main_PLIP
from subtyping.main_wsi_subtyping_YOUR_MODEL import main_subtyping as main_YOUR_MODEL # use your custom model here
import evaluation
from pathlib import Path
import argparse
from datetime import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

### set seed ###
SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fewshot experiment params')
    parser.add_argument('--model', default='KEEP', type=str, help='model name: KEEP, CONCH, MUSK, PLIP, or your custom model')
    parser.add_argument('--dataset', default='ucs', type=str, help='dataset name (lower case)')
    parser.add_argument('--shot', default=10, type=int, help='number of shots, 1, 5, or 10')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--init', default='template', type=str, help='init method for prompt learning, template or rand')
    parser.add_argument('--device', default='cuda:0', type=str, help='gpu device')
    


    args = parser.parse_args()
    model = args.model
    proc_tumor = args.dataset
    shot = args.shot
    epochs = args.epochs
    lr = args.lr
    init = args.init
    device = args.device

    given_param = params.subtype_params[proc_tumor]
    given_param['shot'] = shot
    given_param['epochs'] = epochs
    given_param['lr'] = lr
    given_param['init'] = init
    given_param['device'] = device
    given_param['dataset_path'] = os.path.join(params.MULTIFOLD_DIV_DIR, f'dataset_csv_{shot}shot', given_param['source'], proc_tumor.upper()) + '/'

    if model == 'MUSK':
        given_param['patch_num'] = 100000 # In case of OOM

    log_name = f'pathpt_{model}_{shot}shot_{proc_tumor}'
    eval_type = params.subtype_params[proc_tumor]['eval_type']
    if eval_type == 'patch':
        eval_metric = evaluation.simple_dice_auc  
    elif eval_type == 'wsi':
        eval_metric = evaluation.sub_bacc_wf1
    elif eval_type == 'both':
        eval_metric = dict()
        eval_metric['patch'] = evaluation.simple_dice_auc
        eval_metric['wsi'] = evaluation.sub_bacc_wf1

    current_time = datetime.now().strftime('%b%d-%H-%M-%S-%f')[:-3]
    save_path = './logs/' + log_name + '_' + current_time + ".log"
    log_file = Path(save_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)

    given_param['log_name'] = save_path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(save_path),
            logging.StreamHandler(sys.stdout)
        ],
    )
    logging.info('******************** new start *******************')

    main_func = eval(f'main_{model}')
    main_func(proc_tumor, params.PromptLearnerConfig(n_ctx=32, init = params.subtype_params[proc_tumor]['init']),metric=eval_metric, given_param=given_param)
