import os
import sys
import logging
import torch
import random
import numpy as np
import params
# from main_wsi_subtyping_CONCH import main_subtyping
# from main_wsi_subtyping import main_subtyping
from main.main_wsi_subtyping_MUSK import main_subtyping
import evaluation
from pathlib import Path

proc_tumor = 'ebrains'

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
    log_file.parent.mkdir(parents=True, exist_ok=True)  # 创建所有父目录
    log_file.touch(exist_ok=True)  

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(save_path),             # 输出到文件
            logging.StreamHandler(sys.stdout)              # 输出到终端
        ],
        # filename=log_name + ".log",  # 日志文件路径
        # filemode="a"  # 文件模式：'w' 覆盖，'a' 追加
    )
    logging.info('******************** new start *******************')
    
    main_subtyping(proc_tumor, params.PromptLearnerConfig(n_ctx=32, init = params.subtype_params[proc_tumor]['init']),metric=eval_metric, save_name = log_name)
