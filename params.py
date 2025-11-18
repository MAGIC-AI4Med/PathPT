class PromptLearnerConfig():
    def __init__(self, n_ctx=32, input_size=256, init = 'template'):
        self.model_backbone_name = "ViT-B/16"
        self.n_ctx = n_ctx
        if init == 'template':
            # a template to init learnable prompt
            self.ctx_init="a histopathology image of a histopathology image of a histopathology image of a histopathology image of a histopathology image of a histopathology image of a histopathology image of a histopathology image of "
        elif init =='rand':
            self.ctx_init=None

        self.input_size = [input_size]
        self.token_embedding_size = 768
        self.patch_embedding_size = 1024

CONCH_PATH = "./base_models/conch/pytorch_model.bin" # YOUR PATH TO CONCH MODEL
MUSK_PATH = './base_models/musk' # YOUR PATH TO MUSK MODEL
KEEP_PATH = './base_models/keep' # YOUR PATH TO KEEP MODEL
PLIP_PATH = './base_models/plip' # YOUR PATH TO PLIP MODEL
MULTIFOLD_DIV_DIR = './multifold' # where you place your multifold division csvs
SAVE_DIR = "./fewshot_results/"

# dataset division, classnames & labels
# check this file before you start!!!
DATASET_DIVISION = './multifold/dataset_division.json'

subtype_params = {
    # TCGA-UCS as an example, you can use your customized dataset
    'ucs':{
        "source":'TCGA',
        'dataset_name':'UCS',
        "keep_feature_root" : "./features/TCGA_WSI_feats/keep/UCS/h5_files", # h5 format
        "plip_feature_root" : "./features/TCGA_WSI_feats/plip/UCS/h5_files", # h5 format
        "conch_feature_root" : "./features/TCGA_WSI_feats/conch/UCS/h5_files", # h5 format
        # musk is excluded because it was pretrained on TCGA
        "batch_size": 1,
        "patch_num": None, # patch sampled from a WSI, 'None' means all of the patches are sampled
        'repeats':10, # 10 fold repeats
        'learnable': 'token', # learnable token embedding in prompt learning
        'use_aug': False, # feature augmentation
        'vision_only': False, # vision only linear probing
        'vision_grad': True, # spatial-awareness module
        'prompt_select': True, # manual prompt selection
        'loss_weight': [1.0,0.5,0.1],
        'accumulation_steps': 1,
        'topn':10,
        'logits_thd':0.,
        'eval_type': 'wsi',
        'balance': True,
        'enable_pseudo': True,
    }

}
