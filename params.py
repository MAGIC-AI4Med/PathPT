class PromptLearnerConfig():
    def __init__(self, n_ctx=32, input_size=256, init = None):
        self.model_backbone_name = "ViT-B/16"
        self.n_ctx = n_ctx
        if init == 'template':
            self.ctx_init="a histopathology image of a histopathology image of a histopathology image of a histopathology image of a histopathology image of a histopathology image of a histopathology image of a histopathology image of "
        elif init =='rand':
            self.ctx_init=None
    
        #self.input_size = [input_size]
        self.input_size = [input_size]
        # self.csc = False
        # self.class_token_position = 'end'
        self.token_embedding_size = 768
        self.patch_embedding_size = 1024

# CONCH_PATH = "/ailab/group/pjlab-medai/zhouxiao/pathology/model/conch/pytorch_model.bin"
# LOCAL_CKPT_PATH = "C:\\Users\\Administrator\\Code\\pathology\\model\\conch\\pytorch_model.bin"
# KEEP_PATH = '/ailab/group/pjlab-medai/zhouxiao/pathology/model/keep'
# TESTSET_DIVISION = "/ailab/group/pjlab-medai/zhouxiao/pathology/data/patch_level_data/patch_classification/testset_division.json"
        
conch_PATH = "/ailab/group/pjlab-medai/zhouxiao/pathology/model/conch/pytorch_model.bin"
MUSK_PATH = '/ailab/group/pjlab-medai/zhouxiao/pathology/model/MUSK'
LOCAL_CKPT_PATH = "C:\\Users\\Administrator\\Code\\pathology\\model\\conch\\pytorch_model.bin"
KEEP_PATH = '/ailab/group/pjlab-medai/zhouxiao/pathology/model/keep'
plip_PATH = '/ailab/group/pjlab-medai/zhouxiao/pathology/model/plip'
TESTSET_DIVISION = "/ailab/group/pjlab-medai/zhouxiao/pathology/data/patch_level_data/patch_classification/testset_division.json"
TRAINSET_DIVISION = "/ailab/group/pjlab-medai/zhouxiao/pathology/data/patch_level_data/patch_classification/trainset_division.json"
SAVE_DIR = "/ailab/group/pjlab-medai/zhouxiao/pathology/fewshot_results/"
DATASET_DIVISION = '/ailab/group/pjlab-medai/zhouxiao/pathology/data/tcga_WSI_data/dataset_division.json'    

subtype_params = {
    'sarc':{
        "source":'TCGA',
        'dataset_name':'SARC',
        'dataset_path':'/ailab/user/zhouxiao/WSI_proc_code/TransMIL-main/dataset_csvs/dataset_csv_1shot/TCGA/SARC/',
        'wsi_dir_path' : '/ailab/group/pjlab-medai/zhouxiao/pathology/data/tcga_WSI_data/SARC',
        "keep_feature_root" : "/ailab/group/pjlab-medai/zhouxiao/4090_backup/TCGA_WSI_feats/denoise_uni_doumls_hierarchy_label/SARC/h5_files",
        "plip_feature_root" : "/ailab/group/pjlab-medai/zhouxiao/4090_backup/TCGA_WSI_feats/plip/SARC/h5_files",
        "conch_feature_root" : "/ailab/group/pjlab-medai/zhouxiao/4090_backup/TCGA_WSI_feats/conch/SARC/h5_files",
        "train_dir_path" : None, #"/ailab/group/pjlab-medai/zhouxiao/pathology/features/keep/wsi_features/tcga/SARC/h5_with_label/fine",
        # "label_csv" : "/ailab/group/pjlab-medai/zhouxiao/pathology/features/keep/wsi_features/tcga/SARC/tcga_nsclc_test.csv",
        # "zero_shot_template_file" : "/ailab/group/pjlab-medai/zhouxiao/pathology/prompts/subtyping/tcga_nsclc_prompts_all.json",
        "batch_size": 1,  # 1 is the best
        "patch_num": None, # None is the best
        'epochs':20,
        'repeats':10,
        'shot':10,
        'learnable': 'token',  #'embedding', 'token'
        'use_aug': False,
        'vision_only': False,
        'vision_grad': True,
        'vision_mil':False,
        'prompt_select': True,
        'loss_weight': [1.0,0.5,0.1],
        'accumulation_steps': 1,
        'topn':10,
        'logits_thd':0.,
        'eval_type': 'wsi',  # 'patch', 'wsi', 'both'
        'init': 'template',  #'template', 'rand'
        'device':'cuda:3',
        'lr':1e-4,  # 3e-5 is the best
        'balance': True,  # True is the best
        'enable_pseudo': True,
        # 'use_manual_label': True,
        "log_name":'no_dice_1shot_plip/sarc_1e-4_new',
    },
    'thym':{
        "source":'TCGA',
        'dataset_name':'THYM',
        'dataset_path':'/ailab/user/zhouxiao/WSI_proc_code/TransMIL-main/dataset_csvs/dataset_csv_1shot/TCGA/THYM/',
        'wsi_dir_path' : '/ailab/group/pjlab-medai/zhouxiao/pathology/data/tcga_WSI_data/THYM',
        "keep_feature_root" : "/ailab/group/pjlab-medai/zhouxiao/4090_backup/TCGA_WSI_feats/denoise_uni_doumls_hierarchy_label/THYM/h5_files",
        "plip_feature_root" : "/ailab/group/pjlab-medai/zhouxiao/4090_backup/TCGA_WSI_feats/plip/THYM/h5_files",
        "conch_feature_root" : "/ailab/group/pjlab-medai/zhouxiao/4090_backup/TCGA_WSI_feats/conch/THYM/h5_files",
        "train_dir_path" : "/ailab/group/pjlab-medai/zhouxiao/4090_backup/TCGA_WSI_feats/denoise_uni_doumls_hierarchy_label/THYM/h5_with_label",
        # "label_csv" : "/ailab/group/pjlab-medai/zhouxiao/pathology/features/keep/wsi_features/tcga/SARC/tcga_nsclc_test.csv",
        # "zero_shot_template_file" : "/ailab/group/pjlab-medai/zhouxiao/pathology/prompts/subtyping/tcga_nsclc_prompts_all.json",
        "batch_size": 1,  # 1 is the best
        "patch_num": None, # None is the best
        'epochs':20,
        'repeats':10,
        'shot':10,
        'learnable': 'token',  #'embedding', 'token'
        'use_aug': False,
        'vision_only': False,
        'vision_grad': True,
        'vision_mil':False,
        'prompt_select': True,
        'loss_weight': [1.0,0.5,0.1],
        'accumulation_steps': 1,
        'topn':10,
        'logits_thd':0.,
        'eval_type': 'wsi',  # 'patch', 'wsi', 'both'
        'init': 'template',  #'template', 'rand'
        'device':'cuda:6',
        'lr':1e-4,  # 3e-5 is the best
        'balance': True,  # True is the best
        'enable_pseudo': True,
        # 'use_manual_label': True,
        "log_name":'no_dice_1shot_plip/thym_1e-4_new',
    },
    'ucs':{
        "source":'TCGA',
        'dataset_name':'UCS',
        'dataset_path':'/ailab/user/zhouxiao/WSI_proc_code/TransMIL-main/dataset_csvs/dataset_csv_1shot/TCGA/UCS/',
        'wsi_dir_path' : '/ailab/group/pjlab-medai/zhouxiao/pathology/data/tcga_WSI_data/UCS',
        "keep_feature_root" : "/ailab/group/pjlab-medai/zhouxiao/4090_backup/TCGA_WSI_feats/denoise_uni_doumls_hierarchy_label/UCS/h5_files",
        "plip_feature_root" : "/ailab/group/pjlab-medai/zhouxiao/4090_backup/TCGA_WSI_feats/plip/UCS/h5_files",
        "conch_feature_root" : "/ailab/group/pjlab-medai/zhouxiao/4090_backup/TCGA_WSI_feats/conch/UCS/h5_files",
        "train_dir_path" :"/ailab/group/pjlab-medai/zhouxiao/4090_backup/TCGA_WSI_feats/denoise_uni_doumls_hierarchy_label/UCS/h5_with_label",
        # "label_csv" : "/ailab/group/pjlab-medai/zhouxiao/pathology/features/keep/wsi_features/tcga/SARC/tcga_nsclc_test.csv",
        # "zero_shot_template_file" : "/ailab/group/pjlab-medai/zhouxiao/pathology/prompts/subtyping/tcga_nsclc_prompts_all.json",
        "batch_size": 1,  # 1 is the best
        "patch_num": None, # None is the best
        'epochs':20,
        'repeats':10,
        'shot':10,
        'learnable': 'token',  #'embedding', 'token'
        'use_aug': False,
        'vision_only': False,
        'vision_grad': True,
        'vision_mil':False,
        'prompt_select': True,
        'loss_weight': [1.0,0.5,0.1],
        'accumulation_steps': 1,
        'topn':10,
        'logits_thd':0.,
        'eval_type': 'wsi',  # 'patch', 'wsi', 'both'
        'init': 'template',  #'template', 'rand'
        'device':'cuda:0',
        'lr':1e-4,  # 3e-5 is the best
        'balance': True,  # True is the best
        'enable_pseudo': True,
        # 'use_manual_label': True,
        "log_name":'ucs_1e-4',
    },
}
