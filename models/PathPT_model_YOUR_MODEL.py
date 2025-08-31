import torch
import torch.nn as nn
import random
from model_utils import MultiKernelConv1DTrans, AttentionAgg, ConvAttentionAgg, ConvTransAttentionAgg

# TODO: Import model-specific libraries here
# Examples:
# from transformers import AutoModel, AutoTokenizer, CLIPModel  # For CLIP-like models
# from transformers import XLMRobertaTokenizer                 # For MUSK-like models
# import open_clip_CONCH.custom_tokenizer as conch_tokenizer  # For CONCH-like models
# import your_model_library as your_model                     # For custom models

def prompt_padding(ctx, length, mode='repeat'):
    """Pad prompt context to specified length"""
    if mode == 'repeat':
        # Split ctx into word list by spaces
        words = ctx.split()
        # Calculate the number of repetitions needed
        repeat_times = (length + len(words) - 1) // len(words)
        # Repeat the word list
        padded_words = (words * repeat_times)[:length]
        # Rejoin the word list into a string
        padded_ctx = ' '.join(padded_words)
    else:
        raise ValueError("Unsupported mode. Currently only 'repeat' mode is supported.")
    
    return padded_ctx

class PromptLearnerYOUR_MODEL(nn.Module):
    """Prompt learner for YOUR_MODEL"""
    def __init__(self, cfg, classnames_lst, model, tokenizer, device):
        super().__init__()
        self.device = device
        n_cls = len(classnames_lst)
        n_ctx = cfg.n_ctx
        ctx_init = cfg.ctx_init
        ctx_dim = cfg.token_embedding_size
        model = model.to(self.device)

        print("Initializing a generic context")
        if ctx_init:
            # Use given words to initialize context vectors
            ctx_init = prompt_padding(ctx_init, n_ctx)
            
            # TODO: Implement tokenization for your model
            # Examples:
            
            # For CLIP-like models (KEEP, PLIP):
            # prompt = tokenizer([ctx_init]*n_cls, max_length=256, padding='max_length', 
            #                   truncation=True, return_tensors='pt')
            # with torch.no_grad():
            #     embedding = model.text.get_input_embeddings()(prompt['input_ids'].to(device))  # KEEP
            #     # OR for PLIP:
            #     # config = model.text_model.config
            #     # embed_token = nn.Embedding(config.vocab_size, config.hidden_size).to(device)
            #     # embedding = embed_token(prompt['input_ids'].to(device))
            
            # For MUSK:
            # prompt, prompt_pad = musk_utils.xlm_tokenizer(ctx_init, tokenizer, max_len=100)
            # prompt = torch.cat([torch.tensor(prompt).unsqueeze(0) for i in range(n_cls)])
            # with torch.no_grad():
            #     embedding = model.beit3.text_embed(prompt.to(device))
            
            # For CONCH:
            # prompt = conch_tokenizer.tokenize(tokenizer, [ctx_init]*n_cls)
            # with torch.no_grad():
            #     _, embedding = model._encode_text(prompt.to(device))
            
            # TODO: Replace with your model's text embedding method
            prompt = None  # Implement tokenization here
            with torch.no_grad():
                embedding = None  # Implement text embedding here
            
            ctx_vectors = embedding[:, 1 : 1 + n_ctx, :]  # [CLS] is in the first place
            prompt_prefix = ctx_init

        else:
            # Random initialization
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim).to(device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # Turn it into a torch parameter to be optimized
        embeddings = []  # Contains n_cls elements, each element is the prompt embedding collection
        tokenized_prompts_lst = []
        
        # TODO: Add model-specific variables for padding masks if needed (like MUSK)
        # tokenized_prompts_pad_lst = []  # For MUSK-like models
        
        print(f'classnames: {classnames_lst}')
        for classnames in classnames_lst:
            cls_prompts = [" ".join(["X"] * n_ctx) + " " + name + "." for name in classnames]
            
            # TODO: Implement tokenization for each class prompt
            # Examples:
            
            # For KEEP:
            # tokenized_prompts = tokenizer(cls_prompts, max_length=256, padding='max_length', 
            #                             truncation=True, return_tensors='pt')
            # with torch.no_grad():
            #     embedding = model.text.get_input_embeddings()(tokenized_prompts['input_ids'].to(device))
            
            # For PLIP:
            # tokenized_prompts = tokenizer(cls_prompts, add_special_tokens=True, max_length=77,
            #                             pad_to_max_length=True, return_tensors='pt').to(device)
            # with torch.no_grad():
            #     config = model.text_model.config
            #     embed_token = nn.Embedding(config.vocab_size, config.hidden_size).to(device)
            #     embedding = embed_token(tokenized_prompts['input_ids'].to(device))
            
            # For MUSK:
            # tokenized_prompts = []
            # tokenized_prompts_pad = []
            # for cls_prompt in cls_prompts:
            #     tps, tpsd = musk_utils.xlm_tokenizer(cls_prompt, tokenizer, max_len=100)
            #     tokenized_prompts.append(torch.tensor(tps).unsqueeze(0))
            #     tokenized_prompts_pad.append(torch.tensor(tpsd).unsqueeze(0))
            # tokenized_prompts = torch.cat(tokenized_prompts)
            # tokenized_prompts_pad = torch.cat(tokenized_prompts_pad)
            # with torch.no_grad():
            #     embedding = model.beit3.text_embed(tokenized_prompts.to(device))
            
            # For CONCH:
            # tokenized_prompts = conch_tokenizer.tokenize(tokenizer, cls_prompts)
            # with torch.no_grad():
            #     _, embedding = model._encode_text(tokenized_prompts.to(device))
            
            # TODO: Replace with your model's tokenization and embedding methods
            tokenized_prompts = None  # Implement tokenization here
            with torch.no_grad():
                embedding = None  # Implement text embedding here
            
            tokenized_prompts_lst.append(tokenized_prompts)
            embeddings.append(embedding)
            
            # TODO: Add padding mask handling if needed (for MUSK-like models)
            # tokenized_prompts_pad_lst.append(tokenized_prompts_pad)
            
        # Extract the first element from each embedding for initialization
        init_embedding = torch.stack([embedding[0] for embedding in embeddings])
        
        # TODO: Handle tokenized prompts format based on your model
        # For models with dict format (KEEP, PLIP):
        # tokenized_prompts = {
        #     key: torch.stack(
        #         [single_prompt[key][0, :] for single_prompt in tokenized_prompts_lst]
        #     )
        #     for key in ["input_ids", "attention_mask"]  # Add other keys as needed
        # }
        
        # For models with tensor format (CONCH, MUSK):
        # tokenized_prompts = torch.stack([tps[0] for tps in tokenized_prompts_lst])
        
        # TODO: Replace with your model's tokenized prompt format
        tokenized_prompts = None  # Implement based on your model
        
        # Register buffers for prefix and suffix tokens
        self.register_buffer("token_prefix", init_embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", init_embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        
        # TODO: Add model-specific buffers if needed (like MUSK padding masks)
        # For MUSK-like models:
        # tokenized_prompts_pad = torch.stack([tps_pad[0] for tps_pad in tokenized_prompts_pad_lst])
        # self.register_buffer('prefix_pad', tokenized_prompts_pad[:, :1])
        # self.register_buffer('suffix_pad', tokenized_prompts_pad[:, 1 + n_ctx:])
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.prompt_prefix = prompt_prefix
        self.tokenized_prompts = tokenized_prompts
        self.embeddings = embeddings
        self.tokenized_prompts_lst = tokenized_prompts_lst

    def forward(self):
        ctx = self.ctx
        prefix = self.token_prefix
        suffix = self.token_suffix

        # Concatenate prompt vectors here. ctx is the learnable vector part, 
        # prefix and suffix are fixed vectors corresponding to class tokens
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        # TODO: Handle padding masks if needed (for MUSK-like models)
        # For MUSK:
        # ctx_pad = self.ctx_pad
        # prefix_pad = self.prefix_pad
        # suffix_pad = self.suffix_pad
        # prompts_pad = torch.cat([prefix_pad, ctx_pad, suffix_pad], dim=1)
        # return prompts, prompts_pad
        
        return prompts  # The prompt here is already token embedding
    
    def change_classnames(self):
        """Change classnames randomly for data augmentation"""
        idxs = [random.randrange(0, len(embedding)) for embedding in self.embeddings]
        init_embedding = torch.stack([self.embeddings[i][idx] for i, idx in enumerate(idxs)])
        
        # TODO: Handle different tokenized prompt formats
        # For dict format:
        # self.tokenized_prompts = {
        #     key: torch.stack([self.tokenized_prompts_lst[i][key][idx] for i, idx in enumerate(idxs)])
        #     for key in self.tokenized_prompts.keys()
        # }
        
        # For tensor format:
        # self.tokenized_prompts = torch.stack([self.tokenized_prompts_lst[i][idx] for i, idx in enumerate(idxs)])
        
        self.token_prefix = init_embedding[:, :1, :]
        self.token_suffix = init_embedding[:, 1 + self.n_ctx :, :]

class PPTYOUR_MODEL(nn.Module):
    """PathPT model for YOUR_MODEL"""
    def __init__(self, cfg, classnames_lst, model, tokenizer, device, param, vfeat_dim):
        super().__init__()
        self.prompt_learner = PromptLearnerYOUR_MODEL(cfg, classnames_lst, model, tokenizer, device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.model = model
        self.device = device
        
        ## Learnable prompt embedding
        self.learnable = param['learnable']
        if self.learnable != 'token':
            self.prompt_embedding = nn.Parameter(torch.randn(len(classnames_lst), vfeat_dim) * 0.01, requires_grad=True)
        
        self.vision_only = param['vision_only']
        self.vision_grad = param['vision_grad']
        
        self.vfeat_dim = vfeat_dim
        if self.vision_only:
            # Vision-only mode: direct classification from visual features
            self.mlp = ConvTransAttentionAgg(dim=self.vfeat_dim, cls_num=len(classnames_lst)-1)
        elif self.vision_grad:
            # Vision gradient mode: transform visual features before text matching
            self.mlp = MultiKernelConv1DTrans(in_channels=self.vfeat_dim, out_channels=self.vfeat_dim)

    def forward(self, image_features):
        """
        Forward pass
        
        Args:
            image_features: Visual features from patches (batch, vfeat_dim)
            
        Returns:
            wsi_logits: WSI-level predictions (if vision_only mode)
            patch_logits: Patch-level predictions
        """
        wsi_logits = None
        
        if self.vision_only:
            image_features = image_features.requires_grad_(True)
            image_logits = self.mlp(image_features.squeeze())
            return image_logits, None
        
        elif self.vision_grad:
            image_features = image_features.requires_grad_(True)
            image_features = self.mlp(image_features)
        
        # Normalize image features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        if self.learnable == 'token':
            # Use learnable tokens
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            
            # TODO: Implement text encoding for your model
            # Examples:
            
            # For KEEP:
            # text_features = self.model.text(inputs_embeds=prompts, 
            #                               attention_mask=tokenized_prompts['attention_mask'].to(self.device))
            # text_features = torch.nn.functional.normalize(text_features.pooler_output)
            
            # For PLIP:
            # token_pos_embeddings = self.model.text_model.embeddings(inputs_embeds=prompts)
            # text_outputs = self.model.text_model(
            #     input_ids=self.tokenized_prompts['input_ids'],
            #     attention_mask=self.tokenized_prompts['attention_mask'],
            #     token_pos_emddings=token_pos_embeddings
            # )
            # text_features = self.model.text_projection(text_outputs[1])
            # text_features = torch.nn.functional.normalize(text_features, dim=-1)
            
            # For MUSK:
            # prompts, prompts_pad = self.prompt_learner()
            # encoder_out = self.model.beit3.encoder(
            #     src_tokens=None,
            #     encoder_padding_mask=prompts_pad.to(self.device),
            #     token_embeddings=prompts.to(self.device),
            #     # ... other parameters
            # )
            # text_features = encoder_out['encoder_out'][:,0,:]
            # text_features = self.model.language_head(text_features)
            # text_features = torch.nn.functional.normalize(text_features)
            
            # For CONCH:
            # text_features = self.text_encoder(prompts, tokenized_prompts)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # TODO: Replace with your model's text encoding method
            text_features = None  # Implement text encoding here
            
        elif self.learnable == 'embedding':
            # Use learnable embeddings directly
            text_features = self.prompt_embedding.to(self.device)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        elif self.learnable == 'both':
            # Use both learnable embeddings and tokens
            emb_text_features = self.prompt_embedding.to(self.device)
            emb_text_features = emb_text_features / emb_text_features.norm(dim=-1, keepdim=True)
            
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            
            # TODO: Implement text encoding for your model (same as 'token' case)
            token_text_features = None  # Implement text encoding here
            
            text_features = emb_text_features + token_text_features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity logits
        logits = image_features @ text_features.t()
        
        # Handle different batch sizes
        if len(logits.shape) == 1:  # batch_size = 1
            logits = logits.unsqueeze(0)
        elif len(logits.shape) == 3:
            logits = logits.squeeze(0)
        
        # Apply softmax to get patch-level predictions
        patch_logits = torch.nn.functional.softmax(logits * 10, -1)
        
        return wsi_logits, patch_logits

class CustomYOUR_MODEL(nn.Module):
    """Custom wrapper for YOUR_MODEL (alternative implementation)"""
    def __init__(self, cfg, classnames_lst, model, tokenizer, device, param, vfeat_dim):
        super().__init__()
        self.prompt_learner = PromptLearnerYOUR_MODEL(cfg, classnames_lst, model, tokenizer, device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.model = model
        self.device = device
        
        ## Learnable prompt embedding
        self.learnable = param['learnable']
        if self.learnable != 'token':
            self.prompt_embedding = nn.Parameter(torch.randn(len(classnames_lst), vfeat_dim) * 0.01, requires_grad=True)
        
        self.vision_only = param['vision_only']
        self.vision_grad = param['vision_grad']
        
        self.vfeat_dim = vfeat_dim
        if self.vision_only:
            self.mlp = nn.Sequential(
                nn.Linear(self.vfeat_dim, len(classnames_lst))
            )
        elif self.vision_grad:
            self.mlp = nn.Sequential(
                nn.Linear(self.vfeat_dim, self.vfeat_dim),
                nn.ReLU(),
                nn.Linear(self.vfeat_dim, self.vfeat_dim)
            )

    def forward(self, image_features):
        """Forward pass for custom model"""
        if self.vision_only:
            image_features = image_features.requires_grad_(True)
            image_logits = self.mlp(image_features)
            return image_logits, -1
        
        elif self.vision_grad:
            image_features = image_features.requires_grad_(True)
            image_features = self.mlp(image_features)
        
        # Normalize image features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        if self.learnable == 'token':
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            
            # TODO: Implement text encoding (same as PPTYOUR_MODEL)
            text_features = None  # Implement text encoding here
            
        elif self.learnable == 'embedding':
            text_features = self.prompt_embedding.to(self.device)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        elif self.learnable == 'both':
            # TODO: Implement combined approach (same as PPTYOUR_MODEL)
            text_features = None  # Implement combined approach here

        logits = image_features @ text_features.t()
        
        if len(logits.shape) == 1:  # batch_size = 1
            logits = logits.unsqueeze(0)
            
        correct_logits = torch.nn.functional.softmax(logits * 10, -1)
        
        return correct_logits, text_features

class OriginYOUR_MODEL(nn.Module):
    """Original YOUR_MODEL without prompt learning"""
    def __init__(self, prompts, model, tokenizer, device):
        """
        Initialize with pre-defined prompts
        
        Args:
            prompts: List of prompts corresponding to each class
            model: Pre-trained foundation model
            tokenizer: Tokenizer for the model
            device: Device to use
        """
        super().__init__()
        
        # TODO: Implement tokenization and text encoding for your model
        # Examples:
        
        # For KEEP:
        # self.prompts = tokenizer(prompts, max_length=256, padding='max_length', 
        #                        truncation=True, return_tensors='pt').to(device)
        # text_feature = model.encode_text(self.prompts)
        
        # For PLIP:
        # self.prompts = tokenizer(list(prompts), add_special_tokens=True, max_length=77,
        #                        pad_to_max_length=True, return_tensors='pt').to(device)
        # text_feature = model.get_text_features(**self.prompts)
        # text_feature = torch.nn.functional.normalize(text_feature, dim=-1)
        
        # For MUSK:
        # text_ids = []
        # paddings = []
        # for prompt in prompts:
        #     txt_ids, pad = musk_utils.xlm_tokenizer(prompt, tokenizer, max_len=100)
        #     text_ids.append(torch.tensor(txt_ids).unsqueeze(0))
        #     paddings.append(torch.tensor(pad).unsqueeze(0))
        # text_ids = torch.cat(text_ids)
        # paddings = torch.cat(paddings)
        # with torch.inference_mode():
        #     text_feature = model(text_description=text_ids.to(device),
        #                         padding_mask=paddings.to(device),
        #                         with_head=True, out_norm=True)[1]
        
        # For CONCH:
        # tokenizer = conch_tokenizer.get_tokenizer()
        # self.prompts = conch_tokenizer.tokenize(tokenizer, prompts).to(device)
        # text_feature = model.encode_text(self.prompts)
        # text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        
        # TODO: Replace with your model's tokenization and encoding methods
        self.prompts = None  # Implement tokenization here
        text_feature = None  # Implement text encoding here
        
        self.text_features = text_feature.to(device)
        self.device = device

    def forward(self, image_features):
        """Forward pass for original model"""
        image_features = image_features.to(self.device)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.text_features.t()
        correct_logits = torch.nn.functional.softmax(logits * 10, -1)
        return correct_logits