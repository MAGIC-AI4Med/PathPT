import torch
import torch.nn as nn
import random
from transmil import TransMIL
from model_utils import MultiKernelConv1DTrans, AttentionAgg, ConvAttentionAgg, ConvTransAttentionAgg
import musk.utils as musk_utils
import musk.modeling as musk_modeling

def prompt_padding(ctx, length, mode='repeat'):
    if mode == 'repeat':
        # Split ctx into word list by spaces
        words = ctx.split()
        # Calculate the number of repetitions needed
        repeat_times = (length + len(words) - 1) // len(words)
        # Repeat the word list
        padded_words = (words * repeat_times)[:length]
        # Recombine the word list into a string
        padded_ctx = ' '.join(padded_words)
    else:
        raise ValueError("Unsupported mode. Currently only 'repeat' mode is supported.")
    
    return padded_ctx

class PromptLearnerMUSK(nn.Module):
    def __init__(self, cfg, classnames_lst, model, tokenizer, device):
        super().__init__()
        self.device = device
        n_cls = len(classnames_lst)
        n_ctx = cfg.n_ctx
        ctx_init = cfg.ctx_init
        ctx_dim = cfg.token_embedding_size
        model = model.to(self.device)

        print("Initializing a generic context")
        if ctx_init: # NOTE: tokenized_ctx len != ctx_init len, because some words are tokenized into multiple tokens
            # use given words to initialize context vectors
            # ctx_init = prompt_padding(ctx_init, n_ctx)
            prompt, prompt_pad = musk_utils.xlm_tokenizer(ctx_init, tokenizer, max_len=100)
            
            prompt = torch.cat([torch.tensor(prompt).unsqueeze(0) for i in range(n_cls)]) # (n_cls, 100)
            prompt_pad = torch.cat([torch.tensor(prompt_pad).unsqueeze(0) for i in range(n_cls)]) # (n_cls, 100)
            
            with torch.no_grad():
                embedding = model.beit3.text_embed(prompt.to(device)) # (n_cls, 100, 1024)
            ctx_vectors = embedding[:, 1 : 1 + n_ctx, :] # [cls] is in the first place
            ctx_pad = prompt_pad[:, 1 : 1 + n_ctx].to(device) # [cls] is in the first place
            prompt_prefix = ctx_init

        else:
            # ctx_vectors = torch.empty(n_ctx, ctx_dim).to(device)
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim).to(device)
            ctx_pad = torch.zeros(n_cls, n_ctx).to(device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # turn it into a torch parameter dtype to be optimized
        self.register_buffer("ctx_pad", ctx_pad)  # register as a buffer so it is not optimized
        embeddings = [] # Has n_cls elements, each element is the prompt embedding set of all classnames for that cls (n_cls,n_classname,100,1024)
        tokenized_prompts_lst = []
        tokenized_prompts_pad_lst = []
        print(f'classnames: {classnames_lst}')
        for classnames in classnames_lst:
            # cls_prompts = [prompt_prefix + " " + name + "." for name in classnames]
            cls_prompts = [" ".join(["X"] * n_ctx) + " " + name + "." for name in classnames]
            tokenized_prompts = []
            tokenized_prompts_pad = []
            for cls_prompt in cls_prompts:
                tps, tpsd = musk_utils.xlm_tokenizer(cls_prompt, tokenizer, max_len=100) # (n_classname, 100)
                tokenized_prompts.append(torch.tensor(tps).unsqueeze(0)) # (1, 100)
                tokenized_prompts_pad.append(torch.tensor(tpsd).unsqueeze(0)) # (1, 100)
            tokenized_prompts = torch.cat(tokenized_prompts) # (n_classname, 100)
            tokenized_prompts_pad = torch.cat(tokenized_prompts_pad) # (n_classname, 100)
            tokenized_prompts_lst.append(tokenized_prompts)
            tokenized_prompts_pad_lst.append(tokenized_prompts_pad)
            
            with torch.no_grad():
                embedding = model.beit3.text_embed(tokenized_prompts.to(device)) # (n_classname, 100, 1024)
            embeddings.append(embedding)
            
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        init_embedding = torch.stack([embedding[0] for embedding in embeddings]) #(n_cls, 100, 1024)
        # Extract and concatenate the first element
        tokenized_prompts = torch.stack([tps[0] for tps in tokenized_prompts_lst])# (n_cls, 100)
        tokenized_prompts_pad = torch.stack([tps_pad[0] for tps_pad in tokenized_prompts_pad_lst])# (n_cls, 100)
        #print(init_embedding.shape)


        self.register_buffer("token_prefix", init_embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", init_embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.register_buffer('prefix_pad', tokenized_prompts_pad[:, :1])  # SOS
        self.register_buffer('suffix_pad', tokenized_prompts_pad[:, 1 + n_ctx:])  # CLS, EOS
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.prompt_prefix = prompt_prefix
        # self.tokenized_prompts = tokenized_prompts
        # self.tokenized_prompts_pad = tokenized_prompts_pad
        self.embeddings = embeddings
        # self.tokenized_prompts_lst = tokenized_prompts_lst
        # self.tokenized_prompts_pad_lst = tokenized_prompts_pad_lst

    def forward(self):
        ctx = self.ctx
        # if ctx.dim() == 2:
        #     ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        
        ctx_pad = self.ctx_pad
        prefix_pad = self.prefix_pad
        suffix_pad = self.suffix_pad

        # Here, prompt vector concatenation is performed. ctx is the learnable vector part, prefix and suffix are fixed vectors corresponding to class tokens
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        
        prompts_pad = torch.cat(
            [
                prefix_pad,  # (n_cls, 1)
                ctx_pad,     # (n_cls, n_ctx)
                suffix_pad,  # (n_cls, *)
            ],
            dim=1,
        )

        return prompts, prompts_pad # The prompt here is already token embedding  
    
    # def change_classnames(self):
    #     idxs = [random.randrange(0, len(embedding)) for embedding in self.embeddings]
    #     init_embedding = torch.stack([self.embeddings[i][idx] for i, idx in enumerate(idxs)])
    #     self.tokenized_prompts = torch.stack([self.tokenized_prompts_lst[i][idx] for i, idx in enumerate(idxs)])#TODO
    #     self.token_prefix = init_embedding[:, :1, :]
    #     self.token_suffix = init_embedding[:, 1 + self.n_ctx :, :]
        
    
class CustomMUSK(nn.Module):
    def __init__(self, cfg, classnames_lst, model, tokenizer, device, param, vfeat_dim):
        super().__init__()
        self.prompt_learner = PromptLearnerMUSK(cfg, classnames_lst, model, tokenizer, device)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # self.tokenized_prompts_pad = self.prompt_learner.tokenized_prompts_pad
        self.model = model
        self.device = device
        
        ## learnable prompt embedding
        self.learnable = param['learnable']
        if self.learnable != 'token':
            self.prompt_embedding = nn.Parameter(torch.randn(len(classnames_lst), vfeat_dim) * 0.01, requires_grad=True)
        #############
        
        self.vision_only = param['vision_only']
        self.vision_grad = param['vision_grad']
        
        
        self.vfeat_dim = vfeat_dim
        if self.vision_only:
            self.mlp = nn.Sequential(
                # nn.Linear(self.vfeat_dim, 32),
                # nn.ReLU(),
                nn.Linear(self.vfeat_dim, len(classnames_lst))
            )
        elif self.vision_grad:
            self.mlp = nn.Sequential(
                nn.Linear(self.vfeat_dim, self.vfeat_dim),
                nn.ReLU(),
                nn.Linear(self.vfeat_dim, self.vfeat_dim)
            )
            
    def forward(self, image_features): #(batch, 768)
        
        if self.vision_only:
            image_features = image_features.requires_grad_(True)
            # print(image_features.requires_grad)
            image_logits = self.mlp(image_features)
            # print(image_logits.requires_grad)
            
            return image_logits, -1
        
        elif self.vision_grad:
            image_features = image_features.requires_grad_(True)
            image_features = self.mlp(image_features)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if self.learnable == 'token':
            prompts, prompts_pad = self.prompt_learner()
            encoder_out = self.model.musk_model.beit3.encoder(src_tokens=None,
                                                                encoder_padding_mask=prompts_pad.to(self.device),
                                                                attn_mask=None,
                                                                token_embeddings=prompts.to(self.device),
                                                                multiway_split_position=0,
                                                                incremental_state=None,
                                                                positions=None,
            ) # The reason for placing the encoder here instead of pre-computing the embedding is to backpropagate gradients through the encoder
            text_features = encoder_out['encoder_out'][:,0,:]
            text_features = self.model.language_head(text_features)
            text_features = torch.nn.functional.normalize(text_features)
        elif self.learnable == 'embedding':
            text_features = self.prompt_embedding.to(self.device)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        elif self.learnable == 'both':
            emb_text_features = self.prompt_embedding.to(self.device)
            emb_text_features = emb_text_features / emb_text_features.norm(dim=-1, keepdim=True)
            
            prompts, prompts_pad = self.prompt_learner()
            encoder_out = self.model.musk_model.beit3.encoder(src_tokens=None,
                                                                encoder_padding_mask=prompts_pad.to(self.device),
                                                                attn_mask=None,
                                                                token_embeddings=prompts.to(self.device),
                                                                multiway_split_position=0,
                                                                incremental_state=None,
                                                                positions=None,
            ) # The reason for placing the encoder here instead of pre-computing the embedding is to backpropagate gradients through the encoder
            token_text_features = encoder_out['encoder_out'][:,0,:]
            token_text_features = self.model.language_head(token_text_features)
            token_text_features = torch.nn.functional.normalize(token_text_features)
            
            text_features = emb_text_features + token_text_features
            
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = image_features.float() @ text_features.t().float()
        #print(prompts.shape, tokenized_prompts.shape, text_features.shape, logits.shape)
        if len(logits.shape) == 1: # batch_size = 1
            logits = logits.unsqueeze(0)
        correct_logits = torch.nn.functional.softmax(logits*10,-1)

        return correct_logits, text_features #sims

class PPTMUSK(nn.Module):
    def __init__(self, cfg, classnames_lst, model, tokenizer, device, param, vfeat_dim):
        super().__init__()
        self.prompt_learner = PromptLearnerMUSK(cfg, classnames_lst, model, tokenizer, device)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.model = model
        self.device = device
        
        ## learnable prompt embedding
        self.learnable = param['learnable']
        if self.learnable != 'token':
            self.prompt_embedding = nn.Parameter(torch.randn(len(classnames_lst), vfeat_dim) * 0.01, requires_grad=True)
        #############
        
        self.vision_only = param['vision_only']
        self.vision_grad = param['vision_grad']
        self.vision_mil = param['vision_mil']
        
        
        self.vfeat_dim = vfeat_dim
        if self.vision_only:
            # self.mlp = nn.Sequential(
            #     # nn.Linear(self.vfeat_dim, self.vfeat_dim),
            #     # nn.ReLU(),
            #     nn.Linear(self.vfeat_dim, len(classnames_lst))
            # )
            # self.mlp = MultiKernelConv1DTrans(in_channels=self.vfeat_dim, out_channels=768, cls_num = len(classnames_lst))
            self.mlp = ConvTransAttentionAgg(dim=self.vfeat_dim, cls_num = len(classnames_lst)-1)
        elif self.vision_grad:
            if self.vision_mil:
                self.mil = TransMIL(len(classnames_lst)-1,vfeat_dim)
            else:
                # self.mlp = nn.Sequential(
                #     nn.Linear(self.vfeat_dim, self.vfeat_dim),
                #     nn.ReLU(),
                #     nn.Linear(self.vfeat_dim, self.vfeat_dim)
                # )
                self.mlp = MultiKernelConv1DTrans(in_channels=self.vfeat_dim, out_channels=1024)

    def forward(self, image_features): #(batch, 768)
        
        if self.vision_only:
            image_features = image_features.requires_grad_(True)
            # print(image_features.requires_grad)
            image_logits = self.mlp(image_features.squeeze())
            # print(image_logits.requires_grad)
            
            return image_logits, None
        
        elif self.vision_grad:
            image_features = image_features.requires_grad_(True)
            # print(image_features.requires_grad)
            
            if self.vision_mil:
                results_dict = self.mil(image_features)
                image_features = results_dict['patch_feats']
            else:
                image_features = self.mlp(image_features)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if self.learnable == 'token':
            prompts, prompts_pad = self.prompt_learner()
            encoder_out = self.model.beit3.encoder(src_tokens=None,
                                                                encoder_padding_mask=prompts_pad.to(self.device),
                                                                attn_mask=None,
                                                                token_embeddings=prompts.to(self.device),
                                                                multiway_split_position=0,
                                                                incremental_state=None,
                                                                positions=None,
            ) # The reason for placing the encoder here instead of pre-computing the embedding is to backpropagate gradients through the encoder
            text_features = encoder_out['encoder_out'][:,0,:]
            text_features = self.model.language_head(text_features)
            text_features = torch.nn.functional.normalize(text_features)
        elif self.learnable == 'embedding':
            text_features = self.prompt_embedding.to(self.device)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        elif self.learnable == 'both':
            emb_text_features = self.prompt_embedding.to(self.device)
            emb_text_features = emb_text_features / emb_text_features.norm(dim=-1, keepdim=True)
            
            prompts, prompts_pad = self.prompt_learner()
            encoder_out = self.model.beit3.encoder(src_tokens=None,
                                                                encoder_padding_mask=prompts_pad.to(self.device),
                                                                attn_mask=None,
                                                                token_embeddings=prompts.to(self.device),
                                                                multiway_split_position=0,
                                                                incremental_state=None,
                                                                positions=None,
            ) # The reason for placing the encoder here instead of pre-computing the embedding is to backpropagate gradients through the encoder
            token_text_features = encoder_out['encoder_out'][:,0,:]
            token_text_features = self.model.language_head(token_text_features)
            token_text_features = torch.nn.functional.normalize(token_text_features)
            
            text_features = emb_text_features + token_text_features
            
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = image_features.float() @ text_features.t().float()
        #print(prompts.shape, tokenized_prompts.shape, text_features.shape, logits.shape)
        if len(logits.shape) == 1: # batch_size = 1
            logits = logits.unsqueeze(0)
        elif len(logits.shape) == 3:
            logits = logits.squeeze(0)
        patch_logits = torch.nn.functional.softmax(logits*10,-1)
        
        if self.vision_mil:
            wsi_logits = results_dict['logits']
        else:
            wsi_logits = None
        
        return wsi_logits, patch_logits #sims


class OriginMUSK(nn.Module):
    def __init__(self, prompts, model, tokenizer, device): # prompts is a list, elements are prompts corresponding to each class
        super().__init__()
        model.to(device)
        text_ids = []
        paddings = []
        for prompt in prompts:
            txt_ids, pad = musk_utils.xlm_tokenizer(prompt, tokenizer, max_len=100)
            text_ids.append(torch.tensor(txt_ids).unsqueeze(0))
            paddings.append(torch.tensor(pad).unsqueeze(0))
        text_ids = torch.cat(text_ids)
        paddings = torch.cat(paddings)
        with torch.inference_mode():
            self.text_features = model(
                text_description=text_ids.to(device),
                padding_mask=paddings.to(device),
                with_head=True, 
                out_norm=True
            )[1]
        self.device = device

    def forward(self, image_features,):#ensemble is a useless parameter set to match CustomCONCH
        image_features = image_features.to(self.device)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features.float() @ self.text_features.t().float()
        correct_logits = torch.nn.functional.softmax(logits*10,-1)
        return correct_logits
