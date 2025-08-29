import torch
import torch.nn as nn
import random
from nystrom_attention import NystromAttention
import open_clip_CONCH.custom_tokenizer as conch_tokenizer
from model_utils import MultiKernelConv1DTrans
from torch.nn import functional as F

def prompt_padding(ctx, length, mode='repeat'):
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

class CONCHTextEncoder(nn.Module):
    def __init__(self, model, embed_cls):
        super().__init__()
        text_model = model.text
        self.transformer = text_model.transformer
        self.positional_embedding = text_model.positional_embedding
        self.ln_final = text_model.ln_final
        self.text_projection = text_model.text_projection
        self.dtype = self.transformer.get_cast_dtype()
        self.embed_cls = embed_cls
        self.attn_mask = text_model.attn_mask
        self.heads = text_model.heads
        self.pad_id = text_model.pad_id
        self.output_tokens = text_model.output_tokens
        
        if self.embed_cls:
            self.cls_emb = text_model.cls_emb
        else:
            self.cls_emb = None
        
    def build_cls_mask(self, text_tokens, cast_dtype: torch.dtype):
        cls_mask = (text_tokens != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def forward(self, x, prompt_tokens):
        prompt_tokens = prompt_tokens[:, :-1] if self.embed_cls else prompt_tokens # make space for CLS token
        cast_dtype = self.dtype
        seq_len = x.shape[1]
        x = x.to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(prompt_tokens, cast_dtype).to(attn_mask.device)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[torch.arange(x.shape[0]), prompt_tokens.argmax(dim=-1)], x

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled

class PromptLearnerCONCH(nn.Module): # tokenizer limits prompt length to 128
    def __init__(self, cfg, classnames_lst, model, device):
        super().__init__()
        self.device = device
        n_cls = len(classnames_lst)
        n_ctx = cfg.n_ctx
        ctx_init = cfg.ctx_init
        ctx_dim = cfg.token_embedding_size
        tokenizer = conch_tokenizer.get_tokenizer()
        model = model.to(self.device)

        print("Initializing a generic context")
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = prompt_padding(ctx_init, n_ctx)
            prompt = conch_tokenizer.tokenize(tokenizer, [ctx_init]*n_cls)
            with torch.no_grad(): # freeze model parameters here
                _, embedding = model._encode_text(prompt.to(device)) 
            ctx_vectors = embedding[:, 1 : 1 + n_ctx, :] # [cls] is in the first place
            prompt_prefix = ctx_init

        else:
            # ctx_vectors = torch.empty(n_ctx, ctx_dim).to(device)
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim).to(device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # turn it into a torch parameter dtype to be optimized
        embeddings = [] # has n_cls elements, each element is the prompt embedding collection of all classnames for that cls (n_cls, n_classname, 128, 768)
        tokenized_prompts_lst = []
        print(f'classnames: {classnames_lst}')
        for classnames in classnames_lst:
            # cls_prompts = [prompt_prefix + " " + name + "." for name in classnames]
            cls_prompts = [" ".join(["X"] * n_ctx) + " " + name + "." for name in classnames]
            tokenized_prompts = conch_tokenizer.tokenize(tokenizer, cls_prompts) # (n_classname, 128)
            tokenized_prompts_lst.append(tokenized_prompts)
            with torch.no_grad(): # freeze model parameters here
                _, embedding = model._encode_text(tokenized_prompts.to(device)) # (n_classname, 128, 768)
            embeddings.append(embedding)
            
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        init_embedding = torch.stack([embedding[0] for embedding in embeddings]) #(n_cls, 128, 768)
        tokenized_prompts = torch.stack([tps[0] for tps in tokenized_prompts_lst])
        #print(init_embedding.shape)

        self.register_buffer("token_prefix", init_embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", init_embedding[:, 1 + n_ctx :, :])  # CLS, EOS
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

        # Concatenate prompt vectors here. ctx is the learnable vector part, prefix and suffix are fixed vectors corresponding to class tokens
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts # the prompt here is already token embedding  
    
    def change_classnames(self):
        idxs = [random.randrange(0, len(embedding)) for embedding in self.embeddings]
        init_embedding = torch.stack([self.embeddings[i][idx] for i, idx in enumerate(idxs)])
        self.tokenized_prompts = torch.stack([self.tokenized_prompts_lst[i][idx] for i, idx in enumerate(idxs)])
        self.token_prefix = init_embedding[:, :1, :]
        self.token_suffix = init_embedding[:, 1 + self.n_ctx :, :]


class PPTCONCH(nn.Module):
    def __init__(self, cfg, classnames_lst, model, device, param, vfeat_dim):
        super().__init__()
        self.prompt_learner = PromptLearnerCONCH(cfg, classnames_lst, model, device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.model = model
        self.device = device
        
        ## learnable prompt embedding
        self.learnable = param['learnable']
        if self.learnable != 'token':
            self.prompt_embedding = nn.Parameter(torch.randn(len(classnames_lst), vfeat_dim) * 0.01, requires_grad=True)
        #############
        
        self.vision_only = param['vision_only']
        self.vision_grad = param['vision_grad']
        self.text_encoder = CONCHTextEncoder(model.to(device), embed_cls = True)
        
        self.vfeat_dim = vfeat_dim
        if self.vision_only:
            self.mlp = nn.Sequential(
                # nn.Linear(self.vfeat_dim, 32),
                # nn.ReLU(),
                nn.Linear(self.vfeat_dim, len(classnames_lst))
            )
        elif self.vision_grad:
            self.mlp = MultiKernelConv1DTrans(in_channels=self.vfeat_dim, out_channels=self.vfeat_dim)

    def forward(self, image_features): #(batch, 768)
        
        if self.vision_only:
            image_features = image_features.requires_grad_(True)
            # print(image_features.requires_grad)
            image_logits = self.mlp(image_features)
            # print(image_logits.requires_grad)
            
            return -1, image_logits
        
        elif self.vision_grad:
            image_features = image_features.requires_grad_(True)
            # print(image_features.requires_grad)
            
            image_features = self.mlp(image_features)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if self.learnable == 'token':
            prompt_token_embedding = self.prompt_learner()
            prompt_tokens = self.tokenized_prompts
            # text_features = self.model.text(inputs_embeds=prompts, attention_mask=tokenized_prompts['attention_mask'].to(self.device)) # The reason to put encoder here instead of calculating embedding in advance is to backpropagate gradients through encoder
            text_features,_ = self.text_encoder(prompt_token_embedding, prompt_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        elif self.learnable == 'embedding':
            text_features = self.prompt_embedding.to(self.device)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        elif self.learnable == 'both':
            emb_text_features = self.prompt_embedding.to(self.device)
            emb_text_features = emb_text_features / emb_text_features.norm(dim=-1, keepdim=True)
            
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            token_text_features = self.model.text(inputs_embeds=prompts, attention_mask=tokenized_prompts['attention_mask'].to(self.device)) # The reason to put encoder here instead of calculating embedding in advance is to backpropagate gradients through encoder
            # print(self.prompt_learner.ctx)
            token_text_features = torch.nn.functional.normalize(token_text_features.pooler_output)
            
            text_features = emb_text_features + token_text_features
            
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.t()
        #print(prompts.shape, tokenized_prompts.shape, text_features.shape, logits.shape)
        if len(logits.shape) == 1: # batch_size = 1
            logits = logits.unsqueeze(0)
        elif len(logits.shape) == 3:
            logits = logits.squeeze(0)
        patch_logits = torch.nn.functional.softmax(logits*10,-1)

        wsi_logits = None
        
        return wsi_logits, patch_logits #sims


class CustomCONCH(nn.Module):
    def __init__(self, cfg, classnames_lst, model, tokenizer, device, param, vfeat_dim):
        super().__init__()
        self.prompt_learner = PromptLearnerCONCH(cfg, classnames_lst, model, tokenizer, device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.model = model
        self.device = device

    def forward(self, image_features): #(batch, 768)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        # text_features = self.model.text(inputs_embeds=prompts, attention_mask=tokenized_prompts['attention_mask'].to(self.device)) # The reason to put encoder here instead of calculating embedding in advance is to backpropagate gradients through encoder
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits = image_features @ text_features.t()
        #print(prompts.shape, tokenized_prompts.shape, text_features.shape, logits.shape)
        if len(logits.shape) == 1: # batch_size = 1
            logits = logits.unsqueeze(0)
        elif len(logits.shape) == 3:
            logits = logits.squeeze(0)
            
        correct_logits = torch.nn.functional.softmax(logits*10,-1)
        
        return correct_logits, text_features #sims


class OriginCONCH(nn.Module):
    def __init__(self, prompts, model, device): # prompts is a list, elements are prompts corresponding to each class
        super().__init__()
        tokenizer = conch_tokenizer.get_tokenizer()
        self.prompts = conch_tokenizer.tokenize(tokenizer, prompts).to(device)

        text_feature = model.encode_text(self.prompts)
        self.text_features =  text_feature / text_feature.norm(dim=-1, keepdim=True)
        self.text_features = self.text_features.to(device)
        self.device = device

    def forward(self, image_features): # ensemble is a useless parameter set to match CustomCONCH
        image_features = image_features.to(self.device)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.text_features.t()
        correct_logits = torch.nn.functional.softmax(logits*10,1)
        return correct_logits
