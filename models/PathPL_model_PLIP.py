import torch
import torch.nn as nn
import random
from models.transmil import TransMIL
from model_utils import MultiKernelConv1DTrans, AttentionAgg, ConvAttentionAgg, ConvTransAttentionAgg

def prompt_padding(ctx, length, mode='repeat'):
    if mode == 'repeat':
        # 将 ctx 按空格分割成单词列表
        words = ctx.split()
        # 计算需要重复的次数
        repeat_times = (length + len(words) - 1) // len(words)
        # 重复单词列表
        padded_words = (words * repeat_times)[:length]
        # 将单词列表重新组合成字符串
        padded_ctx = ' '.join(padded_words)
    else:
        raise ValueError("Unsupported mode. Currently only 'repeat' mode is supported.")
    
    return padded_ctx

class PromptLearnerPLIP(nn.Module):
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
            # use given words to initialize context vectors
            ctx_init = prompt_padding(ctx_init, n_ctx)
            # prompt = tokenizer([ctx_init],max_length=256,padding='max_length',truncation=True, return_tensors='pt')
            prompt = tokenizer([ctx_init]*n_cls,add_special_tokens=True,max_length=77,pad_to_max_length=True,return_tensors='pt')
            with torch.no_grad(): # 这里冻结了模型参数
                config = model.text_model.config
                embed_token = nn.Embedding(config.vocab_size, config.hidden_size).to(device)
                embedding = embed_token(prompt['input_ids'].to(device))
            # ctx_vectors = embedding[0, 1 : 1 + n_ctx, :] # [cls] is in the first place
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
        embeddings = [] #有n_cls个元素，每个元素是该cls的所有classname的prompt的embedding集合 (n_cls, n_classname, 256, 768)
        tokenized_prompts_lst = []
        print(f'classnames: {classnames_lst}')
        for classnames in classnames_lst:
            cls_prompts = [" ".join(["X"] * n_ctx) + " " + name + "." for name in classnames]
            tokenized_prompts = tokenizer(cls_prompts,add_special_tokens=True,max_length=77,pad_to_max_length=True,return_tensors='pt').to(device) # (n_classname, 256)
            tokenized_prompts_lst.append(tokenized_prompts)
            with torch.no_grad(): # 这里冻结了模型参数
                embedding = embed_token(tokenized_prompts['input_ids'].to(device)) # (n_classname, 256, 768)
            embeddings.append(embedding)
            
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        init_embedding = torch.stack([embedding[0] for embedding in embeddings]) #(n_cls, 256, 768)
        # 提取并拼接第一个元素
        
        tokenized_prompts = {
            key: torch.stack(
                [single_prompt[key][0, :] for single_prompt in tokenized_prompts_lst],  # 提取每个 tokenized_prompt 的第一个元素（保持二维）
            )
            for key in ["input_ids", "attention_mask"]
        }
        #tokenized_prompts = torch.stack([tps[0] for tps in tokenized_prompts_lst])
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
        # if ctx.dim() == 2:
        #     ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        # 这里进行了prompt vector的拼接。ctx是需要学习的vector部分，prefix和suffix是class token对应的固定向量
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts # 这里的prompt已经是token embedding了  
    
    def change_classnames(self):
        idxs = [random.randrange(0, len(embedding)) for embedding in self.embeddings]
        init_embedding = torch.stack([self.embeddings[i][idx] for i, idx in enumerate(idxs)])
        self.tokenized_prompts = torch.stack([self.tokenized_prompts_lst[i][idx] for i, idx in enumerate(idxs)])#TODO
        self.token_prefix = init_embedding[:, :1, :]
        self.token_suffix = init_embedding[:, 1 + self.n_ctx :, :]
        
    
class PPLPLIP(nn.Module):
    def __init__(self, cfg, classnames_lst, model, tokenizer, device, param, vfeat_dim):
        super().__init__()
        self.prompt_learner = PromptLearnerPLIP(cfg, classnames_lst, model, tokenizer, device)
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
                self.mlp = MultiKernelConv1DTrans(in_channels=self.vfeat_dim, out_channels=self.vfeat_dim, cls_num=len(classnames_lst)-1)

    def forward(self, image_features): #(batch, 768)
        
        wsi_logits = None
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
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            token_pos_embeddings = self.model.text_model.embeddings(inputs_embeds=prompts) # 把encoder放在这里而不是提前算好embedding的原因是要通过encoder反传梯度
            
            text_outputs = self.model.text_model(
                input_ids=self.tokenized_prompts['input_ids'],
                attention_mask=self.tokenized_prompts['attention_mask'],
                position_ids=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=False,
                token_pos_emddings = token_pos_embeddings,
            )

            pooled_output = text_outputs[1]
            text_features = self.model.text_projection(pooled_output)
            
            # print(self.prompt_learner.ctx)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
        elif self.learnable == 'embedding':
            text_features = self.prompt_embedding.to(self.device)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        elif self.learnable == 'both':
            emb_text_features = self.prompt_embedding.to(self.device)
            emb_text_features = emb_text_features / emb_text_features.norm(dim=-1, keepdim=True)
            
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            token_text_features = self.model.text(inputs_embeds=prompts, attention_mask=tokenized_prompts['attention_mask'].to(self.device)) # 把encoder放在这里而不是提前算好embedding的原因是要通过encoder反传梯度
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
        
        # if self.vision_mil:
        #     wsi_logits = results_dict['logits']
        # else:
        #     wsi_logits = None
        
        return wsi_logits, patch_logits #sims


class OriginPLIP(nn.Module):
    def __init__(self, prompts, model, tokenizer, device): #prompts是一个list，元素为每个class对应的prompt
        super().__init__()
        self.prompts = tokenizer(list(prompts),add_special_tokens=True,max_length=77,pad_to_max_length=True,return_tensors='pt').to(device)
        
        text_feature = model.get_text_features(**self.prompts)
        text_feature = torch.nn.functional.normalize(text_feature, dim=-1)
        self.text_features = text_feature.to(device)
        self.device = device

    def forward(self, image_features,):#ensemble为了配合CustomCONCH设置的无用参数
        image_features = image_features.to(self.device)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.text_features.t()
        correct_logits = torch.nn.functional.softmax(logits*10,-1)
        return correct_logits

         