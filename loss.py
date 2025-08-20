import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import copy

def balanced_ce_loss(logits, labels):
    """
    先计算每类样本的平均loss，再对所有类别取平均
    
    参数：
        logits: [batch_size, num_classes]
        labels: [batch_size]
        num_classes: 总类别数
        
    返回：
        loss: 最终按类别平均的损失
        per_class_loss: 每类的平均损失（形状[num_classes]）
    """
    # 计算每个样本的CE损失（不汇总）
    num_classes = logits.shape[-1]
    individual_losses = F.cross_entropy(logits, labels, reduction='none')  # [batch_size]
    
    # 初始化每类统计量
    per_class_loss = torch.zeros(num_classes, device=logits.device)
    per_class_count = torch.zeros(num_classes, device=logits.device)
    
    # 遍历每个类别
    for cls in range(num_classes):
        mask = (labels == cls)
        if mask.any():
            per_class_loss[cls] = individual_losses[mask].mean()  # 该类样本的平均loss
            per_class_count[cls] = 1  # 计数用于后面求平均
    
    # 计算有效类别的平均loss
    valid_classes = (per_class_count > 0)
    loss = per_class_loss[valid_classes].mean()  # 对类别维度取平均
    
    return loss

def compute_min_correct_logits(logits: torch.Tensor, 
                              labels: torch.Tensor) -> torch.Tensor:
    """
    计算每个类别正确预测样本的logits最小值
    
    参数：
        logits: 模型输出 [batch_size, num_classes]
        labels: 真实标签 [batch_size]
    
    返回：
        min_logits: 每个类别的正确logits最小值 [num_classes]
                   (无样本的类别设为NaN)
    """
    # 过滤已知标签样本
    num_classes = logits.shape[1]
    known_mask = (labels >= 0) & (labels < num_classes)
    known_logits = logits[known_mask]
    known_labels = labels[known_mask].long()

    # 初始化结果张量
    min_logits = torch.full((num_classes,), float('nan'), 
                           device=logits.device)

    # 获取预测类别
    pred_classes = torch.argmax(known_logits, dim=1)

    # 遍历每个类别
    for cls in range(num_classes):
        # 筛选该类别且预测正确的样本
        cls_mask = (known_labels == cls) & (pred_classes == cls)
        if cls_mask.any():
            # 提取正确样本对应类别的logits值
            cls_logits = known_logits[cls_mask, cls]
            min_logits[cls] = torch.min(cls_logits)

    return min_logits


def PatchSSLoss(logits, labels, epoch, total_epoch=100, weights = [1,0.5,0.1], balance = False, vision_only = False, pseudo_loss = True):
    """
    参数说明：
    logits: 模型输出 [batch_size, num_classes]
    labels: 原始标签，已知标签>=0，未知标签为负数（表示0和abs(label)类）
    epoch: 当前训练轮次
    total_epoch: 总训练轮次
    
    返回：
    dict: {
        'loss': 总损失,
        'labeled_loss': 已知标签CE损失,
        'pseudo_loss': 伪标签CE损失,
        'candidate_loss': 候选样本损失,
        'valid_pseudo_ratio': 有效伪标签比例
    }
    """
    
    # 动态阈值计算
    top_thd = 2./3.
    threshold = 1.1 if epoch < 10 else top_thd - 0.2 * ((epoch-10)/max(1,total_epoch-10))**0.5
    
    # 初始化损失和统计量
    losses = {
        'labeled_loss': torch.tensor(0.0, device=logits.device),
        'pseudo_loss': torch.tensor(0.0, device=logits.device),
        'candidate_loss': torch.tensor(0.0, device=logits.device),
        'valid_pseudo_ratio': 0.0
    }
    
    logits = logits.view(-1, logits.size(-1))  # [batch*patches, C]
    labels = labels.view(-1)    
    
    # 第一阶段：处理已知标签样本
    known_mask = labels >= 0
    if known_mask.any():
        
        if balance:
            losses['labeled_loss'] = balanced_ce_loss(logits[known_mask], labels[known_mask].long()) #
        else:
            losses['labeled_loss'] = F.cross_entropy(logits[known_mask], labels[known_mask].long())
        #balanced_ce_loss(logits[known_mask], labels[known_mask].long())
        
    
    # 第二阶段：处理候选样本
    if not vision_only:
        candidate_mask = ~known_mask
        if candidate_mask.any():
            # 获取候选类别索引
            k = (-labels[candidate_mask]).long()
            probs = F.softmax(logits[candidate_mask], dim=1)
            p0, pk = probs[:,0], probs[torch.arange(len(k)), k]
            
            # 候选样本的基础损失（始终计算）
            losses['candidate_loss'] = -torch.log(p0 + pk + 1e-8).mean()
            
            # 动态生成伪标签（epoch>=10）
            if epoch >= 10 and pseudo_loss:
                
                # known_labels = labels[known_mask]
                # if known_labels.numel() > 0:
                #     # 提取已知样本的最大logits值
                #     class_thd = compute_min_correct_logits(logits[known_mask], labels[known_mask])
                #     # known_max = logits[known_mask].gather(1, known_labels.view(-1,1)).squeeze(1)  # 取对应真实类别的logit值
                    
                #     # 计算全局默认阈值（所有已知类别的平均）
                #     global_thd = 0.0
                # else:
                #     # 无已知样本时使用固定阈值
                #     class_thd = torch.zeros(logits.shape[1], device=logits.device)
                #     global_thd = 0.0

                # # 阶段2：为候选样本分配对应阈值 -------------------------------
                # candidate_max_logits = probs.amax(dim=1)  # [n_candidate]
                
                # # 获取每个候选样本对应的阈值
                # candidate_thd = torch.where(
                #     class_thd[k] > -float('inf'),  # 检查该类别是否有已知样本
                #     class_thd[k],                  # 使用类特定阈值
                #     global_thd                     # 退回到全局阈值
                # )
                
                candidate_max_logits = probs.amax(dim=1)
                max_indices = torch.argmax(probs, dim=1)  # [n_candidate]
                # 生成置信掩码（最大概率出现在0类或k类）
                conf_mask = (max_indices == 0) | (max_indices == k) #& (candidate_max_logits > 0.5)
                # conf_mask = (p0 + pk) > threshold
                
                valid_pseudo = conf_mask.sum().item()
                losses['valid_pseudo_ratio'] = valid_pseudo / len(conf_mask)
                
                if valid_pseudo > 0:
                    # 生成伪标签（不修改原始labels）
                    # pseudo_labels = torch.where(
                    #     p0[conf_mask] > pk[conf_mask],
                    #     torch.zeros_like(k[conf_mask]),
                    #     k[conf_mask]
                    # ).detach()
                    pseudo_labels = max_indices[conf_mask].detach()
                    
                    # 计算伪标签CE损失（使用新生成的伪标签）
                    if balance:
                        losses['pseudo_loss'] = balanced_ce_loss(logits[candidate_mask][conf_mask], pseudo_labels.long()) #
                    else:
                        losses['pseudo_loss'] = F.cross_entropy(logits[candidate_mask][conf_mask], pseudo_labels.long())
                    # balanced_ce_loss(logits[known_mask], labels[known_mask].long())

                    return_labels = copy.deepcopy(labels)
                    candidate_indices = torch.nonzero(candidate_mask).flatten()  # 返回 [0, 2, 4]
                    final_indices = candidate_indices[conf_mask]  # 返回 [0, 4] (A的第0和第4个元素)
                    return_labels[final_indices] = pseudo_labels
                    losses['return_labels'] = return_labels
                    
                    # aa = sum(return_labels != labels)
                    
    # 组合总损失（可自定义加权）
    losses['loss'] = (
        weights[0]*losses['labeled_loss'] + 
        weights[1] * losses['pseudo_loss'] + 
        weights[2] * losses['candidate_loss']
    )
    
    return losses

def calpatch_loss(logits, labels, unknown_weight = None):
    # 分离已知和未知样本
    known_mask = labels >= 0
    unknown_mask = ~known_mask
    
    # 初始化损失
    ce_loss = 0.0
    candidate_loss = 0.0
    
    # 处理已知样本
    if torch.any(known_mask):
        ce_loss = F.cross_entropy(logits[known_mask], labels[known_mask])
    
    # 处理未知样本（标签为负数）
    if torch.any(unknown_mask):
        unknown_logits = logits[unknown_mask]
        unknown_labels = labels[unknown_mask]
        
        # 转换负数标签为候选类别索引（0和绝对值对应类别）
        abs_labels = (-unknown_labels).long()  # 转换为正数索引
        
        # 计算概率
        probs = F.softmax(unknown_logits, dim=1)
        
        # 获取两个候选类别的概率
        p0 = probs[:, 0]                          # 0类概率
        pk = probs.gather(1, abs_labels.view(-1,1)).squeeze()  # 绝对值对应类概率
        
        # 候选类别概率和（注意数值稳定性）
        p_sum = p0 + pk + 1e-8
        
        # 候选损失 = -log(p0 + pk)
        candidate_loss = -torch.log(p_sum).mean()
    
    # 组合损失（可加权重）
    if unknown_weight is None:
        lambda_coeff = unknown_mask.float().mean() 
    else:
        lambda_coeff = unknown_weight
        
    total_loss = ce_loss + lambda_coeff*candidate_loss
    
    return total_loss