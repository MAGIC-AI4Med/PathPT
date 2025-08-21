import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def add_label_noise(labels, noise_ratio=0.1, num_classes=None):
    """
    Add random noise to labels
    Args:
        labels: Original label tensor (any shape)
        noise_ratio: Noise ratio (default 10%)
        num_classes: Total number of classes (if not provided, automatically inferred from labels)
    Returns:
        Label tensor with added noise
    """
    if num_classes is None:
        num_classes = len(torch.unique(labels))  # Automatically infer number of classes
    
    # 1. Generate random noise positions (10% of indices)
    noise_mask = torch.rand_like(labels.float()) < noise_ratio
    
    # 2. Randomly generate new labels (excluding original labels themselves)
    random_labels = torch.randint(0, num_classes, labels.shape, device=labels.device)
    
    # Ensure new labels are different from original labels (optional)
    while True:
        same_pos = (random_labels == labels) & noise_mask
        if not same_pos.any():
            break
        random_labels[same_pos] = torch.randint(0, num_classes, (same_pos.sum(),), device=labels.device)
    
    # 3. Replace labels
    noisy_labels = labels.clone()
    noisy_labels[noise_mask] = random_labels[noise_mask]
    
    return noisy_labels

def balanced_ce_loss(logits, labels):
    """
    First calculate the average loss for each class, then take the average across all classes
    
    Parameters:
        logits: [batch_size, num_classes]
        labels: [batch_size]
        num_classes: Total number of classes
        
    Returns:
        loss: Final loss averaged by class
        per_class_loss: Average loss for each class (shape [num_classes])
    """
    # Calculate CE loss for each sample (without aggregation)
    num_classes = logits.shape[-1]
    if len(labels.shape) > 1:
        logits = logits.view(-1, logits.size(-1))  # [batch*patches, C]
        labels = labels.view(-1)
    
    # labels = add_label_noise(labels, noise_ratio=0.1, num_classes=num_classes)
    individual_losses = F.cross_entropy(logits, labels, reduction='none')  # [batch_size]
    
    # Initialize per-class statistics
    per_class_loss = torch.zeros(num_classes, device=logits.device)
    per_class_count = torch.zeros(num_classes, device=logits.device)
    
    # Iterate through each class
    for cls in range(num_classes):
        mask = (labels == cls)
        if mask.any():
            per_class_loss[cls] = individual_losses[mask].mean()  # Average loss for this class
            per_class_count[cls] = 1  # Count for later averaging
    
    # Calculate average loss for valid classes
    valid_classes = (per_class_count > 0)
    loss = per_class_loss[valid_classes].mean()  # Average across class dimension
    
    return loss

def compute_min_correct_logits(logits: torch.Tensor, 
                              labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate the minimum logits value for correctly predicted samples of each class
    
    Parameters:
        logits: Model output [batch_size, num_classes]
        labels: True labels [batch_size]
    
    Returns:
        min_logits: Minimum correct logits for each class [num_classes]
                   (classes without samples are set to NaN)
    """
    # Filter known label samples
    num_classes = logits.shape[1]
    known_mask = (labels >= 0) & (labels < num_classes)
    known_logits = logits[known_mask]
    known_labels = labels[known_mask].long()

    # Initialize result tensor
    min_logits = torch.full((num_classes,), float('nan'), 
                           device=logits.device)

    # Get predicted classes
    pred_classes = torch.argmax(known_logits, dim=1)

    # Iterate through each class
    for cls in range(num_classes):
        # Filter samples of this class that are predicted correctly
        cls_mask = (known_labels == cls) & (pred_classes == cls)
        if cls_mask.any():
            # Extract logits values for correct samples of corresponding class
            cls_logits = known_logits[cls_mask, cls]
            min_logits[cls] = torch.min(cls_logits)

    return min_logits


def PatchSSLoss(logits, labels, epoch, total_epoch=100, weights = [1,0.5,0.1], balance = False, vision_only = False, pseudo_loss = True):
    """
    Parameter description:
    logits: Model output [batch_size, num_classes]
    labels: Original labels, known labels >=0, unknown labels are negative (representing class 0 and abs(label))
    epoch: Current training epoch
    total_epoch: Total training epochs
    
    Returns:
    dict: {
        'loss': Total loss,
        'labeled_loss': Known label CE loss,
        'pseudo_loss': Pseudo label CE loss,
        'candidate_loss': Candidate sample loss,
        'valid_pseudo_ratio': Valid pseudo label ratio
    }
    """
    
    # Dynamic threshold calculation
    top_thd = 2./3.
    threshold = 1.1 if epoch < 10 else top_thd - 0.2 * ((epoch-10)/max(1,total_epoch-10))**0.5
    
    # Initialize losses and statistics
    losses = {
        'labeled_loss': torch.tensor(0.0, device=logits.device),
        'pseudo_loss': torch.tensor(0.0, device=logits.device),
        'candidate_loss': torch.tensor(0.0, device=logits.device),
        'valid_pseudo_ratio': 0.0
    }
    
    logits = logits.view(-1, logits.size(-1))  # [batch*patches, C]
    labels = labels.view(-1)    
    
    # First stage: Process known label samples
    known_mask = labels >= 0
    if known_mask.any():
        
        if balance:
            losses['labeled_loss'] = balanced_ce_loss(logits[known_mask], labels[known_mask].long()) #
        else:
            losses['labeled_loss'] = F.cross_entropy(logits[known_mask], labels[known_mask].long())
        #balanced_ce_loss(logits[known_mask], labels[known_mask].long())
        
    
    # Second stage: Process candidate samples
    if not vision_only:
        candidate_mask = ~known_mask
        if candidate_mask.any():
            # Get candidate class indices
            k = (-labels[candidate_mask]).long()
            probs = F.softmax(logits[candidate_mask], dim=1)
            p0, pk = probs[:,0], probs[torch.arange(len(k)), k]
            
            # Basic loss for candidate samples (always calculated)
            losses['candidate_loss'] = -torch.log(p0 + pk + 1e-8).mean()
            
            # Dynamically generate pseudo labels (epoch>=10)
            if epoch >= 10 and pseudo_loss:
                
                # known_labels = labels[known_mask]
                # if known_labels.numel() > 0:
                #     # Extract maximum logits values for known samples
                #     class_thd = compute_min_correct_logits(logits[known_mask], labels[known_mask])
                #     # known_max = logits[known_mask].gather(1, known_labels.view(-1,1)).squeeze(1)  # Take logit value for corresponding true class
                    
                #     # Calculate global default threshold (average of all known classes)
                #     global_thd = 0.0
                # else:
                #     # Use fixed threshold when no known samples
                #     class_thd = torch.zeros(logits.shape[1], device=logits.device)
                #     global_thd = 0.0

                # # Stage 2: Assign corresponding thresholds for candidate samples -------------------------------
                # candidate_max_logits = probs.amax(dim=1)  # [n_candidate]
                
                # # Get threshold for each candidate sample
                # candidate_thd = torch.where(
                #     class_thd[k] > -float('inf'),  # Check if this class has known samples
                #     class_thd[k],                  # Use class-specific threshold
                #     global_thd                     # Fall back to global threshold
                # )
                
                candidate_max_logits = probs.amax(dim=1)
                max_indices = torch.argmax(probs, dim=1)  # [n_candidate]
                # Generate confidence mask (maximum probability appears in class 0 or class k)
                conf_mask = (max_indices == 0) | (max_indices == k) #& (candidate_max_logits > 0.5)
                # conf_mask = (p0 + pk) > threshold
                
                valid_pseudo = conf_mask.sum().item()
                losses['valid_pseudo_ratio'] = valid_pseudo / len(conf_mask)
                
                if valid_pseudo > 0:
                    # Generate pseudo labels (do not modify original labels)
                    # pseudo_labels = torch.where(
                    #     p0[conf_mask] > pk[conf_mask],
                    #     torch.zeros_like(k[conf_mask]),
                    #     k[conf_mask]
                    # ).detach()
                    pseudo_labels = max_indices[conf_mask].detach()
                    
                    # Calculate pseudo label CE loss (using newly generated pseudo labels)
                    if balance:
                        losses['pseudo_loss'] = balanced_ce_loss(logits[candidate_mask][conf_mask], pseudo_labels.long()) #
                    else:
                        losses['pseudo_loss'] = F.cross_entropy(logits[candidate_mask][conf_mask], pseudo_labels.long())
                    # balanced_ce_loss(logits[known_mask], labels[known_mask].long())

                    return_labels = copy.deepcopy(labels)
                    candidate_indices = torch.nonzero(candidate_mask).flatten()  # Returns [0, 2, 4]
                    final_indices = candidate_indices[conf_mask]  # Returns [0, 4] (0th and 4th elements of A)
                    return_labels[final_indices] = pseudo_labels
                    losses['return_labels'] = return_labels
                    
                    # aa = sum(return_labels != labels)
                    
    # Combine total loss (customizable weighting)
    losses['loss'] = (
        weights[0]*losses['labeled_loss'] + 
        weights[1] * losses['pseudo_loss'] + 
        weights[2] * losses['candidate_loss']
    )
    
    return losses

def calpatch_loss(logits, labels, unknown_weight = None):
    # Separate known and unknown samples
    known_mask = labels >= 0
    unknown_mask = ~known_mask
    
    # Initialize losses
    ce_loss = 0.0
    candidate_loss = 0.0
    
    # Process known samples
    if torch.any(known_mask):
        ce_loss = F.cross_entropy(logits[known_mask], labels[known_mask])
    
    # Process unknown samples (labels are negative)
    if torch.any(unknown_mask):
        unknown_logits = logits[unknown_mask]
        unknown_labels = labels[unknown_mask]
        
        # Convert negative labels to candidate class indices (0 and absolute value corresponding class)
        abs_labels = (-unknown_labels).long()  # Convert to positive indices
        
        # Calculate probabilities
        probs = F.softmax(unknown_logits, dim=1)
        
        # Get probabilities for two candidate classes
        p0 = probs[:, 0]                          # Class 0 probability
        pk = probs.gather(1, abs_labels.view(-1,1)).squeeze()  # Absolute value corresponding class probability
        
        # Sum of candidate class probabilities (note numerical stability)
        p_sum = p0 + pk + 1e-8
        
        # Candidate loss = -log(p0 + pk)
        candidate_loss = -torch.log(p_sum).mean()
    
    # Combine losses (with optional weighting)
    if unknown_weight is None:
        lambda_coeff = unknown_mask.float().mean() 
    else:
        lambda_coeff = unknown_weight
        
    total_loss = ce_loss + lambda_coeff*candidate_loss
    
    return total_loss