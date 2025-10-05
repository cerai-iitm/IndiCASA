import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations

class NTBXentLoss(nn.Module):
    
    def __init__(self, temperature=0.5, device='cuda'):
        """
        Normalized Temperature Binary Cross-Entropy Loss from SimCLR paper
        
        Args:
            temperature (float): Temperature parameter that controls the softmax distribution (default: 0.5)
        """
        super().__init__()
        self.temperature = temperature
        self.device = device
        
    def forward(self, stereos, astereos):

        if len(stereos) == len(astereos) == 1:
            csm = F.cosine_similarity(stereos, astereos).to(self.device)
            lossm = F.binary_cross_entropy((csm / self.temperature).sigmoid(), torch.tensor([0.0], device='cuda'), reduction="none").to(self.device)
            
            return lossm


        x = torch.cat((stereos, astereos), dim=0).to(self.device)

        indices1 = list(range(len(stereos)))
        indices2 = list(range(len(stereos), len(stereos)+len(astereos)))
        
        positive_labels = torch.cat([torch.tensor(list(combinations(indices1, 2))), torch.tensor(list(combinations(indices2, 2)))], dim=0)
        # positive_labels = torch.cat([positive_labels, torch.arange(len(x)).reshape(len(x), 1).expand(-1, 2)], dim=0)
        
        target = torch.zeros(len(x), len(x)).to(self.device)
        target[positive_labels[:,0], positive_labels[:,1]] = 1
        target[positive_labels[:,1], positive_labels[:,0]] = 1
        
        csm = F.cosine_similarity(x[None,:,:], x[:,None,:], dim=-1).to(self.device)
        scrm = csm.clone().to(self.device)
        
        ones_matrix = torch.ones((len(x), len(x))).to(self.device)
        lower_triangular = torch.tril(ones_matrix).to(self.device)
        
        scrm[lower_triangular.bool()] = float("inf")
        
        lossm = F.binary_cross_entropy((scrm / self.temperature).sigmoid(), target, reduction="none").to(self.device)
        
        target_pos = target.bool().to(self.device)
        target_neg = ~target_pos.to(self.device)
        
        loss_pos_mat = torch.zeros(x.size(0), x.size(0)).to(self.device).masked_scatter(target_pos, lossm[target_pos])
        loss_neg_mat = torch.zeros(x.size(0), x.size(0)).to(self.device).masked_scatter(target_neg, lossm[target_neg])
        
        loss_pos = loss_pos_mat.sum().sum()
        loss_neg = loss_neg_mat.sum().sum()
        
        total_pairs = (len(x)*((len(x))-1))/2
        n_pos = len(positive_labels)
        n_neg = total_pairs - n_pos

        mean_pos_loss = loss_pos / n_pos
        mean_neg_loss = loss_neg / n_neg
        
        com_loss = mean_pos_loss + mean_neg_loss
        
        return com_loss

class NTXentLoss(nn.Module):
    
    def __init__(self, temperature=0.5, device='cuda'):
        """
        Normalized Temperature Binary Cross-Entropy Loss from SimCLR paper
        
        Args:
            temperature (float): Temperature parameter that controls the softmax distribution (default: 0.5)
        """
        super().__init__()
        self.temperature = temperature
        self.device = device
        
    def forward(self, stereos, astereos):
        
        str_cossim = self.calc_cosine_similarity(stereos, stereos, device=self.device)
        astr_cossim = self.calc_cosine_similarity(astereos, astereos, device=self.device)
        csm_str_astr = F.cosine_similarity(stereos[None,:,:], astereos[:,None,:], dim=-1).to(self.device).reshape(-1)
        
        pos_sim = torch.cat([str_cossim, astr_cossim], dim=0).to(self.device)
        pos_sim_col = pos_sim.unsqueeze(1)
        
        cs_matrix = csm_str_astr.unsqueeze(0).expand(pos_sim.size(0), -1).to(self.device)
        sm_matrix = torch.cat((pos_sim_col, cs_matrix), dim=1).to(self.device)
        
        loss = F.cross_entropy(sm_matrix/self.temperature, torch.zeros(len(sm_matrix), device=self.device).long(), reduction='none').to(self.device)
        losses = loss.mean()
        
        return losses

    def calc_cosine_similarity(self, tensor1, tensor2, device='cuda'):
    
        csm = F.cosine_similarity(tensor1[None,:,:], tensor2[:,None,:], dim=-1).to(device)
        upper_tri_indices = torch.triu_indices(*csm.shape, offset=1, device=device)
        csm_flat = csm[upper_tri_indices[0], upper_tri_indices[1]].to(device)
        
        return csm_flat

class PairLoss(nn.Module):
    
    def __init__(self, threshold=0.9, device='cuda'):
        """
        Pair Loss function for contrastive learning.
        
        Args:
            threshold (float): threshold parameter that controls the softmax distribution (default: 0.5)
        """
        super().__init__()
        self.threshold = threshold
        self.device = device

    def forward(self, stereos, astereos):

        if len(stereos) == len(astereos) == 1:

            csm_loss = F.cosine_similarity(stereos, astereos).to(self.device)
            
            return csm_loss

        
        x = torch.cat((stereos, astereos), dim=0).to(self.device)
        
        indices1 = list(range(len(stereos)))
        indices2 = list(range(len(stereos), len(stereos)+len(astereos)))
        
        positive_labels = torch.cat([torch.tensor(list(combinations(indices1, 2))), torch.tensor(list(combinations(indices2, 2)))], dim=0)
        
        ones_matrix = torch.ones((len(x), len(x))).to(self.device)
        lower_triangular = torch.tril(ones_matrix).to(self.device)
        
        pos_mask = torch.zeros(len(x), len(x)).to(self.device)
        pos_mask[positive_labels[:,0], positive_labels[:,1]] = 1
        pos_mask[positive_labels[:,1], positive_labels[:,0]] = 1
        pos_mask[lower_triangular.bool()] = 0
        
        neg_mask =  1-pos_mask
        neg_mask[lower_triangular.bool()] = 0
        
        total_pairs = (len(x)*((len(x))-1))/2
        n_pos = len(positive_labels)
        n_neg = total_pairs - n_pos
        
        csm = F.cosine_similarity(x[None,:,:], x[:,None,:], dim=-1).to(self.device)
        
        pos_sim = (1.0 - csm.clone().to(self.device)) * pos_mask
        
        pos_loss = pos_sim.sum().sum() / n_pos
        
        neg_loss_agg = torch.maximum(torch.tensor(0), csm.clone().to(self.device) - self.threshold) * neg_mask
        neg_loss = neg_loss_agg.sum().sum() / n_neg
        
        loss = neg_loss + pos_loss
        
        return loss

class TripletLoss(nn.Module):
    
    def __init__(self, threshold=0.3, device='cuda'):
        """
        Normalized Temperature Binary Cross-Entropy Loss from SimCLR paper
        
        Args:
            temperature (float): Temperature parameter that controls the softmax distribution (default: 0.5)
        """
        super().__init__()
        self.threshold = threshold
        self.device = device
        
    def forward(self, stereos, astereos):
        
        str_cossim = self.calc_cosine_similarity(stereos, stereos, device=self.device)
        astr_cossim = self.calc_cosine_similarity(astereos, astereos, device=self.device)
        
        pos_sim = torch.cat([str_cossim, astr_cossim], dim=0).to(self.device)
        
        neg_sim = F.cosine_similarity(stereos[None,:,:], astereos[:,None,:], dim=-1).to(self.device).reshape(-1)
        
        grid1, grid2 = torch.meshgrid(pos_sim, neg_sim, indexing='ij')
        combs = torch.stack([grid1.flatten(), grid2.flatten()], dim=1).to(self.device)
        
        losses = torch.maximum(torch.tensor(0), combs[:, 1] - combs[:, 0] + self.threshold)
        loss = losses.mean()
        
        return loss
    
    def calc_cosine_similarity(self, tensor1, tensor2, device='cuda'):
    
        csm = F.cosine_similarity(tensor1[None,:,:], tensor2[:,None,:], dim=-1).to(device)
        upper_tri_indices = torch.triu_indices(*csm.shape, offset=1, device=device)
        csm_flat = csm[upper_tri_indices[0], upper_tri_indices[1]].to(device)
        
        return csm_flat

