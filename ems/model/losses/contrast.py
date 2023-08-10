import torch
import torch.nn.functional as F

class ContrastLoss:
    def __init__(self):
        pass

    def __call__(self, q, p):
        
        q = F.normalize(q, p=2, dim=1)
        p = F.normalize(p, p=2, dim=1)
        cos = F.cosine_similarity(p,q,dim=1)
        return (1-cos).mean()

    def __repr__(self):
        return "ContrastLoss()"

