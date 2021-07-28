import random
import numpy as np
from numpy.core.shape_base import block
import torch
import torch.nn as nn
from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('inf') # make the rest logits -inf
    return out

@torch.no_grad()
def sample(model, x, steps=None, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indeices in x (of shape (b, t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time, Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    # for k in range(steps):
    #     x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop contest if needed
    #     logits, _ = model(x_cond, x_cond, targets=x_cond)
    #     x = torch.cat((x, logits), dim=1)
    
    x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop contest if needed
    logits, _ = model(x_cond, x_cond, targets=x_cond)

    return logits