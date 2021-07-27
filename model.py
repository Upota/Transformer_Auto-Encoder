import math
import logging  # runtime log

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

def t2v(tau, f, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)

class BaseConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, n_in, block_size, **kwargs):
        self.n_in = n_in
        self.block_size = block_size
        for k,v in kwargs.items():  
            setattr(self, k, v)

class TAEConfig(BaseConfig):   # Transformer Auto-Encoder
    n_layer = 2
    n_head = 4
    n_embd = 64

class MultiHeadAttention(nn.Module):
    """
    A vanilla multi-head self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is notihing too scary here.
    """

    def __init__(self, config, is_masked=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # query, key, value projections for all heads
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # ouput projection (WO)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal maks to ensure that attention is only applied to the left in the input sequence
        self.is_masked = is_masked
        if is_masked is not None:
            self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size)) # masking future positions
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size() # (Batch_size, Block_size, n_embd)
                           # n_embd // n_head = dim of Q/K/V vector in one head
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.is_masked is not None:  # for Masked Multi-head Attention
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf')) # ignore future positions
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class EncoderDecoderAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # query, key, value projections for all heads
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # ouput projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x, enc_out, layer_past=None):
        B, T, C = x.size() # (Batch_size, Block_size, n_embd)
                           # n_embd // n_head = dim of Q/K/V vector in one head
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.key(enc_out).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(enc_out).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ffnn = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
    
    def forward(self, x):
        x = self.ln1(self.attn(x) + x)
        x = self.ln2(self.ffnn(x) + x)

        return x

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.mattn = MultiHeadAttention(config, is_masked=True) # Masked multi-head attention
        self.attn = EncoderDecoderAttention(config) # Encoder-Decoder Attention
        self.ffnn = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
    
    def forward(self, x, enc_out):
        x = self.ln1(self.mattn(x) + x)
        x = self.ln2(self.attn(x, enc_out) + x)
        x = self.ln3(self.ffnn(x) + x)

        return x

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # input embedding stem
        self.time_emb = SineActivation(config.n_in, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
    
    def forward(self, x):
        time_embeddings = self.time_emb(x)                    # each value maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :x.size(1), :]            # each position maps to a (learnable) vector
        x = self.drop(time_embeddings + position_embeddings)   # broadcast for each batch (B, T, n_emb)

        return x

class TAE(nn.Module):
    """ the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()
        # embedding encoder/decoder input
        self.enc_emb = Embedding(config)
        self.dec_emb = Embedding(config)
        # encoders
        self.encoders = nn.Sequential(*[Encoder(config) for _ in range(config.n_layer)])
        # decoders
        self.decoders = nn.ModuleList([Decoder(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.n_in, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config): # train_config에 따라 원하는 최적화 설정
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm)
        for mn, m in self.named_modules(): # module_name, module
            # logger.info("mn: %s, m: %s", mn, m)
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modiles will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('w') or pn.endswith('w0') or pn.endswith('b') or pn.endswith('b0'):
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')
        # no_decay.update(['time_emb.w0', 'time_emb.b0', 'time_emb.b', 'time_emb.w'])

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, x, dec_in, targets=None):
        b, t = x.size() # (Batch_size, Block_size)
        assert t <= self.block_size, "Cannot forward, model block size is exhausted"

        # forward TAE model
        x, dec_in = x.view(b, t, -1), dec_in.view(b, t, -1)
        x, dec_in  = self.enc_emb(x), self.dec_emb(dec_in)
        enc_out = self.encoders(x)
        for layer in self.decoders:
            dec_in = layer(dec_in, enc_out)
        x = self.ln_f(dec_in)
        logits = self.head(x)   # (B, T, n_in)

        if targets is not None:
            loss = F.mse_loss(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss