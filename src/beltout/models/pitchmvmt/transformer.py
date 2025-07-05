import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadedSelfAttentionBlock(nn.Module):
    """
    Performs self-attention on its input vector x of size (T, d_model).

    This is a Pre-LN, dropout-after-residual Transformer block.

    Having dropout after the residual serves not only its usual purpose, but it also 
    gets the model to not rely entirely on the attention-processed pitch information to 
    output its mel frames when the input is unvoiced. While our use of the tiny CREPE model 
    which has fluctuations during unvoiced audio sections unlike the full model, can already 
    potentially provide a good source of information on the characteristics of the unvoiced 
    parts of a performance, it is still essential that the model does not lose the ability 
    to infer that information by itself from the x-vector and the S3 tokens too.
    """
    def __init__(self, d_model, n_heads, d_k, d_v, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.concat_dim = n_heads * d_v # Store concatenated dimension
        self.concat_dim_key = n_heads * d_k

        self.q_proj = nn.Linear(self.d_model, self.concat_dim_key, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.concat_dim_key, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.concat_dim, bias=False)
        self.out_proj = nn.Linear(self.concat_dim, self.d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout) # Dropout for attention weights
        self.resid_dropout = nn.Dropout(dropout) # Dropout for residual connection

        self.layer_norm = nn.LayerNorm(d_model)

        self._init_weights()
        
    def forward(self, x):
        """
        x: input vector (batch_size, T, d_model) - output from previous layer or initial embedding
        """
        batch_size, T, _ = x.shape
        
        # Store original input for residual connection
        residual = x # Use the non-normalized input for residual

        # Pre-Layer Normalization
        x = self.layer_norm(x)

        # Project queries, keys, and values for all heads at once
        q = self.q_proj(x)  # (batch_size, T, concat_dim_key)
        k = self.k_proj(x)  # (batch_size, T, concat_dim_key)
        v = self.v_proj(x)  # (batch_size, T, concat_dim)

        # Split into multiple heads
        # d_t = d_k for query and key projections, d_t = d_v for value projections.
        def split_heads(tensor, d_t):
            return tensor.view(batch_size, T, self.n_heads, d_t).transpose(1, 2)
            # Now shape: (batch_size, n_heads, T, d_t)
        q = split_heads(q, self.d_k)
        k = split_heads(k, self.d_k)
        v = split_heads(v, self.d_v)

        # Compute attention scores (scaled dot-product attention)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d_k)  # (batch_size, n_heads, T, T)?
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights) # Dropout on attention weights

        # Apply attention weights to values
        head_outputs = torch.matmul(attention_weights, v)  # (batch_size, n_heads, T, d_v)
        
        # Concatenate outputs from all heads along the last dimension (d_v)
        concat_output = head_outputs.transpose(1, 2).reshape(batch_size, T, self.concat_dim)  # (batch_size, T, concat_dim)
        
        output = self.out_proj(concat_output)  # (batch_size, T, d_model)

        # Apply dropout, residual connection
        output = residual + output
        output = self.resid_dropout(output)
        
        return output, attention_weights  # This output (batch_size, T, d_model) is passed to the next block/layer
    
    def _init_weights(self):
        # Xavier initialization (Glorot)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(proj.weight)

class FeedForward(nn.Module):
    """Standard feed-forward network in Transformer architecture"""
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)
        nn.init.zeros_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x

