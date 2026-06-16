import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        # pass
        self.linear_layer = nn.Linear(embedding_dim, 3*attention_dim, bias=False)
        self.attention_dim = attention_dim

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        # pass
        # self.linear_layer(embedded) --> batchsize, sequence_len, 3*
        batch_size, seq_len, embedding_dim = embedded.shape

        # projection = self.linear_layer(embedded).view((batch_size,seq_len,3,-1))
        # K,Q,V = projection[:,:,0], projection[:,:,1], projection[:,:,2]
        projection = self.linear_layer(embedded)
        K,Q,V = projection.chunk(3, dim=-1)
        attention_dim = Q.shape[-1]
        scores = Q @ K.transpose(-2, -1)/(attention_dim**0.5)
        
        mask = torch.triu(torch.ones((seq_len,seq_len)), diagonal=1).bool()
        # print(mask.shape, scores.shape, embedded.shape, attention_dim)
        scores = scores.masked_fill(mask, float('-inf'))
        scores = nn.functional.softmax(scores, dim=2)
        return torch.round(scores @ V, decimals=4)
"""
torch.tril:
    Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input, 
    the other elements of the result tensor out are set to 0.
torch.triu(input, diagonal=0, *, out=None) → Tensor
    Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, 
    the other elements of the result tensor out are set to 0.
"""





