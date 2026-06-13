import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        x = np.array(x)                             # (in_dim,)
        W1 = np.array(W1)                           # (hid_dim,in_dim)
        W2 = np.array(W2)                           # (out_dim,hid_dim)
        b1 = np.array(b1)                           # (hid_dim,)
        b2 = np.array(b2)                           # (out_dim,)
        y_true = np.array(y_true)                   # (out_dim,)

        z1 = x @ W1.T + b1                          # (hid_dim,)
        a1 = np.maximum(z1, 0)                      # (hid_dim,)
        y_pre = a1 @ W2.T + b2                      # (out_dim,)

        loss = np.mean((y_pre-y_true)**2)

        dL_dy = 2 * (y_pre-y_true) / y_pre.shape[0] # (out_dim,)
        dL_db2 = dL_dy                              # (out_dim,)
        dL_dW2 = dL_dy.reshape(-1,1) @ a1.reshape(1,-1) # (outdim, hid_dim)
        dL_da1 = W2.T @ dL_dy                       # (hid_dim,)

        dz1 = dL_da1 * (z1 > 0).astype(float)       # (hid_dim,)
        db1 = dz1                                   # (hid_dim,)
        dW1 = dz1.reshape(-1,1) @ x.reshape(1,-1)

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dL_dW2, 4).tolist(),
            'db2': np.round(dL_db2, 4).tolist(),
        }

        

        



