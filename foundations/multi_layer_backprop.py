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
        N = len(y_true)
        
        z1 = np.dot(W1, x) + b1
        a1 = np.maximum(0, z1)

        preds = np.dot(W2, a1) + b2

        # mse
        error = preds - y_true
        loss = np.mean(error ** 2)

        # --- Backward Pass ---
        # Gradient w.r.t loss out of final layer
        d_preds = (2 / N) * error
        
        # Gradients for Layer 2 (Linear)
        db2 = d_preds
        dW2 = np.outer(d_preds, a1)
        
        # Upstream gradient to Layer 1's output
        d_a1 = np.dot(d_preds, W2)
        
        # Derivative of ReLU
        d_z1 = d_a1.copy()
        d_z1[z1 <= 0] = 0
        
        # Gradients for Layer 1 (Linear)
        db1 = d_z1
        dW1 = np.outer(d_z1, x)

        return {
            'loss': float(np.round(loss, 4)),
            'dW1': dW1.round(4).tolist(),
            'db1': db1.round(4).tolist(),
            'dW2': dW2.round(4).tolist(),
            'db2': db2.round(4).tolist()
        }