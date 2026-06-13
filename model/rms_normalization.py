import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        # pass
        x = np.asarray(x)
        gamma = np.asarray(gamma)

        MS = np.mean(x**2)+eps
        x_hat = x/np.sqrt(MS)
        return np.round(x_hat*gamma, 4).tolist()
