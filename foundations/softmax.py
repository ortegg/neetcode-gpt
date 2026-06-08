import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z_stable = z - np.max(z)
        exp_z = np.exp(z_stable)
        probs = exp_z / np.sum(exp_z)
        return np.round(probs, 4)