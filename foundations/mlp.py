import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        cur_act = x.copy()
        layers = len(weights)

        for i in range(layers):
            W = weights[i]
            b = biases[i]

            z = np.dot(cur_act, W) + b
            if i < (layers - 1):
                cur_act = np.maximum(0, z)
            else:
                cur_act = z

        return np.round(cur_act, 5)