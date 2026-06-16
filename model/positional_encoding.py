import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        # pass
        PE = np.zeros((seq_len, d_model))

        positions = np.arange(seq_len).reshape((-1,1))
        div = np.exp(
            -np.arange(0,d_model,2)/d_model * np.log(1e4)
        )

        PE[:,::2] = np.sin(positions*div)
        PE[:,1::2] = np.cos(positions*div)
        return np.round(PE, 5)

        """
        Note: 
            1. The dimensions of the position encoder is the same as the embedings.
                The first dimension is the length of the sequence, and the second one
                is the embedding dimension. 
            2. The use of broadcast.
                The first step of broadcast is the align the dimension, basically adding
                1 to the left of the array. Then, check whether each dimension can be extended
            3. The, the batch version?
                only calcualte one, then using broadcast also,
        """
