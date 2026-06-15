import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        # pass
        all_words = positive+negative

        word_set = sorted({word for sentence in all_words for word in sentence.split()})
        sorted_set = sorted(word_set)
        word_dict = {word: idx+1 for idx, word in enumerate(sorted_set)}

        ids = [torch.tensor([word_dict[word] for word in string.split()]) for string in all_words]

        from torch.nn.utils.rnn import pad_sequence
        res = pad_sequence(ids, batch_first=True)
        return res



"""
from torch.nn.utils.rnn import pad_sequence
a = torch.ones(25, 300)
b = torch.ones(22, 300)
c = torch.ones(15, 300)
pad_sequence([a, b, c]).size()

>> Out size: torch.Size([25, 3, 300])

`pad_sequence` stacks a list of Tensors along a new dimension, and 
pads them to equal length. sequences can be list of sequences with 
size L x *, where L is length of the sequence and * is any number of 
dimensions (including 0). If batch_first is False, the output is of 
size T x B x *, and B x T x * otherwise, where B is the batch size 
(the number of elements in sequences), T is the length of the longest 
sequence.
"""
