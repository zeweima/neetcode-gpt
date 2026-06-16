from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        sort_set = sorted(set(text))
        stoi = {}
        itos = {}
        for idx, c in enumerate(sort_set):
            stoi[c] = idx
            itos[idx] = c
        return stoi, itos

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        # pass
        return [stoi[c] for c in text]

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        # pass
        return ''.join([itos[idx] for idx in ids])
