from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        # pass
        res = []
        for num in numbers:
            tokens = self.greedy_tokenization(str(num), vocab)
            res.append(tokens)
        return res 


    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        # pass
        tokens = self.greedy_tokenization(text, vocab)
        return len(tokens)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        # pass
        tokens = self.greedy_tokenization(text, vocab)
        words = text.split()
        return round(len(tokens) / len(words), 4)

    def greedy_tokenization(self, text: str, vocab: Dict[str, int]):
        tokens = []
        i=0
        while i < len(text):
            for length in range(len(text)-i,0,-1):
                if text[i:i+length] in vocab:
                    break
            tokens.append(text[i:i+length])
            i = i + length
        return tokens



