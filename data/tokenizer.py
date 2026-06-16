from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        # pass
        merge_rule = []
        tokens = list(corpus)
        for _ in range(num_merges):
            if len(tokens)<2:
                break
            pair_count = {}

            for pair in zip(tokens[:-1],tokens[1:]):
                # print(pair)
                if pair in pair_count:
                    pair_count[pair]+=1
                else:
                    pair_count[pair]=1

            best_pair = self.get_max(pair_count)

            merge_rule.append(list(best_pair))

            new_tokens = [] 
            i=0
            while i< len(tokens):
                if i < len(tokens)-1 and (tokens[i], tokens[i+1])== best_pair:
                    new_tokens.append(tokens[i]+tokens[i+1])
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            
            tokens = new_tokens
            print(tokens)
        return merge_rule
            
    def get_max(self, pair_count):
        if not pair_count: return None

        max_item = 'PLACEHOLD' 
        max_frequency = 0

        for key, value in pair_count.items():
            if value > max_frequency:
                 max_item = key
                 max_frequency = value
            elif value == max_frequency:
                if key < max_item:
                    max_item = key
        return max_item



"""
The detailed process of BPR:
1. convert words to characters
2. count the frequency of neighbors
3. find the pair with highest frequency
4. merge this pair into a new tokne
5. Repeatly doing 2-4 until meet the vocab size
"""

