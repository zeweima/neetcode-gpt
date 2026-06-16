import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(
        self, model: nn.Module, data: torch.Tensor, 
        epochs: int, context_length: int, batch_size: int, 
        lr: float
    ) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        # pass 
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            ids = torch.randint(data.shape[0]-context_length, (batch_size,))
            x = torch.stack(
                [data[idx:idx+context_length] for idx in ids],
                dim=0
            )
            y = torch.stack(
                [data[idx+1:idx+context_length+1] for idx in ids],
                dim=0
            )

            logits = model(x)
            B,T,C = logits.shape
            # To use cross entropy loss, the observation should be a one d array
            #   The logits should be a two d array, the first dimension is the samples,
            #    and the second dimension is the log logit for each class. 
            loss = loss_fn(logits.view(-1,C),y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return round(loss.item(), 4)


