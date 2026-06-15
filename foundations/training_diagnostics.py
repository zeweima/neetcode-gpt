import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        res = []
        with torch.no_grad():
            for layer in model.children():
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    mean = x.mean().item()
                    std = x.std().item()
                    if x.dim()>=2:
                        dead_frac = (x <= 0).all(dim=0).float().mean().item()
                    else:
                        dead_frac = (x <= 0).float().mean().item()
                    res.append({
                        'mean': round(mean, 4),
                        'std': round(std, 4),
                        'dead_fraction': round(dead_frac, 4)
                    })
        return res


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        y_pred = model(x)
        loss = nn.MSELoss()(y_pred, y)
        loss.backward()
        res = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                mean = grad.mean().item()
                std = grad.std().item()
                norm = torch.norm(grad).item()

                res.append({
                    'mean': round(mean, 4),
                    'std': round(std, 4),
                    'norm': round(norm, 4)
                })
        return res

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for stat in activation_stats:
            if stat['dead_fraction'] > 0.5: return 'dead_neurons'
        
        for stat in gradient_stats:
            if stat['norm'] > 1000: return 'exploding_gradients'
            
        if gradient_stats[-1]['norm'] < 1e-5: return 'vanishing_gradients'

        for stat in activation_stats:
            if stat['std'] < 0.1: return 'vanishing_gradients'
            if stat['std'] > 10.0: return 'exploding_gradients'
            
        return 'healthy'