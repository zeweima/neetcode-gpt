class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        x = init

        for n_iter in range(iterations):
            x -= learning_rate *self.gradient(x)
        
        return round(x, 5)

    def gradient(self, x):
        return 2*x

