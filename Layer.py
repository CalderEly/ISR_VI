import torch

from SpikeFunc import SpikeFunc

class LiFLayer:
    def __init__(self, threshold, decay, num_in, num_neurons):
        self.threshold = threshold
        self.decay = decay
        self.W = (torch.randn(num_neurons, num_in) * 0.5).requires_grad_(True)

    def forward(self, U, S, x):
        
        current = x @ self.W.T  # Compute the current input to the neurons

        U = self.decay * U + current - S * self.threshold

        S =  SpikeFunc.apply(U - self.threshold)  # Compute the spikes based on the updated potential

        return U, S