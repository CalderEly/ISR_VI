#Uses torch autoencoder for backprop. This will be replaced later, so no need to understand fully. 
#Main thing is that cannot use true derivative of spike function, so we use surrogate gradient.
#Derivative of spike is 0 becuase spikes are inherently 0 most of the time.

import math

import torch

class SpikeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        alpha = 2.0               # surrogate steepness (the forgiving parameter)
        surrogate = alpha / (2 * (1 + (math.pi / 2 * alpha * v) ** 2))
        return grad_output * surrogate