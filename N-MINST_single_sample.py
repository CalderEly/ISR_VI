import torch

import tonic
from tonic.transforms import ToFrame

from Layer import LiFLayer

decay = 0.9
threshold = 1.0

sensor_size = tonic.datasets.NMNIST.sensor_size   # (34, 34, 2)
T = 15

frame_transform = ToFrame(sensor_size=sensor_size, n_time_bins=T)

dataset = tonic.datasets.NMNIST(save_to='/Users/calderely/ISR_VI/data', train=True,
                                transform=frame_transform)
frames, label = dataset[0]
frames = torch.tensor(frames, dtype=torch.float32)   # numpy -> tensor
x = frames.reshape(T, -1)

hidden_layer = LiFLayer(threshold, decay, num_in=2312, num_neurons=1000)
output_layer = LiFLayer(threshold, decay, 1000, 10)

goal = torch.tensor([1])  # Target output for the network

optimizer = torch.optim.Adam([hidden_layer.W, output_layer.W], lr=0.01)  # Optimizer for the weights of both layers

for epoch in range(100):

    U1 = torch.zeros(1, 1000)  # Initialize the potential tensor
    S1 = torch.zeros(1, 1000)  # Initialize the spike tensor

    U2 = torch.zeros(1, 10)  # Initialize the potential tensor
    S2 = torch.zeros(1, 10)  # Initialize the spike tensor

    rec_mem = []
    rec_spks = []

    for i in range(10):  # Simulate for 10 time steps
        U1, S1 = hidden_layer.forward(U1, S1, x)
        U2, S2 = output_layer.forward(U2, S2, S1)  # Use the spikes from layer1 as input to layer2

        rec_spks.append(S2)  # Store the output spikes from layer2
        rec_mem.append(U2)  # Store the potential from layer2

    rec_mem = torch.stack(rec_mem)  # Stack the output spikes over time
    loss = torch.nn.functional.cross_entropy(rec_mem.sum(0), goal)  # Compute the loss based on membrane potential

    optimizer.zero_grad()  # Reset gradients
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update the weights

    print(f"loss at epoch {epoch}: {loss.item()}")
    print(f"Dead neurons in hidden layer: {(S1.sum(0) == 0).sum().item()} out of {S1.shape[1]}")

spk_out = torch.stack(rec_spks)  # Stack the output spikes over time
print("recorded spike train shape:", spk_out.shape)

spike_counts = spk_out.sum(0)     # [1, 3] — total spikes per class over the run
print("spike counts per class:", spike_counts)
prediction = spike_counts.argmax(1)   # which class fired most
print("predicted class:", prediction)