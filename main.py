import torch

from Layer import LiFLayer

decay = 0.9
threshold = 1.0

x = torch.tensor([1., 0., 1., 0.])

layer1 = LiFLayer(threshold, decay, num_in=4, num_neurons=5)
layer2 = LiFLayer(threshold, decay, 5, 3)

goal = torch.tensor([1])  # Target output for the network

optimizer = torch.optim.Adam([layer1.W, layer2.W], lr=0.01)  # Optimizer for the weights of both layers

for epoch in range(100):

    U1 = torch.zeros(1, 5)  # Initialize the potential tensor
    S1 = torch.zeros(1, 5)  # Initialize the spike tensor

    U2 = torch.zeros(1, 3)  # Initialize the potential tensor
    S2 = torch.zeros(1, 3)  # Initialize the spike tensor

    rec_mem = []
    rec_spks = []

    for i in range(10):  # Simulate for 10 time steps
        U1, S1 = layer1.forward(U1, S1, x)
        U2, S2 = layer2.forward(U2, S2, S1)  # Use the spikes from layer1 as input to layer2

        rec_spks.append(S2)  # Store the output spikes from layer2
        rec_mem.append(U2)  # Store the potential from layer2

    rec_mem = torch.stack(rec_mem)  # Stack the output spikes over time
    loss = torch.nn.functional.cross_entropy(rec_mem.sum(0), goal)  # Compute the loss based on membrane potential

    optimizer.zero_grad()  # Reset gradients
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update the weights

    print(f"step {i}: U = {U2}, S = {S2}")

spk_out = torch.stack(rec_spks)  # Stack the output spikes over time
print("recorded spike train shape:", spk_out.shape)

spike_counts = spk_out.sum(0)     # [1, 3] — total spikes per class over the run
print("spike counts per class:", spike_counts)
prediction = spike_counts.argmax(1)   # which class fired most
print("predicted class:", prediction)