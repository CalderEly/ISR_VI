import Layer
import torch
import tonic
import tonic.transforms as transforms
import torch

# download=True fetches it the first time; sensor_size is fixed for N-MNIST
dataset = tonic.datasets.NMNIST(save_to='data', train=True)

events, label = dataset[0]     # first sample
print("label:", label)
print("events type:", type(events))
print("events shape:", events.shape)
print("first few events:", events[:5])

sensor_size = tonic.datasets.NMNIST.sensor_size   # (34, 34, 2)
T = 10

frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T)

dataset = tonic.datasets.NMNIST(save_to='data', train=True,
                                transform=frame_transform)

frames, label = dataset[0]
print("frames type:", type(frames))
print("frames shape:", frames.shape)    # expect [T, 2, 34, 34] = [15, 2, 34, 34]

frames = torch.tensor(frames, dtype=torch.float32)   # numpy -> tensor
frames_flat = frames.reshape(T, -1)                   # [15, 2312]
print("flattened shape:", frames_flat.shape)          # [15, 2312]
print("2*34*34 =", 2*34*34)                           # 2312, confirms the flatten

layer1 = Layer.LiFLayer(threshold=1.0, decay=0.9, num_in=2312, num_neurons=1000)
out1 = Layer.LiFLayer(threshold=1.0, decay=0.9, num_in=1000, num_neurons=10)

rec_spks = []

U1 = torch.zeros(1, 1000)  # Initialize the potential tensor
S1 = torch.zeros(1, 1000)  # Initialize the spike tensor

U2 = torch.zeros(1, 10)    # Initialize the potential tensor for the output layer
S2 = torch.zeros(1, 10)          # ADD: initialize output spike state


for i in range(10):
    U1, S1 = layer1.forward(U1, S1, frames_flat[i])  # Use the i-th frame as input
    U2, S2 = out1.forward(U2, S2, S1)  # Use the spikes from layer1 as input to out1
    hidden_rec = []  # Initialize a list to record spikes from the first layer
    hidden_rec.append(S1)  # Record spikes from the first layer

    rec_spks.append(S2)

    print(f"Timestep {i+1}:")
    print("U1 shape:", U1.shape) 
    print("S2 shape:", S2.shape)
    print("Spikes:", S2.sum())

    hidden_rec = torch.stack(hidden_rec)
    print("hidden mean firing rate:", hidden_rec.mean().item())
    print("hidden dead fraction:", (hidden_rec.sum(0) == 0).float().mean().item())

spk_rec = torch.stack(rec_spks)      # [15, 1, 10]
print("output spike train shape:", spk_rec.shape)   # want [15, 1, 10]
print("spike counts:", spk_rec.sum(0))              # [1, 10]
print("predicted class:", spk_rec.sum(0).argmax(1)) # which class fired most
print("true label:", label)