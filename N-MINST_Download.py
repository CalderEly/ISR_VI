import tonic
import tonic.transforms as transforms
import torch

# download=True fetches it the first time; sensor_size is fixed for N-MNIST
dataset = tonic.datasets.NMNIST(save_to='/Users/calderely/ISR_VI/data', train=True)

events, label = dataset[0]     # first sample
print("label:", label)
print("events type:", type(events))
print("events shape:", events.shape)
print("first few events:", events[:5])

sensor_size = tonic.datasets.NMNIST.sensor_size   # (34, 34, 2)
T = 15

frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T)

dataset = tonic.datasets.NMNIST(save_to='/Users/calderely/ISR_VI/data', train=True,
                                transform=frame_transform)

frames, label = dataset[0]
print("frames type:", type(frames))
print("frames shape:", frames.shape)    # expect [T, 2, 34, 34] = [15, 2, 34, 34]

frames = torch.tensor(frames, dtype=torch.float32)   # numpy -> tensor
frames_flat = frames.reshape(T, -1)                   # [15, 2312]
print("flattened shape:", frames_flat.shape)          # [15, 2312]
print("2*34*34 =", 2*34*34)                           # 2312, confirms the flatten