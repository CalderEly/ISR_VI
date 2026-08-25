import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader

sensor_size = tonic.datasets.NMNIST.sensor_size  # (34, 34, 2)
T = 10

frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T)

dataset = tonic.datasets.NMNIST(save_to='data', train=True,
                                transform=frame_transform)


loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

batch = next(iter(loader))
frames, labels = batch

print('batch frames shape:', frames.shape)
print('batch labels shape:', labels.shape)
print('batch labels:', labels)

frames = frames.float()
frames = frames.permute(1, 0, 2, 3, 4)              # [T, batch, 2, 34, 34]
frames = frames.reshape(T, frames.shape[1], -1)     # [T, batch, 2312]
print('reshaped:', frames.shape)                     # [10, 4, 2312]

