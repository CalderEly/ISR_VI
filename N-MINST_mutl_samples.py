import torch
from torch.utils.data import DataLoader

import tonic
from tonic.transforms import ToFrame

from Layer import LiFLayer

# Simulation / dataset settings
decay = 0.9
threshold = 1.0
T = 15
sensor_size = tonic.datasets.NMNIST.sensor_size  # (34, 34, 2)

batch_size = 64
num_epochs = 1

frame_transform = ToFrame(sensor_size=sensor_size, n_time_bins=T)
dataset = tonic.datasets.NMNIST(save_to='C:\\Users\\calde\\Documents\\Github\\ISR_VI\\data', train=True, transform=frame_transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

hidden_size = 1000
output_size = 10

hidden_layer = LiFLayer(threshold, decay, num_in=2312, num_neurons=hidden_size)
output_layer = LiFLayer(threshold, decay, num_in=hidden_size, num_neurons=output_size)
optimizer = torch.optim.Adam([hidden_layer.W, output_layer.W], lr=0.0003)

test_dataset = tonic.datasets.NMNIST(save_to='C:\\Users\\calde\\Documents\\Github\\ISR_VI\\data', train=False, transform=frame_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

def evaluate(hidden_layer, output_layer):
    total_correct = 0
    total_samples = 0
    fr_sum = 0.0
    dead_counts = torch.zeros(hidden_size)   # accumulate per-neuron spikes across all test data

    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.float().permute(1, 0, 2, 3, 4).reshape(T, -1, 2312)
            labels = labels.long()
            batch = frames.shape[1]

            U1 = torch.zeros(batch, hidden_size); S1 = torch.zeros(batch, hidden_size)
            U2 = torch.zeros(batch, output_size); S2 = torch.zeros(batch, output_size)

            mem_rec = []            # record OUTPUT MEMBRANE POTENTIALS
            hidden_rec = []

            for t in range(T):
                U1, S1 = hidden_layer.forward(U1, S1, frames[t])
                U2, S2 = output_layer.forward(U2, S2, S1)
                mem_rec.append(U2)
                hidden_rec.append(S1)

            mem_rec = torch.stack(mem_rec)               # [T, batch, 10]
            preds = mem_rec.sum(0).argmax(dim=1)         # spike-count decode
            total_correct += (preds == labels).sum().item()
            total_samples += batch

            hidden_rec = torch.stack(hidden_rec)         # [T, batch, 1000]
            fr_sum += hidden_rec.mean().item() * batch
            dead_counts += hidden_rec.sum(dim=(0,1))     # per-neuron total spikes across batch

    accuracy = total_correct / total_samples
    mean_fr = fr_sum / total_samples
    dead_frac = (dead_counts == 0).float().mean().item()   # dead across the ENTIRE test set
    return accuracy, mean_fr, dead_frac

if __name__ == '__main__':
    for epoch in range(num_epochs):
        for batch_idx, (frames, labels) in enumerate(loader):
            frames = frames.float().permute(1, 0, 2, 3, 4).reshape(T, -1, 2312)
            labels = labels.long()
            batch = frames.shape[1]

            U1 = torch.zeros(batch, hidden_size)
            S1 = torch.zeros(batch, hidden_size)
            U2 = torch.zeros(batch, output_size)
            S2 = torch.zeros(batch, output_size)

            hidden_rec = []
            rec_mem = []

            for t in range(T):
                U1, S1 = hidden_layer.forward(U1, S1, frames[t])
                U2, S2 = output_layer.forward(U2, S2, S1)
                hidden_rec.append(S1)
                rec_mem.append(U2)

            hidden_rec = torch.stack(hidden_rec)
            rec_mem = torch.stack(rec_mem)

            logits = rec_mem.mean(0)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted = logits.argmax(dim=1)
            acc = (predicted == labels).float().mean().item()

            mean_fr = hidden_rec.mean().item()
            dead_frac = (hidden_rec.sum(dim=(0, 1)) == 0).float().mean().item()

            if batch_idx % 5 == 0:
                print(
                    f'epoch={epoch} batch={batch_idx}/{len(loader)} '
                    f'loss={loss.item():.4f} acc={acc*100:.1f}% '
                    f'mean_fr={mean_fr:.4f} dead_frac={dead_frac:.4f}'
                )

            if batch_idx % 50 == 0:
                accuracy, mean_fr, dead_frac = evaluate(hidden_layer, output_layer)
                print(f'Validation accuracy={accuracy*100:.1f}% mean_fr={mean_fr:.4f} dead_frac={dead_frac:.4f}')

        print(f'Finished epoch {epoch} after {len(loader)} batches')

    print('Training pass complete.')

accuracy, mean_fr, dead_frac = evaluate(hidden_layer, output_layer)
print(f'Test accuracy={accuracy*100:.1f}% mean_fr={mean_fr:.4f} dead_frac={dead_frac:.4f}')