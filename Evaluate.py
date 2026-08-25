import torch

def evaluate(hidden_layer, output_layer, loader, T = 10):
    total_correct = 0
    total_samples = 0
    fr_sum = 0.0
    dead_counts = torch.zeros(hidden_layer.get_num_neurons())   # accumulate per-neuron spikes across all test data

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.float().permute(1, 0, 2, 3, 4).reshape(T, -1, 2312)
            labels = labels.long()
            batch = frames.shape[1]

            U1 = torch.zeros(batch, hidden_layer.get_num_neurons()); S1 = torch.zeros(batch, hidden_layer.get_num_neurons())
            U2 = torch.zeros(batch, output_layer.get_num_neurons()); S2 = torch.zeros(batch, output_layer.get_num_neurons())

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