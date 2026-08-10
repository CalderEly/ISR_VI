import torch

import Layer

'''
    This is purely an experimental idea to test the time complexity of chains v.s. layers
        - Based off of the adaptability aspect of each neuron connects to another but I cannot imagine that they work as
            "layers"
        - No internet used in dev. so completely unoptimized, and there might be papers already tested this
        - Could provide further adaptability
'''

class Chain:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.check_validity(layer)

        self.layers.append(layer)


    #Checks the validity of the layer addition by running a fast test
    #Will find math errors at very least(neuron # mismatches etc.)
    def check_validity(self, layer) -> bool:

        len = len(self.layers)

        orig_layer = self.layers[len - 1]
        new_layer = layer
        
        goal = torch.tensor([1])  # Target output for the network
        
        optimizer = torch.optim.Adam([orig_layer.W, new_layer.W], lr=0.01)  # Optimizer for the weights of both layers
        
        try:
            for epoch in range(100):

                U1 = torch.zeros(1, 1000)  # Initialize the potential tensor
                S1 = torch.zeros(1, 1000)  # Initialize the spike tensor

                U2 = torch.zeros(1, 10)  # Initialize the potential tensor
                S2 = torch.zeros(1, 10)  # Initialize the spike tensor

                rec_mem = []
                rec_spks = []

                for i in range(10):  # Simulate for 10 time steps
                    U1, S1 = orig_layer.forward(U1, S1, x)
                    U2, S2 = new_layer.forward(U2, S2, S1)  # Use the spikes from layer1 as input to layer2

                    rec_spks.append(S2)  # Store the output spikes from layer2
                    rec_mem.append(U2)  # Store the potential from layer2

                rec_mem = torch.stack(rec_mem)  # Stack the output spikes over time
                loss = torch.nn.functional.cross_entropy(rec_mem.sum(0), goal)  # Compute the loss based on membrane potential

                optimizer.zero_grad()  # Reset gradients
                loss.backward()  # Backpropagate the loss
                optimizer.step()  # Update the weights

            return True

        except:
            print("Issue with adding layer. Check layer configs")

            return False