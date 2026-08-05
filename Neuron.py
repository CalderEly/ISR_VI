class Neuron:
    def __init__(self, threshold, base_potential, decay_rate):
        self.threshold = threshold
        self.base_potential = base_potential
        self.decay_rate = decay_rate
        self.potential = base_potential
        self.last_spike = 0

    def calc_membrane(self, input_current):
        self.potential = self.decay_rate*self.potential + input_current - self.last_spike*self.threshold
        if self.potential >= self.threshold:
            self.last_spike = 1
        else:
            self.last_spike = 0
        return self.potential, self.last_spike


neuron = Neuron(threshold=1.0, base_potential=0.0, decay_rate=0.9)
for i in range(10):
    input_current = 0.5  # Example input current
    potential, spike = neuron.calc_membrane(input_current)
    print(f"Time step {i}: Potential = {potential}, Spike = {spike}")