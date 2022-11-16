
import numpy as np

class Node:
    def __init__(self, activation, name, connectome, bias, is_output):
        self.name = name
        self.connectome = connectome
        self.bias = bias
        self.weight_dict = {}
        self.is_output = is_output
        self.activation = activation

    def instantiate_weighting(self):
        if self.is_output:
            print("Output Node, no connections made")
        else:
            for node in self.connectome:
                self.weight_dict[node] = (3*np.random.random_sample())