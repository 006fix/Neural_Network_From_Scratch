
import numpy as np
import Function_Library.Activation_Functions as Act_Func

class Node:
    def __init__(self, activation, name, connectome, bias, is_output):
        self.name = name
        self.connectome = connectome
        self.bias = bias
        self.weight_dict = {}
        self.is_output = is_output
        self.activation = activation
        self.forward_nodes = []
        self.backward_nodes = []

    def instantiate_weighting(self):
        if self.is_output:
            pass
            #below is not needed for now
            #print("Output Node, no connections made")
        else:
            for node in self.connectome:
                print(f"self connectome is {self.connectome}")
                print(node)
                self.weight_dict[node] = (3*np.random.random_sample())

    def apply_applicable_activation(self, input_val):
        if self.activation == 'sigmoid':
            return Act_Func.sigmoid(input_val)
        elif self.activation == 'relu':
            return Act_Func.relu(input_val)
        elif self.activation == 'tanh':
            return Act_Func.tanh(input_val)
        elif self.activation == 'leaky_relu':
            return Act_Func.leaky_relu(input_val)
        elif self.activation == 'linear':
            return input_val
        else:
            raise ValueError(f"""I'm afraid I was unable to find a suitale activation function for 
                Node {self.name}, activation function provided was {self.activation}""")
