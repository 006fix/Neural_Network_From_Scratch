#input criteria for the neural network:

    #base data (train/test)
    #data labels (train/test)
    #layers structure (i.e [20,10,5] = 3 hidden layers, of size 20,10,5
        #internally within the NN, these will be stored as a dictionary of named nodes
        #each node will be its own class structure item, containing weights and biases
        #naming convention = node_{layer_num}_{node_num}
            #layer and node num will be 0 indexed
    #connectome structure - 2 variations
        #var1 - dense. input as string, simply generates a classic dense neural network
        #var2 - sparse. input as a dict.
            #dict key = node name (node connections go *from*)
            #dict value = list, consisting of node names (node connections go *to*)
    #activation function (string) (DO I NEED TO CHANGE THIS LATER? DIDN'T SEEM TO WORK GREAT)
    #num inputs - number of inputs
    #num outputs - number of output variables


#

import Classes.Neural_Node as Node
import Function_Library.Preprocessing_Functions as Proc_Func

class Neural_Network:
    def __init__(self, base_data_train, base_data_test, base_label_train, base_label_test,
                 layer_structure, connectome_structure, activation_function, num_inputs, num_outputs,
                 output_function):
        #leaving these commented out for now until I know the internal structure generation works
        self.norm_data_train = Proc_Func.normalize(base_data_train)
        self.norm_data_test = Proc_Func.normalize(base_data_test)
        self.label_train = base_label_train
        self.label_test = base_label_test
        self.node_dict = {}
        self.layer_structure = layer_structure
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.connectome_structure = connectome_structure
        self.activation_function = activation_function
        self.output_function = output_function
        #this curently holds layer 0 (input layer)
            #additional function will populate with existing layers
        self.layers = [0]
        #basic at input, will be overwritten later with the true value
        self.max_layer = 0

    def populate_layers(self):
        for i in range(len(self.layer_structure)):
            holdval = i+1
            self.layers.append(holdval)
        #find max val, add layer 1 above this (output layer
        holdval2 = max(self.layers)
        holdval2 += 1
        self.max_layer = holdval2
        self.layers.append(holdval2)

    #function to handle the "dense" input for connectome structure
    #where exists, manually creates a dense network in the same style as a normal input
    def connectome_validity(self):
        #only triggers if the connectome structure was given as dense
        if self.connectome_structure == 'dense':
            connectome_holder = {}
            for i in (range(len(self.layer_structure) + 1)):
                #if input row, use num_inputs
                #else, use layer_structure of layer n to compare to layer n+1
                if i == 0:
                    iterator = self.num_inputs
                else:
                    iterator = self.layer_structure[i-1]
                #if final item of the loop
                #in this case, compare to output features instead
                if i == len(self.layer_structure):
                    second_iterator = self.num_outputs
                else:
                    second_iterator = self.layer_structure[i]
                for j in range(iterator):
                    keyval = f"node_{i}_{j}"
                    holdlist = []
                    for k in range(second_iterator):
                        valval = f"node_{i+1}_{k}"
                        holdlist.append(valval)
                    connectome_holder[keyval] = holdlist
            self.connectome_structure = connectome_holder
        else:
            #otherwise, ignore
            print("Option 'dense' was not selected. Provided connectome structure will be utilised")

    def create_internal_structure(self):
        #instantialise the node dict:
        #storing every node as the number 5 for now until i have something to provide them true contents
        #instantalise input layer
        for i in range(self.num_inputs):
            keyval = f"node_0_{i}"
            hold_node = Node.Node(self.activation_function, keyval, self.connectome_structure[keyval], 0, False)
            hold_node.instantiate_weighting()
            self.node_dict[keyval] = hold_node
        #now instantialise inner layers:
        #we'll add one to the index of the len of the list each time to allow for the 0 indexed inputs nodes
        for i in range(len(self.layer_structure)):
            curr_layer = 1+i
            for j in range(self.layer_structure[i]):
                keyval = f"node_{curr_layer}_{j}"
                hold_node = Node.Node(self.activation_function, keyval, self.connectome_structure[keyval],0, False)
                hold_node.instantiate_weighting()
                self.node_dict[keyval] = hold_node
        # instantlise output layer
        # work out index of num_layers (since 0 indexed = len(layer_structure) + 1
        max_layer = len(self.layer_structure) + 1
        for i in range(self.num_outputs):
            keyval = f"node_{max_layer}_{i}"
            #note variation here - final is_output variable = True, hence second weighting variable = False
            hold_node = Node.Node(self.output_function, keyval, False, 0, True)
            hold_node.instantiate_weighting()
            self.node_dict[keyval] = hold_node

    # designed input : for input_val in self.norm_data_train:
    #future modification - include batch size as stopper of some kind, idk
    def forward_prop(self, input_val):
        pass