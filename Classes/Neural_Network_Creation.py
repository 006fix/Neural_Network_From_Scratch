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


class Neural_Network:
    def __init__(self, base_data_train, base_data_test, base_label_train, base_label_test,
                 layer_structure, connectome_structure, activation_function, num_inputs, num_outputs):
        #leaving these commented out for now until I know the internal structure generation works
        #self.norm_data_train = Proc_Func.normalize(base_data_train)
        #self.norm_data_test = Proc_Func.normalize(base_data_test)
        self.label_train = base_label_train
        self.label_test = base_label_test
        self.node_dict = {}
        self.layer_structure = layer_structure
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.connectome_structure = connectome_structure
        self.activation_function = activation_function

    def create_internal_structure(self):
        #instantialise the node dict:
        #storing every node as the number 5 for now until i have something to provide them true contents
        #instantalise input layer
        for i in range(self.num_inputs):
            keyval = f"node_0_{i}"
            self.node_dict[keyval] = 5
        #now instantialise inner layers:
        #we'll add one to the index of the len of the list each time to allow for the 0 indexed inputs nodes
        for i in range(len(self.layer_structure)):
            curr_layer = 1+i
            for j in range(self.layer_structure[i]):
                keyval = f"node_{curr_layer}_{j}"
                self.node_dict[keyval] = 5
        # instantlise output layer
        # work out index of num_layers (since 0 indexed = len(layer_structure) + 1
        max_layer = len(self.layer_structure) + 1
        for i in range(self.num_outputs):
            keyval = f"node_{max_layer}_{i}"
            self.node_dict[keyval] = 5


