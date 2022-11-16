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
        #THESE AREN'T WORKING, FIX LATER
        #self.norm_data_train = Proc_Func.normalize(base_data_train)
        #self.norm_data_test = Proc_Func.normalize(base_data_test)
        self.norm_data_train = base_data_train
        self.norm_data_test = base_data_test
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
        #dictionary to store nodes in each layer, saves recalculating later
        self.layer_node_dict = {}

    #because this will now check the dict of nodes, needs to run once those are populated
    def populate_layers(self):
        #initial part of function gives us details on layers
        for i in range(len(self.layer_structure)):
            holdval = i+1
            self.layers.append(holdval)
        #find max val, add layer 1 above this (output layer
        holdval2 = max(self.layers)
        holdval2 += 1
        self.max_layer = holdval2
        self.layers.append(holdval2)
        #secondary part below gives us a dict containing each node at each layer
        for i in self.layers:
            for key in self.node_dict:
                checkstr = f"_{i}_"
                keystr = checkstr.replace("_","")
                newkeystr = f"layer_{keystr}"
                if checkstr in key:
                    try:
                        holdval = self.layer_node_dict[newkeystr]
                        holdval.append(key)
                        self.layer_node_dict[newkeystr] = holdval
                    except:
                        holdval = [key]
                        self.layer_node_dict[newkeystr] = holdval




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

    #new function to provide each node with a list of nodes it passes values forward to
    def provide_nodes_forward(self):
        #look through each node
        for node in self.node_dict:
            #if we examine self.node_dict[node].weight_dict, we get a list of keys
            #these keys correspond to nodes that this nodes contributes to
            #as such, if we provide these keys to the forward_nodes of this dict, we will get the forward nodes
            rel_dict = self.node_dict[node].weight_dict
            for key in rel_dict:
                self.node_dict[node].forward_nodes.append(key)

    #new function to provide each node with a list of nodes it receives values from
    def provide_nodes_backward(self):
        #look through each nodes
        for node in self.node_dict:
            #for each of these nodes, look at the weight dict:
            for node2 in self.node_dict[node].weight_dict:
                #for each node2 in these node dicts, find that node2, update its backwards
                #to include node2 in its backwards dict
                self.node_dict[node2].backward_nodes.append(node)





    # designed input : for input_val in self.norm_data_train:
        #SHOULD THIS BE INSIDE THE LOOP? WILL IT AFFECT SPEED?
    #future modification - include batch size as stopper of some kind, idk

    def forward_prop(self, input_val):
        print(f"input val is {input_val}")
        #dict to store values passing forward from each node, used to determine contributory nodes
        forward_passing_dict = {}
        #start at layer 0, add flat inputs
        #then iterate for remaining layers:
        for key in self.layer_node_dict:
            if key == 'layer_0':
                holdval = self.layer_node_dict[key]
                for val in range(len(holdval)):
                    node_result = self.layer_node_dict[key][val]
                    forward_passing_dict[node_result] = input_val[val]
                    print(f" having parsed node {node_result}, dict = {forward_passing_dict}")
            #all remaining nodes
            #add sum of weight*value together, apply relevant function
            else:
                #take each node in this layer
                holdval = self.layer_node_dict[key]
                print(f"SECONDARY STAGE - HOLDVAL = {holdval}")
                #below will be each individual node in the layer
                for node in holdval:
                    print(f"SECONDARY STAGE - ACTIVE NODE = {node}")
                    # initialise value to hold the final result
                    final_weight = 0
                    rel_node = self.node_dict[node]
                    print(f"SECONDARY STAGE - REL NODE = {rel_node.name}")
                    contributing_node_dict = rel_node.weight_dict
                    print(f"SECONDARY STAGE - REL NODE_WEIGHTS = {contributing_node_dict}")
                    #now look at each node in this dict, multiply by the weight value, add together
                    for contr_node in contributing_node_dict:
                        node_weight = contributing_node_dict[contr_node]
                        node_send_val = forward_passing_dict[contr_node]
                        fin_val = node_weight * node_send_val
                        final_weight += fin_val
                    #now we have the sum of all final weights, apply the activation_function
                    final_weight = node.apply_applicable_activation(final_weight)
                    forward_passing_dict[node] = final_weight
        return forward_passing_dict


