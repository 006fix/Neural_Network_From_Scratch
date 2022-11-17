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

    #function to force reset of all nodes backpropogation data, to be run after every batch
    #also needs to run once at startup
    def reset_nodes_for_backprop(self):
        for node in self.node_dict:
            rel_node = self.node_dict[node]
            rel_node.populate_blank_dict_count()



    # designed input : for input_val in self.norm_data_train:
        #SHOULD THIS BE INSIDE THE LOOP? WILL IT AFFECT SPEED?
    #future modification - include batch size as stopper of some kind, idk

    def forward_prop(self, input_val, target_val):
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
                    contributing_node_list = rel_node.backward_nodes
                    print(f"SECONDARY STAGE - REL NODE_INCOMING_LINKS = {contributing_node_list}")
                    #we are now iterating through each node in the layer
                    #current node is held under the var 'node' (node name) and 'rel_node' (node itself)
                    #we also have a list of contributory nodes, stored as a list of node names under 'contributing node list'

                    #step 1 - find the outvals, and weights
                    #for each node in contributing node list:
                    for sending_node in contributing_node_list:
                        #find the node in the dict forward_passing_dict
                        sending_node_forward_val = forward_passing_dict[sending_node]
                        #this holds the value its passing forward

                        #find the node in the self.node_dict dict
                        #find the nodes.weight_dict variable
                        #locate the node this is going to (stored under var 'node'), and find its weight
                        sending_node_weight = self.node_dict[sending_node].weight_dict[node]

                        #multiply these values together
                        sending_node_final = sending_node_weight * sending_node_forward_val
                        print(f"I am about to modify the values for {node}. Original forward val was {sending_node_forward_val}, weight to modify with is {sending_node_weight}")
                        final_weight += sending_node_final
                    print(f"FINAL WEIGHT for node {node} = {final_weight}")
                    #apply the activation function
                    #this is held within the apply_applicable_activation function on the node in question
                    #in this case, that's the var 'rel_node'
                    #add the bias function
                    final_weight += rel_node.bias
                    print(f"I just added {rel_node.bias} to the final weight, to allow for bias addition prior to activation function")
                    modified_output = rel_node.apply_applicable_activation(final_weight)
                    print(f"MODIFIED WEIGHT FOR NODE {node}. It was {final_weight}, it is now {modified_output}")
                    #finally, add this modified output to the weight forward dict
                    forward_passing_dict[node] = modified_output
        print("forward passing dict follows")
        print(forward_passing_dict)
        #THIS COST FUNCTION WILL ONLY WORK FOR INSTANCES WITH 1 OUTPUT VARIABLE
        #find the output variable
        output_variables = []
        holdstr = f"_{self.max_layer}_"
        for key in forward_passing_dict:
            if holdstr in key:
                output_variables.append(key)
        #THIS IS THE BIT THAT WOULD HAVE TO CHANGE FOR MULTIPLE OUTPUT VARIABLES
        cost = (forward_passing_dict[output_variables[0]] - target_val)**2
        derived_val = forward_passing_dict[output_variables[0]]
        print(f"Final cost is {cost}")

        return cost, forward_passing_dict, target_val, derived_val


    def backward_prop(self, cost, forward_passing_dict, target_val, derived_val, learning_rate):
        #lets work backwards in layers since that's most intuitive
        backwards_layers = list(reversed(self.layers))
        for layer in backwards_layers:
            keyval = f"layer_{layer}"
            nodes_in_layer = self.layer_node_dict[keyval]
            print(nodes_in_layer)
            #now we have the nodes in that layer as a list
            #if this is the minimum layer, we'd skip this entirely, but for now, continue
            for node in nodes_in_layer:
                #get a variable holding the actual node
                active_node = self.node_dict[node]
                #for top row
                if layer == self.max_layer:
                    #mod bias
                    bias_mod = 2*(derived_val-target_val)
                    bias_new = active_node.bias - (learning_rate * bias_mod)
                    holdval = active_node.modification_dict['bias']
                    holdval += bias_new
                    active_node.modification_dict['bias'] = holdval
                    print(f"Old bias was 0, new bias is {holdval}")
                    #mod weights
                    #modification at this top layer = (2*(derived val - target val)) * output val of source of weight val
                    print("Backward nodes follow")
                    print(active_node.backward_nodes)
                    #now for each of these backwards nodes, we can modify their weights by finding them, finding
                        #their output val, then modifying their modification_dict with the suitable values
                    for backward_node in active_node.backward_nodes:
                        backward_node_object = self.node_dict[backward_node]
                        backward_node_outval = forward_passing_dict[backward_node]
                        weight_bias_premod = (2*(derived_val-target_val)) * backward_node_outval
                        #the above 3 provide us the object, and the premod weight bias
                        #now we need its original weight
                        orig_weight = backward_node_object.weight_dict[node]
                        new_weight = orig_weight - (learning_rate * weight_bias_premod)
                        #map it onto that subsidiary nodes weight dict
                        holdval = backward_node_object.modification_dict[node]
                        holdval += new_weight
                        backward_node_object.modification_dict[node] = holdval
                        print(f"for node {node}, we have modified the dict of {backward_node}, adding a new weight value of {holdval}, in place of the prior value of {backward_node_object.weight_dict[node]}")
        #now we've modified everything, update every single nodes modification count by 1
        for node in self.node_dict:
            hold_node = self.node_dict[node]
            prev_count = hold_node.modification_count
            prev_count += 1
            hold_node.modification_count = prev_count

    def backward_prop_generalised(self, cost, forward_passing_dict, target_val, derived_val, learning_rate):
        #lets work backwards in layers since that's most intuitive
        backwards_layers = list(reversed(self.layers))

        #ADDITION
        #hold variable to hold the generalised (2*(ypred-yact)), to save calculation cost of regenerating
        repeated_multiplier = 2*(derived_val-target_val)

        for layer in backwards_layers:
            keyval = f"layer_{layer}"
            nodes_in_layer = self.layer_node_dict[keyval]
            print(nodes_in_layer)
            #now we have the nodes in that layer as a list
            #if this is the minimum layer, we'd skip this entirely, but for now, continue
            for node in nodes_in_layer:
                #the below code should be complete, assuming that the node is the output node
                #as such, make a check on the below actual node. If yes, run the entirety of the below
                #if no, print an output simply stating this for now
                #get a variable holding the actual node
                active_node = self.node_dict[node]
                output_node = active_node.is_output
                if output_node:
                    #for top row
                    #mod bias
                    bias_mod = repeated_multiplier
                    bias_new = active_node.bias - (learning_rate * bias_mod)
                    holdval = active_node.modification_dict['bias']
                    holdval += bias_new
                    active_node.modification_dict['bias'] = holdval
                    print(f"Old bias was 0, new bias is {holdval}")
                    #mod weights
                    #modification at this top layer = (2*(derived val - target val)) * output val of source of weight val
                    print("Backward nodes follow")
                    print(active_node.backward_nodes)
                    #now for each of these backwards nodes, we can modify their weights by finding them, finding
                        #their output val, then modifying their modification_dict with the suitable values
                    for backward_node in active_node.backward_nodes:
                        backward_node_object = self.node_dict[backward_node]
                        backward_node_outval = forward_passing_dict[backward_node]
                        weight_bias_premod = repeated_multiplier * backward_node_outval
                        #the above 3 provide us the object, and the premod weight bias
                        #now we need its original weight
                        orig_weight = backward_node_object.weight_dict[node]
                        new_weight = orig_weight - (learning_rate * weight_bias_premod)
                        #map it onto that subsidiary nodes weight dict
                        holdval = backward_node_object.modification_dict[node]
                        holdval += new_weight
                        backward_node_object.modification_dict[node] = holdval
                        print(f"for node {node}, we have modified the dict of {backward_node}, adding a new weight value of {holdval}, in place of the prior value of {backward_node_object.weight_dict[node]}")
                else:
                    print(f"I'm afraid node {node} was not an output node, as such this operation has not run")
                    #NOW WE COMMENCE THE ALGORITHM
                    #we already have an algorithm above that will give us the basic generic algorithm
                    #above we have some code that will get us a generic structure
                    #in theory, this could be said to be bla * unknown var, where unknown var is all the additional variables we must derive
                    #given this, if we write code which will tell us all the additional unknown variables we must derive, we can slot this into the above
                    #and then no need of an if/else

        #now we've modified everything, update every single nodes modification count by 1
        for node in self.node_dict:
            hold_node = self.node_dict[node]
            prev_count = hold_node.modification_count
            prev_count += 1
            hold_node.modification_count = prev_count





