
import Classes.Neural_Network_Creation as NN_Creator
test_NN = NN_Creator.Neural_Network([0.02, 0.5], [5,5], 3, 3, [5,5], "dense", "tanh", 2, 1, 'linear')
test_NN.connectome_validity()
test_NN.create_internal_structure()
test_NN.populate_layers()
test_NN.provide_nodes_forward()
test_NN.provide_nodes_backward()
test_NN.reset_nodes_for_backprop()
holdval = test_NN.node_dict
#dict looks to be working as intended, test below
#print(holdval)
holdval2 = test_NN.connectome_structure
#dense connectome fix appeasr to be working as intended, test below
print(holdval2)
#node dict appears to be working well, holds a good set of nodes, that are functional instantiations of the node class
holdval3 = test_NN.node_dict
#print(holdval3)
#appears to be a valid test of a connection, shows the node contains two weightings, each of which are different, is not outut
#done using input NN_Creator.Neural_Network([5,5], [5,5], 1, 1, [2,3], "dense", "relu", 2, 2)
holdval4 = test_NN.node_dict['node_0_0']
#print(holdval4.weight_dict)
#print(holdval4.is_output)
#appears to be a valid test of connection, shows the node (output) contains two no weightings, is output
#done using input NN_Creator.Neural_Network([5,5], [5,5], 1, 1, [2,3], "dense", "relu", 2, 2)
holdval5 = test_NN.node_dict['node_1_0']
print("WEIGHT DICT FOLLOWS")
print(holdval5.weight_dict)
#print(holdval5.is_output)
#works as intended, provides a suitable number for max layers
holdval6 = test_NN.max_layer
#print(holdval6)
#works as intended, provides a suitable list of valid layers
holdval7 = test_NN.layers
#print(holdval7)
#works as intended, provides a valid list of all nodes
holdval8 = test_NN.layer_node_dict
#print(holdval8)
#test confirmed, provides accurate validity of which node contributes to this node
holdval9 = test_NN.node_dict['node_1_0']
print("FOWARD NODES FOLLOW")
print(holdval9.forward_nodes)
holdval10 = test_NN.node_dict['node_1_0']
print("backwards_nodes follow")
print(holdval10.backward_nodes)


print("COMMENCING SECONDARY STAGE TESTING")
print("COMMENCING SECONDARY STAGE TESTING")
print("COMMENCING SECONDARY STAGE TESTING")
print("COMMENCING SECONDARY STAGE TESTING")
#NEW TEST SEQUENCE - NOW WE PROVIDE OUR INPUT VALUES INTO OUR FORWARD PROPOGATION ALGORITHM
cost, forward_passing_dict, target_val, derived_val = test_NN.forward_prop(test_NN.norm_data_train, test_NN.label_train)

print("COMMENCING TERTIARY STAGE TESTING")
print("COMMENCING TERTIARY STAGE TESTING")
print("COMMENCING TERTIARY STAGE TESTING")
print("COMMENCING TERTIARY STAGE TESTING")
#triggering the backprop algorithm
print(holdval10.modification_count)
test_NN.backward_prop_generalised(cost, forward_passing_dict, target_val, derived_val, 0.01)
print(holdval10.modification_count)
