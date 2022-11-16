
import Classes.Neural_Network_Creation as NN_Creator
test_NN = NN_Creator.Neural_Network([5,6], [5,6], 1, 1, [6,3], "dense", "relu", 2, 1, "relu")
test_NN.populate_layers()
test_NN.connectome_validity()
test_NN.create_internal_structure()
holdval = test_NN.node_dict
#dict looks to be working as intended, test below
print(holdval)
holdval2 = test_NN.connectome_structure
#dense connectome fix appeasr to be working as intended, test below
print(holdval2)
#node dict appears to be working well, holds a good set of nodes, that are functional instantiations of the node class
holdval3 = test_NN.node_dict
#print(holdval3)
#appears to be a valid test of a connection, shows the node contains two weightings, each of which are different, is not outut
#done using input NN_Creator.Neural_Network([5,5], [5,5], 1, 1, [2,3], "dense", "relu", 2, 2)
#holdval4 = test_NN.node_dict['node_0_1']
#print(holdval4.weight_dict)
#print(holdval4.is_output)
#appears to be a valid test of connection, shows the node (output) contains two no weightings, is output
#done using input NN_Creator.Neural_Network([5,5], [5,5], 1, 1, [2,3], "dense", "relu", 2, 2)
#holdval5 = test_NN.node_dict['node_3_1']
#print(holdval5.weight_dict)
#print(holdval5.is_output)
holdval6 = test_NN.max_layer
print(holdval6)
holdval7 = test_NN.layers
print(holdval7)
