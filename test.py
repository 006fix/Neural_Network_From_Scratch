
import Classes.Neural_Network_Creation as NN_Creator
test_NN = NN_Creator.Neural_Network([5,5], [5,5], 1, 1,
                 [2,3,2], "dense", "relu", 5, 1)
test_NN.create_internal_structure()
holdval = test_NN.node_dict
print(holdval)