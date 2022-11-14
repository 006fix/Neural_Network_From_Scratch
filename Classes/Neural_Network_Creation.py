
class Neural_Network(base_data_train, base_data_test, architecture, activation, num_labels, num_features, layer_size_list):

    def __init__(self):
        import Function_Library.Preprocessing_Functions as Proc_Func
        self.norm_data_train = Proc_Func.normalize(base_data_train)
        self.norm_data_test = Proc_Func.normalize(base_data_test)
        self.layers = {}
        self.architecture = architecture
        self.activation = activation
        self.paramaters = {}
        self.num_labels = num_labels
        self.num_features = num_features
        self.architecture.append(self.num_labels)
        self.architecture.insert(0, self.num_features)
        self.L = len(architecture)

        def initialize_parameters(self):
            for i in range(1, self.L):
                print(f"Initializing parameters for layer: {i}.")
                self.parameters["w" + str(i)] = np.random.randn(self.architecture[i], self.architecture[i - 1]) * 0.01
                self.parameters["b" + str(i)] = np.zeros((self.architecture[i], 1))

        initialize_parameters(self)



