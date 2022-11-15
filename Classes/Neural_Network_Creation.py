import numpy as np
import Function_Library.Activation_Functions as Act_Func
from tqdm import tqdm


class Neural_Network:

    def __init__(self, base_data_train, base_data_test, base_label_train, base_label_test, architecture, activation, num_labels, num_features):
        import Function_Library.Preprocessing_Functions as Proc_Func
        self.norm_data_train = Proc_Func.normalize(base_data_train)
        self.norm_data_test = Proc_Func.normalize(base_data_test)
        self.label_train = base_label_train
        self.label_test = base_label_test
        self.layers = {}
        self.architecture = architecture
        self.activation = activation
        self.parameters = {}
        self.m = self.norm_data_train.shape[1]
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

    def predict(self, base_data_train):
        params = self.parameters
        n_layers = self.L - 1
        values = [x]
        for l in range(1, n_layers):
            z = np.dot(params["w" + str(l)], values[l - 1]) + params["b" + str(l)]
            a = eval(self.activation)(z)
            values.append(a)
        z = np.dot(params["w" + str(n_layers)], values[n_layers - 1]) + params["b" + str(n_layers)]
        a = Act_Func.softmax(z)
        if x.shape[1] > 1:
            ans = np.argmax(a, axis=0)
        else:
            ans = np.argmax(a)
        return ans


    def accuracy(self, base_data_train, label_train):
        P = self.predict(base_data_train)
        return sum(np.equal(P, np.argmax(label_train, axis=0))) / label_train.shape[1] * 100

    def forward(self):
        params = self.parameters
        self.layers["a0"] = self.norm_data_train
        for l in range(1, self.L - 1):
            self.layers["z" + str(l)] = np.dot(params["w" + str(l)],
                                               self.layers["a" + str(l - 1)]) + params["b" + str(l)]
            print(self.activation)
            print(self.activation(self.layers["z" + str(l)]))
            self.layers["a" + str(l)] = self.activation(self.layers["z" + str(l)])
            assert self.layers["a" + str(l)].shape == (self.architecture[l], self.m)
        self.layers["z" + str(self.L - 1)] = np.dot(params["w" + str(self.L - 1)],
                                                    self.layers["a" + str(self.L - 2)]) + params["b" + str(self.L - 1)]
        self.layers["a" + str(self.L - 1)] = Act_Func.softmax(self.layers["z" + str(self.L - 1)])
        self.output = self.layers["a" + str(self.L - 1)]
        assert self.output.shape == (self.num_labels, self.m)
        assert all([s for s in np.sum(self.output, axis=1)])

        cost = - np.sum(self.norm_data_train * np.log(self.output + 0.000000001))

        return cost, self.layers

    def backpropagate(self):
        derivatives = {}
        dZ = self.output - self.label_train
        assert dZ.shape == (self.num_labels, self.m)
        dW = np.dot(dZ, self.layers["a" + str(self.L - 2)].T) / self.m
        db = np.sum(dZ, axis=1, keepdims=True) / self.m
        dAPrev = np.dot(self.parameters["w" + str(self.L - 1)].T, dZ)
        derivatives["dW" + str(self.L - 1)] = dW
        derivatives["db" + str(self.L - 1)] = db

        for l in range(self.L - 2, 0, -1):
            dZ = dAPrev * Act_Func.derivative(self.activation, self.layers["z" + str(l)])
            dW = 1. / self.m * np.dot(dZ, self.layers["a" + str(l - 1)].T)
            db = 1. / self.m * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = np.dot(self.parameters["w" + str(l)].T, (dZ))
            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db
        self.derivatives = derivatives

        return self.derivatives

    def fit(self, epochs, lr=0.01):
        self.costs = []
        self.initialize_parameters()
        self.accuracies = {"train": [], "test": []}
        for epoch in tqdm(range(epochs), colour="BLUE"):
            cost, cache = self.forward()
            self.costs.append(cost)
            derivatives = self.backpropagate()
            for layer in range(1, self.L):
                self.parameters["w" + str(layer)] = self.parameters["w" + str(layer)] - lr * derivatives["dW" + str(layer)]
                self.parameters["b" + str(layer)] = self.parameters["b" + str(layer)] - lr * derivatives["db" + str(layer)]
            train_accuracy = self.accuracy(self.norm_data_train, self.label_train)
            test_accuracy = self.accuracy(self.norm_data_test, self.label_test)
            if epoch % 2 == 0:
                print(f"Epoch: {epoch:3d} | Cost: {cost:.3f} | Accuracy: {train_accuracy:.3f}")
            self.accuracies["train"].append(train_accuracy)
            self.accuracies["test"].append(test_accuracy)
        print("Training terminated")