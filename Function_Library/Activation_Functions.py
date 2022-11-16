
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def leaky_relu(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, z, z * 0.01)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - np.max(z))
    return e / np.sum(e, axis=0)

def derivative(function_name: str, z: np.ndarray) -> np.ndarray:
    error_str = "No such activation"
    if function_name == "sigmoid":
        return sigmoid(z) * (1 - sigmoid(z))
    elif function_name == "tanh":
        return 1 - np.square(tanh(z))
    elif function_name == "relu":
        y = (z > 0) * 1
        return y
    elif function_name == "leaky_relu":
        return  np.where(z > 0, 1, 0.01)
    else:
        return error_str
