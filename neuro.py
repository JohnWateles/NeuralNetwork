import copy
from time import perf_counter
import ctypes
from typing import Callable

import numpy as np


def show_execution_time(func: Callable):
    def wrapper(*args, **kwargs):
        _start = perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {perf_counter() - _start}")
        return result
    return wrapper


class _NeuronStructure(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_double),
        ("before_activation", ctypes.c_double),
        ("bias", ctypes.c_double),
        ("weights", ctypes.POINTER(ctypes.c_double))
    ]


class _LayerStructure(ctypes.Structure):
    _fields_ = [
        ("len", ctypes.c_uint64),
        ("func", ctypes.c_void_p),
        ("derivate_func", ctypes.c_void_p),
        ("neurons", ctypes.POINTER(_NeuronStructure))
    ]


class _NeuralNetworkStructure(ctypes.Structure):
    _fields_ = [
        ("countLayers", ctypes.c_uint64),
        ("layers", ctypes.POINTER(ctypes.POINTER(_LayerStructure)))
    ]


clib = ctypes.WinDLL("./neuro.dll")

clib.createNeuralNetwork.argtype = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_char_p]
clib.createNeuralNetwork.restype = ctypes.POINTER(_NeuralNetworkStructure)

clib.destroyNeuralNetwork.argtype = [ctypes.POINTER(_NeuralNetworkStructure)]
clib.destroyNeuralNetwork.restype = ctypes.c_int

clib.setInputValues.argtype = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(_NeuralNetworkStructure)]
clib.setInputValues.restype = ctypes.c_int

clib.getOutputValues.argtype = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(_NeuralNetworkStructure)]
clib.getOutputValues.restype = ctypes.c_int

clib.getOutputValuesSoftMax.argtype = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(_NeuralNetworkStructure)]
clib.getOutputValuesSoftMax.restype = ctypes.c_int

clib.getLengthInputLayer.argtype = [ctypes.POINTER(_NeuralNetworkStructure)]
clib.getLengthInputLayer.restype = ctypes.c_uint64

clib.getLengthOutputLayer.argtype = [ctypes.POINTER(_NeuralNetworkStructure)]
clib.getLengthOutputLayer.restype = ctypes.c_uint64

clib.forwardPropagation.argtype = [ctypes.POINTER(_NeuralNetworkStructure)]

clib.backwardPropagation.argtype = [ctypes.POINTER(_NeuralNetworkStructure),
                                    ctypes.POINTER(ctypes.c_double), ctypes.c_double]

clib.printNeuralNetwork.argtype = [ctypes.POINTER(_NeuralNetworkStructure)]

clib.saveNeuralNetwork.argtype = [ctypes.c_char_p, ctypes.POINTER(_NeuralNetworkStructure)]

clib.loadNeuralNetwork.argtype = [ctypes.c_char_p]
clib.loadNeuralNetwork.restype = ctypes.POINTER(_NeuralNetworkStructure)


class NeuralNetwork:
    def __init__(self, layers_lengths: list[int] | tuple[int], function: str = "sigmoid"):
        function = function.lower()
        __layers_length = copy.copy(layers_lengths)
        __layers_length.insert(0, len(__layers_length))
        _layers_lengths = ctypes.c_uint64 * len(__layers_length)
        _layers_lengths = _layers_lengths(*__layers_length)
        self._struct = clib.createNeuralNetwork(_layers_lengths, function.encode())

    def print(self):
        clib.printNeuralNetwork(self._struct)

    def set_input_value(self, values):
        _values = ctypes.c_double * len(values)
        _values = _values(*values)
        clib.setInputValues(_values, self._struct)

    def get_output_values(self):
        _values = (ctypes.c_double * self.len_output_layer)()
        clib.getOutputValues(_values, self._struct)
        values = np.array(_values)
        return values

    def get_output_values_softmax(self):
        _values = (ctypes.c_double * self.len_output_layer)()
        clib.getOutputValuesSoftMax(_values, self._struct)
        values = np.array(_values)
        return values

    def forward_propagation(self):
        clib.forwardPropagation(self._struct)

    def backward_propagation(self, y: list[float] | tuple[float], learning_rate=0.005):
        if len(y) != self.len_output_layer:
            raise ValueError("Check length output layer")
        _y = ctypes.c_double * len(y)
        _y = _y(*y)
        clib.backwardPropagation(self._struct, _y, ctypes.c_double(learning_rate))

    def save(self, filename: str):
        if "." not in filename:
            filename += ".nn"
        clib.saveNeuralNetwork(filename.encode(), self._struct)

    @property
    def len_input_layer(self):
        return clib.getLengthInputLayer(self._struct)

    @property
    def len_output_layer(self):
        return clib.getLengthOutputLayer(self._struct)

    def __del__(self):
        clib.destroyNeuralNetwork(self._struct)


def load_neural_network(filename: str) -> NeuralNetwork:
    if "." not in filename:
        filename += ".nn"
    new_object = NeuralNetwork.__new__(NeuralNetwork)
    new_object._struct = clib.loadNeuralNetwork(filename.encode())
    return new_object
