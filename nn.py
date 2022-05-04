import numpy as np

from network import Network
from fc import FCLayer
from activation_layer import ActivationLayer
from utils import tanh, d_tanh


class NN:
    def __init__(self):
        pass

    # build network
    def build_network():
        net = Network()
        net.add(FCLayer(2, 3))
        net.add(ActivationLayer(tanh, d_tanh))
        net.add(FCLayer(3, 1))
        net.add(ActivationLayer(tanh, d_tanh))
        return net

