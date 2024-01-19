
# ------------ BIBLIOTECAS ------------
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from funciones import convertir_a_gris
from funciones import calcular_V
from funciones import calcular_U
from funciones import lista_pesos_a_dicc_pesos

import torch.nn.functional as F
import PIL
import gym_super_mario_bros
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

class RedNeuronalManual:
    def __init__(self, n_input,n_hidden1,n_hidden2,n_output):
        # Tamaños de las capas
        entrada_shape = n_input
        capa_oculta1_neuronas = n_hidden1
        capa_oculta2_neuronas = n_hidden2
        salida_neuronas = n_output

        # Inicialización de pesos
        self.weights1 = None
        self.bias1 = None

        self.weights2 = None
        self.bias2 = None

        self.weights3 = None
        self.bias3 = None

    def inicializar_pesos(self, entrada_neuronas, salida_neuronas):
        # Inicialización de pesos utilizando el método de Xavier/Glorot
        limit = np.sqrt(6 / (entrada_neuronas + salida_neuronas))
        return np.random.uniform(-limit, limit, (entrada_neuronas, salida_neuronas))

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoide(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Manejo de estabilidad numérica
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def forward(self, entrada, pesos):
        # Capa 1
        self.weights1 = pesos['weights1']
        self.bias1 = np.zeros((1, 40))
        capa1_salida = np.dot(entrada, self.weights1) + self.bias1
        capa1_activacion = self.relu(capa1_salida)

        # Capa 2
        self.weights2 = pesos['weights2']
        self.bias2 = np.zeros((1, 20))
        capa2_salida = np.dot(capa1_activacion, self.weights2) + self.bias2
        capa2_activacion = self.relu(capa2_salida)

        # Capa de salida
        self.weights3 = pesos['weights3']
        self.bias3 = np.zeros((1, 12))
        salida = np.dot(capa2_activacion, self.weights3) + self.bias3
        salida_activacion = self.softmax(salida)

        return salida_activacion
